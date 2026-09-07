"""Train a small generated-audio correction LoRA on frozen released VDN/Turbo.

The 33B H3 base, VDN branch, Stage-B LoRA and finished Stage-DMD Turbo LoRA are frozen.
Only ``audio_fix`` sidecar A/B matrices are trainable. The sidecar is applied directly
only to generated-audio rows of the packed DiT sequence; a video-preservation loss also
protects against indirect audio->video feedback through later dense attention.

Example:

  torchrun --standalone --nproc_per_node=1 src/training/train_audio_fix.py \
    --config configs/training/audio_fix_res10.yaml \
    data.index_file=/path/to/video_index.jsonl distributed.shard_size=1
"""
from __future__ import annotations

import gc
import glob
import json
import math
import os
import re
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    align_num_frames,
    audio_latent_num_frames,
    video_latent_num_frames,
)
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree

from src.checkpoints import load_checkpoint
from src.config import load_config, resolved_dict
from src.config.audio_fix import AudioFixConfig, validate_audio_fix
from src.models.factory import build_model, inject_adapters, load_model_weights
from src.models.hybrid_transform import set_layout, set_softmax_backend
from src.paths import resolve_weights
from src.training import dmd
from src.training import fsdp_stage as fs
from src.training.audio_fix import (
    ResMultistepFewStepSchedule,
    dense_audio_target,
    frozen_production_target,
    rollout,
    set_production_role,
)
from src.training.audio_fix_adapter import (
    AudioFixAdapterBank,
    audio_fix_spec,
    average_gradients,
    broadcast_parameters,
    export_composed_stage,
    require_unit_lora_scale,
)
from src.training.t2va_batch import PackedPrompt, initial_noise
from src.utils import run_lock

STAGE = "audio_fix"
WEIGHTS_PREFIX = "audio-fix-step-"
LATENT_CHANNELS = 24


def _source_step(path, artifact):
    """Recover a source step from structured metadata or a conventional path name."""
    candidates = [artifact.step, artifact.metadata.get("step"), path,
                  artifact.metadata.get("converted_from")]
    for value in candidates:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            match = re.search(r"(?:step[-_]?)(\d+)", value)
            if match:
                return int(match.group(1))
    return None


def validate_source(cfg, path, artifact):
    """Validate that the seed is exactly the released canonical step-250 stack."""
    step = _source_step(path, artifact)
    if step != cfg.initialization.source_step:
        raise RuntimeError(
            f"audio-fix requires released step {cfg.initialization.source_step}, got {step}")
    spec = artifact.model_spec
    require_unit_lora_scale(spec, cfg.initialization.vdn_adapter_name)
    turbo = require_unit_lora_scale(spec, cfg.initialization.turbo_adapter_name)
    if not turbo.get("config", {}).get("exact_targets"):
        raise RuntimeError("released Turbo adapter must carry exact_targets")
    names = []
    for i, entry in enumerate(spec.get("adapters", [])):
        names.append(entry.get("config", {}).get("name", "default" if i == 0 else None))
    if cfg.adapter.name in names:
        raise RuntimeError(f"source checkpoint already contains {cfg.adapter.name!r}")
    return spec


def generation_geometry(generation):
    """Resolve the configured decoded geometry to H3 video/audio latent dimensions."""
    frames = align_num_frames(generation.num_frames, 17, 5)
    num_latent_frames = video_latent_num_frames(frames, 17, 5)
    num_audio_latents = audio_latent_num_frames(frames)
    return ((LATENT_CHANNELS, num_latent_frames,
             generation.latent_height, generation.latent_width), int(num_audio_latents))


def _tree_cpu(value):
    """Recursively move optimizer/checkpoint tensor state to CPU before serialization."""
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _tree_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_tree_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_tree_cpu(v) for v in value)
    return value


def _save_train_state(output_dir, step, epoch, in_epoch, iteration, bank, optimizer,
                      noise_generator, rank, world, cfg):
    """Save sidecar, optimizer, dataset cursor and per-rank RNG state on rank zero."""
    rng = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
        "noise": noise_generator.get_state().cpu(),
    }
    all_rng = [None] * world
    dist.all_gather_object(all_rng, rng)
    if rank != 0:
        return
    state = {
        "format": 1,
        "stage": STAGE,
        "step": int(step),
        "epoch": int(epoch),
        "in_epoch": int(in_epoch),
        "iteration": int(iteration),
        "world_size": int(world),
        "bank": _tree_cpu(bank.state_dict()),
        "optimizer": _tree_cpu(optimizer.state_dict()),
        "rng_per_rank": all_rng,
        "resolved_config": resolved_dict(cfg),
    }
    path = os.path.join(output_dir, f"train_state_step{step:06d}.pt")
    torch.save(state, path)
    stale = sorted(glob.glob(os.path.join(output_dir, "train_state_step*.pt")))[
        :-cfg.checkpoint.keep_states]
    for old in stale:
        os.remove(old)
    print(f"[step {step}] audio-fix train state -> {path}", flush=True)


def _resume(output_dir, bank, optimizer, noise_generator, resumable_sampler, rank, world):
    """Restore the newest compatible train state and deterministic data/RNG position."""
    states = sorted(glob.glob(os.path.join(output_dir, "train_state_step*.pt")))
    if not states:
        return 0, 0, 0, 0
    # Train state is intentionally tensors + primitive containers only. Restricted
    # loading prevents a replaced local resume file from executing arbitrary pickle code
    # before the format/stage checks below can reject it.
    state = torch.load(states[-1], map_location="cpu", weights_only=True)
    if state.get("format") != 1 or state.get("stage") != STAGE:
        raise RuntimeError(f"unsupported audio-fix train state {states[-1]}")
    bank.load_state_dict(state["bank"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    optimizer.train()
    saved_world = int(state.get("world_size", -1))
    epoch, in_epoch = int(state["epoch"]), int(state["in_epoch"])
    if saved_world == world:
        rng = state["rng_per_rank"][rank]
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"])
        noise_generator.set_state(rng["noise"])
        resumable_sampler.skip = in_epoch
    else:
        epoch, in_epoch = int(state["step"]), 0
        if rank == 0:
            print(f"resume world changed {saved_world}->{world}; reset data cursor/RNG", flush=True)
    if rank == 0:
        print(f"auto-resumed audio-fix step {state['step']} from {states[-1]}", flush=True)
    return int(state["step"]), epoch, in_epoch, int(state.get("iteration", state["step"]))


def _grad_norm(parameters):
    """Return the un-clipped global L2 norm for logging only."""
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().pow(2).sum())
    return math.sqrt(total)


def _optimizer(cfg, bank):
    """Build the pinned conservative Prodigy-Plus Schedule-Free baseline."""
    # Prodigy-Plus recommends lr=1 with Schedule-Free, a constant external scheduler,
    # and no external gradient clipping when StableAdamW is enabled. We follow that
    # contract exactly; experimental SPEED/FOCUS/OrthoGrad variants stay off.
    return ProdigyPlusScheduleFree(
        bank.parameters(),
        lr=cfg.optimizer.lr,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.weight_decay,
        d0=cfg.optimizer.d0,
        d_coef=cfg.optimizer.d_coef,
        d_limiter=cfg.optimizer.d_limiter,
        prodigy_steps=cfg.optimizer.prodigy_steps,
        schedulefree_c=cfg.optimizer.schedulefree_c,
        eps=cfg.optimizer.eps,
        split_groups=cfg.optimizer.split_groups,
        factored=cfg.optimizer.factored,
        factored_fp32=cfg.optimizer.factored_fp32,
        use_stableadamw=cfg.optimizer.use_stableadamw,
        use_schedulefree=cfg.optimizer.use_schedulefree,
        use_speed=False,
        stochastic_rounding=False,
        fused_back_pass=False,
        use_cautious=False,
        use_grams=False,
        use_adopt=False,
        use_orthograd=False,
        use_focus=False,
    )


def run(cfg):
    """Execute the distributed on-policy audio-fix optimization loop."""
    output_dir = cfg.checkpoint.output_dir
    max_steps = cfg.training.max_steps
    final_dir = os.path.join(output_dir, f"{WEIGHTS_PREFIX}{max_steps:06d}")
    fs.refuse_if_guarded(output_dir, final_dir, cfg.training.ignore_stopped)

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    torch.cuda.set_per_process_memory_fraction(cfg.distributed.memory_fraction)
    torch.manual_seed(cfg.training.seed)

    source_path = resolve_weights(cfg.initialization.checkpoint)
    source_artifact = load_checkpoint(source_path)
    source_spec = validate_source(cfg, source_path, source_artifact)
    resolved_spec, adapter_cfg = audio_fix_spec(
        source_spec,
        name=cfg.adapter.name,
        rank=cfg.adapter.rank,
        alpha=cfg.adapter.alpha,
        source_adapter=cfg.initialization.turbo_adapter_name,
    )
    targets = adapter_cfg["targets"]
    if rank == 0:
        fs.print_architecture(source_spec, label="frozen released VDN/Turbo architecture")
        print(f"audio-fix targets: {len(targets)} sequence-space linears, "
              f"rank {cfg.adapter.rank}, alpha {cfg.adapter.alpha}", flush=True)

    model = build_model(source_spec, device="cpu", base_source=cfg.initialization.base_source)
    model = inject_adapters(model, source_spec)
    loaded = load_model_weights(model, source_artifact.weights)
    model.requires_grad_(False)
    set_softmax_backend(model, cfg.runtime.kernels.softmax_backend)
    set_production_role(
        model, cfg.initialization.vdn_adapter_name, cfg.initialization.turbo_adapter_name)

    bank = AudioFixAdapterBank(
        model, targets, cfg.adapter.rank, cfg.adapter.alpha, name=cfg.adapter.name)
    hook_count = bank.install(model)
    del source_artifact
    gc.collect()

    model = fs.shard_model(
        model, world, cfg.distributed.shard_size, device, rank,
        activation_checkpointing=cfg.distributed.activation_checkpointing)
    bank.to(device)
    broadcast_parameters(bank)
    torch.cuda.empty_cache()

    optimizer = _optimizer(cfg, bank)
    optimizer.train()
    parameters = list(bank.parameters())
    if rank == 0:
        total = sum(p.numel() for p in model.parameters())
        print(f"frozen DiT: {total/1e9:.1f}B params; loaded {loaded} checkpoint tensors", flush=True)
        print(f"audio_fix: {bank.parameter_count/1e6:.1f}M fp32 trainable params, "
              f"{hook_count} scoped hooks; Prodigy-Plus Schedule-Free", flush=True)
        print(f"saved-tensor CPU offload: {cfg.distributed.offload_activations}", flush=True)

    video_shape, num_audio_latents = generation_geometry(cfg.generation)
    aligned_frames = align_num_frames(cfg.generation.num_frames, 17, 5)
    dataset, distributed_sampler, resumable_sampler, data_loader = fs.make_loader(
        cfg.data.index_file, cfg.data.num_workers, world, rank, cfg.training.seed,
        text_only=True)
    if rank == 0:
        print(f"dataset: {len(dataset)} prompts (text only), generating {video_shape} "
              f"video rows + {num_audio_latents} audio latents", flush=True)
        metrics_path = os.path.join(output_dir, "metrics.jsonl")

    noise_generator, _unused = fs.make_generators(cfg.training.seed, rank, device)
    step = epoch = in_epoch = iteration = 0
    if cfg.training.auto_resume:
        step, epoch, in_epoch, iteration = _resume(
            output_dir, bank, optimizer, noise_generator, resumable_sampler, rank, world)
    schedule = ResMultistepFewStepSchedule(
        cfg.rollout.num_steps, cfg.rollout.video_shift, cfg.rollout.audio_shift)

    deploy_dtype = getattr(torch, cfg.checkpoint.adapter_dtype)
    early_saves = set(cfg.checkpoint.early_saves)

    def save_deployment(current_step):
        """Export Schedule-Free eval weights as a directly loadable VDN stage."""
        optimizer.eval()
        try:
            state = bank.export_peft_state(deploy_dtype) if rank == 0 else None
            if rank == 0:
                out = os.path.join(output_dir, f"{WEIGHTS_PREFIX}{current_step:06d}")
                export_composed_stage(
                    source_path, out, resolved_spec, adapter_cfg, state,
                    metadata={
                        "stage": STAGE,
                        "step": int(current_step),
                        "source_checkpoint": cfg.initialization.checkpoint,
                        "source_step": cfg.initialization.source_step,
                        "stage_b_strength": 1.0,
                        "turbo_strength": 1.0,
                        "rollout_sampler": cfg.rollout.sampler,
                        "sigma_schedule": cfg.rollout.sigma_schedule,
                        "rollout_num_steps": cfg.rollout.num_steps,
                        "video_shift": cfg.rollout.video_shift,
                        "audio_shift": cfg.rollout.audio_shift,
                        "requested_frames": cfg.generation.num_frames,
                        "aligned_frames": int(aligned_frames),
                        "video_latent_frames": int(video_shape[1]),
                        "audio_latent_frames": int(num_audio_latents),
                        "optimizer": "prodigy-plus-schedule-free",
                        "scope": "generated_audio",
                        "adapter_dtype": cfg.checkpoint.adapter_dtype,
                    },
                )
                print(f"[step {current_step}] deployable VDN+Turbo+audio_fix -> {out}", flush=True)
        finally:
            optimizer.train()

    if 0 in early_saves and step == 0:
        save_deployment(0)

    model.train()
    bank.train()
    offload = cfg.distributed.offload_activations
    t_log = time.time()

    try:
        while step < max_steps:
            distributed_sampler.set_epoch(epoch)
            for sample in data_loader:
                in_epoch += 1
                fs.phase_reset()
                torch.cuda.reset_peak_memory_stats()
                with fs.phase("data"):
                    packed = PackedPrompt(
                        sample["prompt_embeds"], sample["text_token_tags"],
                        video_shape, num_audio_latents, device)
                    set_layout(model, packed.layout)
                fs._PHASE["seq_len"] = float(packed.seq_len)

                index = dmd.shared_rollout_index(cfg.training.seed, iteration, schedule.num_steps)
                iteration += 1
                video_rows, audio_rows = initial_noise(
                    video_shape, num_audio_latents, noise_generator, device)

                with fs.phase("rollout"):
                    video_rows, audio_rows = rollout(
                        model, bank, schedule, packed, video_rows, audio_rows, index,
                        cfg.initialization.vdn_adapter_name,
                        cfg.initialization.turbo_adapter_name)
                    torch.cuda.empty_cache()

                with fs.phase("dense_audio_teacher"):
                    dense_a = dense_audio_target(
                        model, bank, schedule, packed, video_rows, audio_rows, index)
                    torch.cuda.empty_cache()

                with fs.phase("production_baseline"):
                    frozen_v, frozen_a = frozen_production_target(
                        model, bank, schedule, packed, video_rows, audio_rows, index,
                        cfg.initialization.vdn_adapter_name,
                        cfg.initialization.turbo_adapter_name)
                    baseline_audio_gap = torch.nn.functional.mse_loss(frozen_a, dense_a)
                    torch.cuda.empty_cache()

                set_production_role(
                    model, cfg.initialization.vdn_adapter_name,
                    cfg.initialization.turbo_adapter_name)
                optimizer.zero_grad(set_to_none=True)
                with bank.scope(packed.audio_indices, enabled=True):
                    with fs.phase("student_forward"):
                        t_v, t_a = schedule.times(index)
                        native_audio = schedule.native_audio(index, audio_rows)
                        velocity_v, velocity_a = fs.student_forward(
                            model,
                            packed.inputs(video_rows, native_audio, t_v, t_a),
                            offload,
                        )
                        student_v, student_a = schedule.x0(
                            index, video_rows, audio_rows,
                            velocity_v[0].float(), velocity_a[0].float())
                        audio_loss = torch.nn.functional.mse_loss(student_a, dense_a)
                        video_loss = torch.nn.functional.mse_loss(student_v, frozen_v)
                        loss = (cfg.training.audio_teacher_weight * audio_loss
                                + cfg.training.video_preserve_weight * video_loss)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"non-finite audio-fix loss at step {step + 1}, grid {index}")
                    with fs.phase("backward"):
                        loss.backward()

                average_gradients(bank, world)
                grad_norm = _grad_norm(parameters)
                with fs.phase("optim"):
                    optimizer.step()
                step += 1

                values = torch.tensor([
                    float(loss.detach()), float(audio_loss.detach()), float(video_loss.detach()),
                    float(baseline_audio_gap.detach()), float(grad_norm), float(index),
                ], device=device)
                dist.all_reduce(values, op=dist.ReduceOp.AVG)
                group = optimizer.param_groups[0]
                d_value = float(group.get("d", cfg.optimizer.d0))
                effective_lr = float(group.get("effective_lr", group["lr"])) * d_value
                elapsed = time.time() - t_log
                peak = torch.cuda.max_memory_allocated() / 2**30
                if rank == 0:
                    row = {
                        "step": step,
                        "loss": float(values[0]),
                        "audio_teacher_loss": float(values[1]),
                        "video_preserve_loss": float(values[2]),
                        "frozen_audio_dense_gap": float(values[3]),
                        "grad_norm": float(values[4]),
                        "rollout_index": int(round(float(values[5]))),
                        "prodigy_d": d_value,
                        "effective_lr": effective_lr,
                        "seconds": elapsed,
                        "peak_gib": peak,
                        **fs.phase_fields(),
                    }
                    print(
                        f"[step {step}/{max_steps}] loss={row['loss']:.5f} "
                        f"audio={row['audio_teacher_loss']:.5f} "
                        f"video={row['video_preserve_loss']:.5f} "
                        f"base_gap={row['frozen_audio_dense_gap']:.5f} "
                        f"k={row['rollout_index']} gn={row['grad_norm']:.3f} "
                        f"d={d_value:.2e} eff_lr={effective_lr:.2e} "
                        f"{elapsed:.1f}s peak={peak:.1f}GiB {fs.phase_summary()}",
                        flush=True,
                    )
                    with open(metrics_path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row) + "\n")
                t_log = time.time()

                del packed, video_rows, audio_rows, dense_a, frozen_v, frozen_a
                del velocity_v, velocity_a, student_v, student_a, loss, audio_loss, video_loss

                periodic = step % cfg.checkpoint.save_every == 0 or step == max_steps
                if periodic or step in early_saves:
                    save_deployment(step)
                if periodic:
                    _save_train_state(
                        output_dir, step, epoch, in_epoch, iteration, bank, optimizer,
                        noise_generator, rank, world, cfg)
                if step >= max_steps:
                    break
            epoch += 1
            in_epoch = 0
            resumable_sampler.skip = 0

        dist.barrier()
        if rank == 0:
            print("audio-fix training done", flush=True)
    finally:
        bank.uninstall()
        if rank == 0:
            run_lock.release(output_dir)
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    """Load the structured audio-fix config and start training."""
    cfg = load_config(AudioFixConfig, extra_validators=[validate_audio_fix])
    run(cfg)


if __name__ == "__main__":
    main()
