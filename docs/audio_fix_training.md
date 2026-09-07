# Audio-fix LoRA training

This recipe trains a **small correction LoRA on top of the finished released VDN/Turbo model**. It does not retrain Stage-B or Turbo.

The frozen training stack is:

```text
MiniMax H3 base
+ VDN linear branch
+ Stage-B `default` adapter @ native/unit scale 1.0
+ released Stage-DMD `turbo` adapter @ native/unit scale 1.0
```

The only trainable weights are a new `audio_fix` LoRA. Its residual is injected only on the generated-audio rows of packed DiT sequence-space Linear targets. The trainer uses the 10-step deterministic `res_multistep` sampler contract from the production investigation, a dense released-H3 audio teacher on the same on-policy state/grid point, and a frozen-production video target to penalize indirect video drift.

## Frozen adapter scale contract

Training intentionally uses the **canonical finished checkpoint at unit adapter scale**. Stage-B is `1.0` and Turbo is `1.0` throughout rollout, frozen-production target and student forward. No ComfyUI inference-strength multiplier is baked into the correction.

The current deployment can still run Turbo at a tuned strength such as `0.75`; that is a **post-training validation/deployment setting**. The first validation of a new audio-fix checkpoint should use Turbo `1.0`, because that is the stack against which the correction was learned. Then repeat the matched seeds at Turbo `0.75` to verify that the correction remains useful at the actual deployment strength.

The trainer also validates the source Turbo ModelSpec before deriving correction targets: every resolved Turbo target must have native LoRA `alpha / rank == 1`, including per-target rank/alpha patterns. This prevents an accidentally pre-scaled Turbo artifact from silently becoming the training reference.

## Production sampler and chunk contract

The current Continuum generation path requests **7 seconds per H3 chunk at 24 fps**:

```text
7 s * 24 fps = 168 requested pixel frames
```

MiniMax H3's video VAE accepts `17 * n + 5` frame counts, so the same canonical alignment used by the H3 pipeline snaps that request upward to:

```text
175 aligned pixel frames
52 video latent frames
292 audio latent frames (40 Hz audio-latent grid)
```

The trainer therefore stores `generation.num_frames = 168` and lets `align_num_frames()` perform the same H3 alignment internally. The older OpenVDN Stage-DMD default of 345 frames corresponds to about 14.4 seconds / 102 video latent frames and is intentionally **not** used for this correction run; it is roughly twice the temporal extent of one production Continuum H3 call.

This first corrective trainer matches the **sampler, sigma source, AV coordinate system, shifts, step count and per-H3-call duration**. It remains deliberately text-only and full-grid. It does **not** claim to reproduce reference/keyframe rows, Spectrum forecast skipping, Continuum prefix state, or Progressive Mixed-Grid handoff inside the training loop. Those remain mandatory decoded deployment tests.

### Exact production sigma source

The production graph feeds `res_multistep` from **MiniMax H3 SA-Solver Scheduler → `simple_control`**. Despite the node name, `simple_control` does not run SA-Solver equations; it is the scheduler's exact parity control and delegates to current ComfyUI `simple` sigma placement. The sampler remains `res_multistep`.

The trainer reproduces that discrete `simple` table selection directly. ComfyUI's flow model has a 1000-point table and `simple_scheduler` walks it backwards using `int(x * len(table) / steps)`. At exactly 10 steps this selects the shared base coordinates:

```text
1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0
```

before applying the MiniMax video/audio shifts. The CPU suite also tests a non-divisor step count so this code cannot silently regress into a continuous `linspace` approximation.

### MiniMax AV sampler parity

ComfyUI does **not** hand `res_multistep` two independent video/audio sigma streams. `ModelSamplingAV` carries both modalities on the video sigma grid and transforms the audio state around every H3 forward:

```text
y_audio = (sigma_video / sigma_audio) * x_audio
```

For MiniMax's shifted flow grids this carried state is exactly an ordinary video-sigma flow whose clean endpoint is

```text
(video_shift / audio_shift) * x0_audio
```

which is `4 * x0_audio` at the production shifts `12 / 3`. The trainer therefore keeps its audio sampler state in that carried coordinate, converts it back to native `sigma_audio` space only for the H3 forward, and applies RES history to both streams on the **video sigma grid**. This distinction matters for a second-order multistep method; independently running RES on the native audio sigma grid is not the same production trajectory.

The CPU math suite checks the carry identity independently of ComfyUI as well as the deterministic RES update.

### VDN attention backend parity

Training keeps OpenVDN's differentiable `flex` window-softmax backend. The production ComfyUI VDN node commonly uses its `grouped` backend. These are not different VDN attention masks: both compute the same per-query set of global rows, local window rows and configured anchor rows, with one exact softmax over that set. `grouped` decomposes the mask into dense SDPA calls; `flex` expresses the same operator through the training-capable sparse mask. Kernel choice/reduction order can produce small BF16 numerical differences, but there is no deliberate attention-topology mismatch to “fix” by forcing the trainer onto the inference backend.

## Required model files

You need only these persistent model components from `OpenVDN/vdn-minimax-h3`:

```text
ckpts/
├── h3-base/
│   └── transformer/
└── stage-dmd-step-250/
    ├── model_spec.json
    ├── metadata.json
    ├── linear_branch/
    └── adapters/
        ├── default/
        └── turbo/
```

You do **not** need a separate `stage-b-step-2000` directory or Larry's raw Turbo initializer for this recipe. `stage-dmd-step-250` is the finished frozen VDN/Stage-B/Turbo stack that the new adapter corrects.

Download the minimum training weights with:

```bash
hf download OpenVDN/vdn-minimax-h3 \
  --include "h3-base/transformer/**" \
  --include "stage-dmd-step-250/**" \
  --local-dir ckpts
```

The video VAE and audio VAE are not needed for optimization. They are only needed later if you use this repository's renderer to decode validation samples.

The Qwen3-VL conditioner is needed only when encoding new prompts. It can be unloaded after the prompt `.pt` files have been written.

Use the official BF16 Diffusers/OpenVDN training weights here. Do not use the ComfyUI INT8/ConvRot or pruned deployment checkpoint as the training base.

## Environment

Follow the repository environment exactly; in particular, use Python 3.12, the pinned PyTorch/CUDA build, and the patched Diffusers tree:

```bash
conda create -n vdn python=3.12 -y
conda activate vdn
pip install uv

uv pip install torch==2.13.0 \
  --index-url https://download.pytorch.org/whl/cu129
uv pip install --prerelease=allow -e .
bash scripts/setup_diffusers.sh
```

`prodigy-plus-schedule-free==2.0.1` is pinned by this repository.

### Host RAM

The trainer constructs the BF16 H3 model on CPU before FSDP2 placement. A host/WSL memory limit around 42 GiB is insufficient for the 33B base. For a one-GPU workstation, provide substantially more host memory; 80–96 GiB is a practical target if the machine has it. Activation offload also uses host memory, so swap should be a safety margin rather than the normal activation store.

## Prompt dataset

The first audio-fix recipe is text-only. It does not load training video or audio latents, and it does not synthesize reference/keyframe conditioning rows. That keeps the corrective experiment small and isolates the general VDN/Turbo audio prior. Reference-heavy production workflows must still be included in decoded validation; if the learned correction only works for plain T2V, the next dataset revision must add the corresponding reference-conditioning topology rather than assuming text-only generalization.

Prepare a JSONL of prompts that deliberately covers both regressions and controls. At minimum include:

- visible people with no requested dialogue;
- explicit silence/no-speech scenes;
- whispered dialogue;
- ordinary prompted speech;
- ambient/action scenes where effects should remain but speech should not appear.

For example:

```jsonl
{"prompt":"A woman silently reads beside an open window. No dialogue or speech. Only quiet room ambience and distant rain."}
{"prompt":"Close-up conversation. The woman whispers: 'Stay here.' Her delivery is very quiet and breathy."}
{"prompt":"A mechanic works under a car. No talking. Metallic tool sounds and workshop ambience only."}
```

Encode the prompts once:

```bash
python src/inference/encode_prompt.py \
  --jsonl data/audio_fix/prompts.jsonl \
  --take 256 \
  --out_dir data/audio_fix/text \
  --name sample
```

The existing dataset API derives `text/<name>.pt` from the `latent_path` field used by earlier training stages. In text-only mode the corresponding `video/<name>.pt` file does **not** need to exist. Create the index like this:

```bash
python - <<'PY'
import glob, json, os
root = os.path.abspath("data/audio_fix")
text = sorted(glob.glob(os.path.join(root, "text", "sample_*.pt")))
os.makedirs(root, exist_ok=True)
with open(os.path.join(root, "video_index.jsonl"), "w") as f:
    for path in text:
        name = os.path.basename(path)
        f.write(json.dumps({"latent_path": os.path.join(root, "video", name)}) + "\n")
print(f"wrote {len(text)} rows")
PY
```

## Validate before loading the model

For a single GPU:

```bash
python src/training/train_audio_fix.py \
  --config configs/training/audio_fix_res10.yaml \
  --validate-only \
  data.index_file="$PWD/data/audio_fix/video_index.jsonl" \
  distributed.shard_size=1
```

Expected output:

```text
config ok
```

## One-step GPU smoke test

Run this before a long job, using a fresh/empty output directory:

```bash
bash scripts/training/audio_fix_res10.sh \
  data.index_file="$PWD/data/audio_fix/video_index.jsonl" \
  data.num_workers=0 \
  distributed.shard_size=1 \
  training.max_steps=1 \
  training.auto_resume=false \
  checkpoint.output_dir=ckpts/train/audio_fix_smoke \
  checkpoint.save_every=1 \
  'checkpoint.early_saves=[0,1]'
```

This must exercise model loading, frozen VDN+Stage-B+Turbo **at native unit scale**, 7-second/168-frame requested chunk geometry, exact simple-control sigma placement, Comfy-compatible carried-audio RES state, dense-H3 teacher forward, generated-audio sidecar hooks, activation-checkpoint replay, backward, Prodigy-Plus, and deployment export.

The generated-audio scope remains active around both the graph-building student forward and `loss.backward()`. That is required because non-reentrant activation checkpointing replays the frozen block forwards during backward; a scope that ended immediately after the initial forward would fail closed on recomputation.

## Full single-GPU run

```bash
bash scripts/training/audio_fix_res10.sh \
  data.index_file="$PWD/data/audio_fix/video_index.jsonl" \
  data.num_workers=0 \
  distributed.shard_size=1
```

For an eight-GPU node:

```bash
NPROC_PER_NODE=8 bash scripts/training/audio_fix_res10.sh \
  data.index_file="$PWD/data/audio_fix/video_index.jsonl"
```

The default recipe is:

```text
frozen source          stage-dmd-step-250
Stage-B train scale    1.0 (native/unit)
Turbo train scale      1.0 (native/unit)
trainable adapter      audio_fix, rank 32 / alpha 32
adapter scope          generated audio only
chunk request          168 frames = 7 seconds at 24 fps
H3 aligned chunk       175 pixel / 52 video-latent / 292 audio-latent frames
sampler                10-step deterministic res_multistep
sigma schedule         MiniMax H3 SA-Solver Scheduler simple_control / exact Comfy simple
sampler coordinate     Comfy ModelSamplingAV carried audio on video sigma grid
video/audio shifts     12 / 3
audio teacher          dense released H3, same current state and grid point
video preserve target  frozen released VDN+Stage-B+Turbo
VDN train backend      flex (same window/global/anchor softmax operator as Comfy grouped)
optimizer              Prodigy-Plus Schedule-Free 2.0.1
Prodigy LR             1.0
betas                   0.95 / 0.99
external LR scheduler  none
external grad clipping none
```

The baseline deliberately leaves Prodigy-Plus experimental SPEED/FOCUS/OrthoGrad/Cautious/Grams/ADOPT features disabled.

## Checkpoints and deployment

Early deployable checkpoints are written at steps:

```text
0, 1, 2, 4, 8, 16, 32
```

and then according to `checkpoint.save_every`.

Each `audio-fix-step-XXXXXX/` output is an exploded VDN checkpoint directory. Existing `linear_branch`, `default`, and `turbo` files are hard-linked from the frozen source where the filesystem permits; only `adapters/audio_fix` and the updated metadata/spec are new model content.

The audio-fix adapter is exported in FP32 by default during the first validation cycle. It is small enough that avoiding a second quantization variable is preferable. BF16 export can be selected later with `checkpoint.adapter_dtype=bfloat16`.

## What to validate

Do not promote a checkpoint from scalar loss alone. Compare matched seeds against the released VDN/Turbo stack for:

- false-speech/VAD rate on no-dialogue prompts;
- ASR non-empty rate and transcript duration;
- voice loudness;
- whisper-vs-normal delivery compliance;
- prompted-dialogue accuracy;
- normal ambience/effect quality;
- video quality, action fidelity, environment, and shot structure;
- the full Spectrum + Progressive Mixed-Grid + Continuum path;
- reference-heavy cases separately from plain T2V.

The trainer logs both the audio-teacher loss and a video-preservation loss. A falling audio loss is not sufficient if video-preservation error or decoded visual drift increases.

## ComfyUI validation

The resulting checkpoint adds an `audio_fix` adapter whose ModelSpec declares `scope: generated_audio` and `target_policy: portable_sequence_linear`. The ComfyUI VDN integration must honor that scope rather than merging it globally.

For the **first clean checkpoint comparison**, undo the inference diagnostics used to isolate the original bug. Test the learned correction against the same canonical stack it saw during training:

```text
Stage-B strength                     = 1.00
Turbo strength                       = 1.00
audio_fix_strength                   = 1.00
global_gate_mode                     = checkpoint
adapter_ablation                     = none
audio_adapter_strength               = 1.00
conditioning_adapter_strength        = 1.00
audio_video_context_strength         = 1.00
conditioning_video_context_strength  = 1.00
```

Run matched seeds with `audio_fix_strength=0` and `1` to isolate the learned correction itself.

Then repeat the same matched-seed set at the current deployment Turbo setting:

```text
Turbo strength     = 0.75
audio_fix_strength = 1.00
```

Keep every other diagnostic/isolation control at its normal value. Turbo `0.75` is deliberately **not** the training scale; this second pass measures how well the canonical correction transfers to the tuned deployment strength.
