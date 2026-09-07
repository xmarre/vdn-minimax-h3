"""Precomputed-latent dataset for the MiniMax-H3 T2VA fine-tune.

One sample = one clip, pre-encoded with the H3 VAEs and text encoder:

  video : `<root>/video/<id>.pt`  (24, 102, 48, 84) bf16 — VAE latents, already in the
          normalized space the transformer denoises in (std ~= 1; the diffusers decode
          block un-normalizes with `latents_std * x + latents_mean` afterwards).
  audio : `<root>/audio/<id>.pt`  (2, 32, 575) bf16 — stereo audio-VAE latents,
          40 latents/s over the same 345-frame / 14.375 s window.
  text  : `<root>/text/<id>.pt`   {"prompt_embeds": (L, 5120) bf16,
                                   "text_token_tags": (L,) int64}
          precomputed Qwen3-VL rows in the exact format `MiniMaxH3Transformer3DModel`
          consumes, tags included.

The index (`video_index.jsonl`) carries `latent_path` spelled as the corresponding
`video/<id>.pt`; audio/text sidecars are found by sibling path arithmetic. In
``text_only`` mode (Stage-DMD and audio-fix training), the video path is only an
identifier: the video/audio files do not need to exist and are never opened.
"""

import json
import os

import torch
from torch.utils.data import Dataset


class H3LatentT2VADataset(Dataset):
    def __init__(self, index_file: str, text_only: bool = False):
        """`text_only`: yield only prompt rows; no video/audio latent file is opened."""
        self.text_only = text_only
        self.records = []
        with open(index_file) as f:
            for line in f:
                rec = json.loads(line)
                if "error" in rec:
                    continue
                self.records.append(rec["latent_path"])
        if not self.records:
            raise RuntimeError(f"No usable rows in {index_file}")

        if text_only:
            self.video_shape = None
        else:
            # The packed-sequence geometry is batch-shared, so every clip must have the same
            # latent shape (e.g. (24, 102, 48, 84) for 345 frames at 384x672).
            first = torch.load(self.records[0], map_location="cpu", weights_only=True)
            self.video_shape = tuple(first.shape)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        video_path = self.records[idx]
        base = os.path.dirname(os.path.dirname(video_path))
        name = os.path.basename(video_path)

        text = torch.load(os.path.join(base, "text", name), map_location="cpu", weights_only=True)
        if self.text_only:
            return {
                "prompt_embeds": text["prompt_embeds"],  # (L, 5120) bf16
                "text_token_tags": text["text_token_tags"],  # (L,) int64
            }
        video = torch.load(video_path, map_location="cpu", weights_only=True)
        audio = torch.load(os.path.join(base, "audio", name), map_location="cpu", weights_only=True)

        if tuple(video.shape) != self.video_shape:
            raise ValueError(f"{video_path}: shape {tuple(video.shape)} != {self.video_shape}")

        return {
            "video_latents": video,  # (24, T, h, w) bf16
            "audio_latents": audio,  # (2, 32, A) bf16
            "prompt_embeds": text["prompt_embeds"],  # (L, 5120) bf16
            "text_token_tags": text["text_token_tags"],  # (L,) int64
        }


def collate_single(samples: list) -> dict:
    """batch_size=1: text length varies per clip, so the sample passes through as-is."""
    assert len(samples) == 1
    return samples[0]
