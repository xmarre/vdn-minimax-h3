from __future__ import annotations

import json

import torch

from src.training.dataset_h3_latents import H3LatentT2VADataset


def test_text_only_dataset_does_not_open_video_or_audio_files(tmp_path):
    root = tmp_path / "dataset"
    text = root / "text"
    text.mkdir(parents=True)
    torch.save({
        "prompt_embeds": torch.randn(3, 8, dtype=torch.bfloat16),
        "text_token_tags": torch.ones(3, dtype=torch.long),
    }, text / "00000.pt")

    # The video/audio files intentionally do not exist. The latent_path is only the
    # stable sibling-path identifier in text-only Stage-DMD/audio-fix training.
    index = root / "video_index.jsonl"
    with index.open("w") as f:
        f.write(json.dumps({"latent_path": str(root / "video" / "00000.pt")}) + "\n")

    dataset = H3LatentT2VADataset(str(index), text_only=True)
    assert dataset.video_shape is None
    sample = dataset[0]
    assert tuple(sample["prompt_embeds"].shape) == (3, 8)
    assert tuple(sample["text_token_tags"].shape) == (3,)
