"""Download VAD and Turn Detector models to data/models/.

    make download
"""

from __future__ import annotations

import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "data" / "models"
VAD_DIR = MODELS_DIR / "vad"
TURN_DIR = MODELS_DIR / "turn"

##### VAD (SILERO) #####

HG_TURN_REPO = "livekit/turn-detector"
HG_TURN_REVISION = "v0.4.1-intl"


def download_vad() -> None:
    """Copy bundled Silero VAD ONNX from the installed package."""
    VAD_DIR.mkdir(parents=True, exist_ok=True)
    target = VAD_DIR / "silero_vad.onnx"

    if target.exists():
        print(f"  ✓ VAD model already at {target}")
        return

    import importlib.resources

    source = importlib.resources.files("livekit.plugins.silero") / "resources" / "silero_vad.onnx"
    with importlib.resources.as_file(source) as src_path:
        shutil.copy2(src_path, target)

    print(f"  ✓ VAD model copied to {target} ({target.stat().st_size / 1024:.0f} KB)")


##### TURN DETECTOR (MULTILINGUAL) #####


def download_turn() -> None:
    """Download MultilingualModel from HuggingFace to data/models/turn/."""
    TURN_DIR.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    revision = HG_TURN_REVISION

    AutoTokenizer.from_pretrained(HG_TURN_REPO, revision=revision, cache_dir=str(TURN_DIR))
    print("  ✓ Turn detector tokenizer")

    for fname, subfolder in [("model_q8.onnx", "onnx"), ("languages.json", None)]:
        kwargs: dict = {
            "repo_id": HG_TURN_REPO,
            "filename": fname,
            "revision": revision,
            "cache_dir": str(TURN_DIR),
        }
        if subfolder:
            kwargs["subfolder"] = subfolder
        path = hf_hub_download(**kwargs)
        print(f"  ✓ {fname} → {path}")


##### MAIN #####


if __name__ == "__main__":
    print("Downloading VAD model (Silero)...")
    download_vad()
    print("\nDownloading Turn Detector model (Multilingual)...")
    download_turn()
    print("\nDone. Models at:", MODELS_DIR)
