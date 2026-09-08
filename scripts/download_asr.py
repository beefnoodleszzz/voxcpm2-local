"""Explicit installation of the optional, revision-pinned local ASR weights."""

import json

import _bootstrap  # noqa: F401
from huggingface_hub import snapshot_download

from src.voice_library import ROOT


def main():
    lock = json.loads((ROOT / "configs/asr.model.lock.json").read_text())
    path = snapshot_download(
        repo_id=lock["repo"], revision=lock["revision"], local_dir=ROOT / lock["local_path"]
    )
    print(path)


if __name__ == "__main__":
    main()
