import os
from pathlib import Path

from lit_gpt.scripts.convert_hf_checkpoint import convert_hf_checkpoint
from lit_gpt.scripts.download import download_from_hub


def prepare_weights(
    repo_id: str,
):
    local_dir = Path(f"checkpoints/{repo_id}")
    if local_dir.exists():
        print(f"weights already exists at {local_dir}")
        return local_dir
    download_from_hub(repo_id=repo_id)
    convert_hf_checkpoint(checkpoint_dir=local_dir)
    return local_dir
