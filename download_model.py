from huggingface_hub import hf_hub_download
import os

repo_id = "laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K"
filename = "open_clip_pytorch_model.bin"
local_directory = "clip_model"

os.makedirs(local_directory, exist_ok=True)

hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_directory
)

print(f"Model {repo_id} downloaded to {local_directory}/")
