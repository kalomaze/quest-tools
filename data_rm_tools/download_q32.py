from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id="Qwen/Qwen2.5-32B",
    local_dir="/root/cproject_updated/Qwen2.5-32B"
)

print(f"Downloaded to: {model_path}")