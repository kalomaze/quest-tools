from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="Quest-AI/quest-corruption-200k-dataset",
    filename="HuggingFaceFW_fineweb-edu_sample-350BT_train_200000_filtered_corrupted_axolotl.jsonl",
    repo_type="dataset",
    local_dir="/root/cproject_updated"
)

print(f"Downloaded to: {file_path}")