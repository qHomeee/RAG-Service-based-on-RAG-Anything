$ErrorActionPreference = "Stop"

Write-Host "[repair_env] uninstall conflicting HF packages..."
pip uninstall -y accelerate huggingface_hub transformers

Write-Host "[repair_env] reinstall from requirements..."
pip install --no-cache-dir -r requirements.txt

Write-Host "[repair_env] sanity check 1 (hub/transformers/accelerate)..."
python -c "from huggingface_hub import split_torch_state_dict_into_shards; import transformers, accelerate; print('OK')"

Write-Host "[repair_env] sanity check 2 (sentence-transformers model load)..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('loaded')"

Write-Host "[repair_env] done"
