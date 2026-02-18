$ErrorActionPreference = "Stop"

Write-Host "[repair_env] uninstall conflicting HF packages..."
pip uninstall -y accelerate huggingface-hub huggingface_hub transformers tokenizers

Write-Host "[repair_env] reinstall core requirements..."
pip install --no-cache-dir -r requirements-core.txt

Write-Host "[repair_env] sanity check 1 (transformers import/version)..."
python -c "import transformers; print(transformers.__version__)"

Write-Host "[repair_env] sanity check 2 (sentence-transformers model load)..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('loaded')"

Write-Host "[repair_env] done"
