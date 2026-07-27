#!/bin/bash
set -x
export HF_HOME=/workspace/hf
pip install -q --break-system-packages "transformers==4.56.2" "peft>=0.13" "accelerate>=0.34" "huggingface_hub>=0.24" safetensors numpy scikit-learn pyyaml
python3 -c "import peft,transformers,huggingface_hub;print(\"IMPORTS_OK\",peft.__version__,transformers.__version__)"
python3 - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen2.5-7B-Instruct", ignore_patterns=["*.pth","*.msgpack","*.h5"], max_workers=16)
print("DL_OK", p)
open("/workspace/DONE_model","w").write(p)
PY
echo "=== SETUP2 DONE ==="
