#!/bin/bash
set -x
export HF_HOME=/workspace/hf
pip install -q "transformers==4.56.2" "peft>=0.13" "accelerate>=0.34" "huggingface_hub>=0.24" safetensors numpy scikit-learn pyyaml 2>&1 | tail -5
python3 -c "import peft,transformers;print(\"OK\",peft.__version__,transformers.__version__)"
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --exclude "*.pth" 2>&1 | tail -3
echo "=== SETUP DONE ==="
touch /workspace/DONE_setup
