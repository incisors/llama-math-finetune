#!/usr/bin/env bash
# RunPod environment setup. Run once after `git clone` on a fresh pod.
#
# Usage:
#   bash setup.sh
#   source ~/.bashrc          # picks up HF_HOME in this shell
#
# Assumes Network Volume mounted at /workspace.

set -euo pipefail

# 1. Persist HF cache on the Network Volume so models survive pod restarts.
mkdir -p /workspace/hf_cache

# Idempotent: only append the exports if they aren't already in ~/.bashrc.
if ! grep -q "HF_HOME=/workspace/hf_cache" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc <<'EOF'

# llama-math-finetune: keep HF cache on the persistent network volume
export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub
EOF
    echo "Added HF_HOME exports to ~/.bashrc"
fi

# Also export in the current process so the sanity check below sees them.
export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub

# 2. Install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Sanity
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}, GPUs={torch.cuda.device_count()}')"

cat <<'EOF'

==========================================================
NEXT STEPS (manual, interactive):
  source ~/.bashrc             # picks up HF_HOME in this shell
  huggingface-cli login        # paste HF token (needs Llama-3.1 access)
  wandb login                  # paste W&B API key
==========================================================
EOF
