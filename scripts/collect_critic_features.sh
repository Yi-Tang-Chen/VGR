set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

export TOKENIZERS_PARALLELISM=false

python training/collect_critic_features.py \
    --actor_ckpt_path models/LLaDOU-v0-Math \
    --output_path datasets_cache/critic_train.pt \
    --gsm8k_path datasets/gsm8k \
    --math_path datasets/MATH \
    --tasks gsm8k,math \
    --batch_size 4 \
    --num_workers 2 \
    --steps 128 \
    --gen_length 256 \
    --block_length 8 \
    --k_steps 2 \
    --num_generations 1 \
    --temperature 1.0 \
    --actor_inference True \
    --seed 113
