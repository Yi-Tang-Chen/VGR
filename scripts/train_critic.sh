set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

export TOKENIZERS_PARALLELISM=false

python training/train_critic.py \
    --dataset_path datasets_cache/critic_train.pt \
    --critic_out_dir runs/critic/LLaDOU-v0-Math \
    --batch_size 512 \
    --num_workers 2 \
    --epochs 5 \
    --loss_type bce \
    --lr 2.0e-4 \
    --seed 113
