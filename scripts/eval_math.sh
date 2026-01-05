export PROC_PER_NODES=1

torchrun \
    --standalone \
    --nproc-per-node=$PROC_PER_NODES \
    --master-port=23443 \
    math_metrics.py \
        --ckpt_path models/LLaDOU-v0-Math \
        --local_data_path datasets/gsm8k \
        --num_workers 4 \
        --seed 112 \