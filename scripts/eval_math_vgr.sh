export PROC_PER_NODES=1

torchrun \
    --standalone \
    --nproc-per-node=$PROC_PER_NODES \
    --master-port=23443 \
    math_metrics.py \
        --ckpt_path models/LLaDOU-v0-Math \
        --critic_path runs/critic/LLaDOU-v0-Math/critic_head_epoch1.pt \
        --local_data_path datasets/gsm8k \
        --num_workers 4 \
        --gate_start_step 32 \
        --retry_m 8 \
        --max_backtracks_total 32 \
        --seed 112 \
