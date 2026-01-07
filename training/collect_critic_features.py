import json
import os
import random
import time
from typing import List

import click
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer

from dataloaders.collate_fn_math import collate_fn_math, collate_fn_gsm8k
from math_metrics import reward_from_responses_gsm8k, reward_from_responses_math
from networks.lladou_v0 import LLaDOUModelLM, sample


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloader(
    task: str,
    local_path: str,
    split: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    max_rows: int,
):
    if task == "gsm8k":
        ds = load_dataset(local_path, split=split, data_dir="main")
        collate_fn = collate_fn_gsm8k
    elif task == "math":
        ds = load_dataset(local_path, split=split)
        collate_fn = collate_fn_math
    else:
        raise ValueError(f"Unknown task: {task}")

    if max_rows > 0:
        ds = ds.select(range(min(len(ds), max_rows)))
    ds = ds.shuffle(seed=seed).with_format("torch")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )


def outcome_reward(task: str, batch, responses, num_generations, device):
    if task == "gsm8k":
        answers = batch["answers"] * num_generations
        rewards = reward_from_responses_gsm8k(answers, responses)
        return torch.tensor(rewards, device=device, dtype=torch.float32)
    if task == "math":
        answers = batch["answers"] * num_generations
        rewards = reward_from_responses_math(answers, responses)
        return torch.tensor(rewards, device=device, dtype=torch.float32)
    raise ValueError(f"Unknown task: {task}")


@click.command()
@click.option("--actor_ckpt_path", type=str, required=True)
@click.option("--output_path", type=str, default="datasets_cache/critic_train.pt")
@click.option("--gsm8k_path", type=str, default="datasets/gsm8k")
@click.option("--math_path", type=str, default="datasets/MATH")
@click.option("--tasks", type=str, default="gsm8k,math")
@click.option("--batch_size", type=int, default=1)
@click.option("--num_workers", type=int, default=2)
@click.option("--steps", type=int, default=256)
@click.option("--gen_length", type=int, default=256)
@click.option("--block_length", type=int, default=8)
@click.option("--num_generations", type=int, default=1)
@click.option("--temperature", type=float, default=1.0)
@click.option("--actor_inference", type=bool, default=False)
@click.option("--k_steps", type=int, default=2)
@click.option("--pool", type=str, default="mean")
@click.option("--max_rows", type=int, default=0)
@click.option("--seed", type=int, default=113)
@click.option("--log_every", type=int, default=50)
def main(
    actor_ckpt_path,
    output_path,
    gsm8k_path,
    math_path,
    tasks,
    batch_size,
    num_workers,
    steps,
    gen_length,
    block_length,
    num_generations,
    temperature,
    actor_inference,
    k_steps,
    pool,
    max_rows,
    seed,
    log_every,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(actor_ckpt_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 126081

    actor_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    actor = LLaDOUModelLM.from_pretrained(
        pretrained_model_name_or_path=actor_ckpt_path,
        trust_remote_code=True,
        torch_dtype=actor_dtype,
    )
    actor.eval().requires_grad_(False).to(device)

    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    dataloaders = {}
    if "gsm8k" in task_list:
        dataloaders["gsm8k"] = build_dataloader(
            "gsm8k", gsm8k_path, "train", batch_size, num_workers, seed, max_rows
        )
    if "math" in task_list:
        dataloaders["math"] = build_dataloader(
            "math", math_path, "train", batch_size, num_workers, seed, max_rows
        )
    if not dataloaders:
        raise ValueError("No tasks selected. Use --tasks gsm8k,math or a subset.")

    features_list: List[torch.Tensor] = []
    timesteps_list: List[torch.Tensor] = []
    rewards_list: List[torch.Tensor] = []
    task_ids_list: List[torch.Tensor] = []

    rng = random.Random(seed)
    total_batches = sum(len(dl) for dl in dataloaders.values())
    pbar = tqdm(total=total_batches, desc="collect")

    hidden_size = None
    has_extra_features = False
    total_prompts = 0
    total_rows = 0
    start_time = time.perf_counter()

    for task_idx, (task, dl) in enumerate(dataloaders.items()):
        for batch in dl:
            total_prompts += len(batch["problems"])
            if k_steps <= 1:
                step_indices = [steps // 2]
            else:
                step_indices = rng.sample(range(steps), k=min(k_steps, steps))
            # Keep sampling ops in fp32; sample() already autocasts model forward.
            with torch.inference_mode():
                sample_outputs = sample(
                    actor,
                    batch,
                    tokenizer,
                    device=device,
                    reward_fn=lambda b, r, n, d: outcome_reward(task, b, r, n, d),
                    num_generations=num_generations,
                    repeat_times=1,
                    temperature=temperature,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    inference=actor_inference,
                    record_pooled_hidden_steps=step_indices,
                    pooled_hidden_pool=pool,
                    record_trajectory=False,
                )

            rewards = sample_outputs["rewards"]
            batch_rows = rewards.numel()
            pooled_states = sample_outputs.get("pooled_hidden_states")
            pooled_steps = sample_outputs.get("pooled_hidden_steps")
            pooled_extras = sample_outputs.get("pooled_extra_features")
            if pooled_states is None or pooled_steps is None:
                raise RuntimeError("sample() did not return pooled hidden states. Update networks/lladou_v0.py.")
            if pooled_extras is not None:
                has_extra_features = True

            for idx, (pooled, si) in enumerate(zip(pooled_states, pooled_steps)):
                if pooled_extras is not None:
                    pooled = torch.cat([pooled, pooled_extras[idx].to(pooled.device)], dim=-1)
                if hidden_size is None:
                    hidden_size = pooled.shape[-1]
                features_list.append(pooled.to(dtype=torch.float16).cpu())
                timesteps_list.append(
                    torch.full((batch_rows,), float(si) / float(steps), dtype=torch.float16).cpu()
                )
                rewards_list.append(rewards.to(dtype=torch.float16).cpu())
                task_ids_list.append(torch.full((batch_rows,), task_idx, dtype=torch.int8))
                total_rows += batch_rows

            del sample_outputs, pooled_states, pooled_steps, pooled_extras, rewards
            if device == "cuda":
                torch.cuda.empty_cache()

            if log_every > 0 and (pbar.n % log_every == 0):
                elapsed = time.perf_counter() - start_time
                pbar.set_postfix(
                    task=task,
                    rows=total_rows,
                    prompts=total_prompts,
                    t=f"{elapsed:.1f}s",
                )

            pbar.update(1)

    pbar.close()

    features = torch.cat(features_list, dim=0)
    timesteps = torch.cat(timesteps_list, dim=0)
    rewards = torch.cat(rewards_list, dim=0)
    task_ids = torch.cat(task_ids_list, dim=0)

    data = {
        "features": features,
        "timesteps": timesteps,
        "rewards": rewards,
        "task_ids": task_ids,
        "meta": {
            "hidden_size": hidden_size,
            "pool": pool,
            "use_timestep": True,
            "extra_feature_names": (
                ["mask_frac", "filled_frac", "block_id_norm", "block_step_frac"] if has_extra_features else []
            ),
            "steps": steps,
            "gen_length": gen_length,
            "block_length": block_length,
            "k_steps": k_steps,
            "num_generations": num_generations,
            "temperature": temperature,
            "actor_inference": actor_inference,
            "tasks": task_list,
            "seed": seed,
        },
    }

    torch.save(data, output_path)
    print(f"Saved features to {output_path}")
    print(f"Total prompts: {total_prompts} | Total rows: {total_rows} | Hidden size: {hidden_size}")


if __name__ == "__main__":
    main()
