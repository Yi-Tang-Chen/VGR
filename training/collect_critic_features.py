import json
import os
import random
import time
from typing import List

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_candidate_env = [
    os.environ.get("TMPDIR"),
    os.environ.get("TEMP"),
    os.environ.get("TMP"),
]
_candidates = [c for c in _candidate_env if c]
_candidates += [
    "/dev/shm",
    "/tmp",
    "/var/tmp",
    "/usr/tmp",
    os.path.join(_repo_root, ".tmp"),
]
_tmpdir = None
for _path in _candidates:
    if not _path:
        continue
    try:
        os.makedirs(_path, exist_ok=True)
        _probe = os.path.join(_path, f".lladou_tmp_test_{os.getpid()}")
        with open(_probe, "w", encoding="utf-8") as _handle:
            _handle.write("1")
        try:
            os.remove(_probe)
        except OSError:
            pass
        _tmpdir = os.path.abspath(_path)
        break
    except OSError:
        continue
if _tmpdir is None:
    raise RuntimeError(
        "No usable temporary directory. Set TMPDIR to a path with free space."
    )
os.environ.setdefault("TMPDIR", _tmpdir)
os.environ.setdefault("TEMP", _tmpdir)
os.environ.setdefault("TMP", _tmpdir)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import click
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer

from dataloaders.collate_fn_math import (
    collate_fn_math,
    collate_fn_gsm8k,
    extract_answer_gsm8k,
)
from evaluate.grader import math_equal
from evaluate.parser import extract_answer
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


def decode_generations(tokenizer, gen_ids, eos_id, include_raw=False):
    gen_ids = gen_ids.detach().cpu()
    decoded = []
    decoded_raw = [] if include_raw else None
    for row in gen_ids:
        ids = row.tolist()
        if eos_id is not None and eos_id in ids:
            ids = ids[: ids.index(eos_id)]
        decoded.append(tokenizer.decode(ids, skip_special_tokens=True))
        if decoded_raw is not None:
            decoded_raw.append(tokenizer.decode(ids, skip_special_tokens=False))
    return decoded, decoded_raw


@click.command()
@click.option("--actor_ckpt_path", type=str, required=True)
@click.option("--output_path", type=str, default="datasets_cache/critic_train.pt")
@click.option("--gsm8k_path", type=str, default="datasets/gsm8k")
@click.option("--math_path", type=str, default="datasets/MATH")
@click.option("--tasks", type=str, default="gsm8k,math")
@click.option("--batch_size", type=int, default=2)
@click.option("--num_workers", type=int, default=4)
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
@click.option("--debug_samples", type=int, default=0)
@click.option("--debug_path", type=str, default="datasets_cache/critic_debug.jsonl")
@click.option("--ban_eos", type=bool, default=False)
@click.option("--align_to_max_prompt", type=bool, default=False)
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
    debug_samples,
    debug_path,
    ban_eos,
    align_to_max_prompt,
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
    if align_to_max_prompt:
        tokenizer.padding_side = "left"

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
    debug_records = []
    debug_written = 0

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
                    reward_fn=None,
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
                    record_final_output=True,
                    ban_eos=ban_eos,
                    align_to_max_prompt=align_to_max_prompt,
                )

            final_output = sample_outputs.get("final_output")
            if final_output is None:
                raise RuntimeError("sample() did not return final_output. Update networks/lladou_v0.py.")
            prompt_seq_len = sample_outputs["attention_mask"].shape[1] - gen_length
            gen_ids = final_output[:, prompt_seq_len : prompt_seq_len + gen_length]
            gen_texts, gen_texts_raw = decode_generations(
                tokenizer, gen_ids, tokenizer.eos_token_id, include_raw=(debug_samples > 0)
            )
            responses_full = None
            if debug_samples > 0:
                responses_full = tokenizer.batch_decode(final_output, skip_special_tokens=True)

            answers = batch["answers"] * num_generations
            if task == "gsm8k":
                rewards = torch.tensor(
                    reward_from_responses_gsm8k(answers, gen_texts), dtype=torch.float32
                )
            else:
                rewards = torch.tensor(
                    reward_from_responses_math(answers, gen_texts), dtype=torch.float32
                )
            batch_rows = rewards.numel()
            if debug_samples > 0 and debug_written < debug_samples:
                problems = batch["problems"] * num_generations
                for i, (prob, ans, resp, gen_resp, gen_resp_raw) in enumerate(
                    zip(problems, answers, responses_full, gen_texts, gen_texts_raw)
                ):
                    if debug_written >= debug_samples:
                        break
                    prompt_len = int(sample_outputs["prompt_len"][i].item())
                    eos_id = tokenizer.eos_token_id
                    mask_id = getattr(tokenizer, "mask_token_id", None)
                    if mask_id is None:
                        mask_id = getattr(actor.config, "mask_token_id", None)
                    eos_frac = (
                        (gen_ids[i] == eos_id).float().mean().item() if eos_id is not None else 0.0
                    )
                    mask_frac = (
                        (gen_ids[i] == mask_id).float().mean().item() if mask_id is not None else 0.0
                    )
                    if task == "gsm8k":
                        ext_ans = extract_answer_gsm8k(ans)
                        ext_res = extract_answer(gen_resp)
                        reward_dbg = 1.0 if math_equal(ext_ans, ext_res) else 0.0
                    else:
                        ext_ans = extract_answer(ans)
                        ext_res = extract_answer(gen_resp)
                        reward_dbg = 1.0 if math_equal(ext_ans, ext_res, timeout=True) else 0.0
                    debug_records.append(
                        {
                            "task": task,
                            "problem": prob,
                            "answer": ans,
                            "response": resp,
                            "gen_response": gen_resp,
                            "gen_response_raw": gen_resp_raw,
                            "gen_eos_frac": eos_frac,
                            "gen_mask_frac": mask_frac,
                            "prompt_len": prompt_len,
                            "ext_answer": ext_ans,
                            "ext_response": ext_res,
                            "reward": reward_dbg,
                        }
                    )
                    debug_written += 1
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

            del (
                sample_outputs,
                pooled_states,
                pooled_steps,
                pooled_extras,
                rewards,
                final_output,
                gen_ids,
            )
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
    if debug_records:
        os.makedirs(os.path.dirname(debug_path) or ".", exist_ok=True)
        with open(debug_path, "w", encoding="utf-8") as handle:
            for item in debug_records:
                handle.write(json.dumps(item, ensure_ascii=True) + "\n")
        print(f"Saved debug samples to {debug_path}")


if __name__ == "__main__":
    main()
