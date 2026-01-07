import os
import json
import glob
import click 
import numpy as np
from tqdm import tqdm
from typing import Sequence
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

from networks.lladou_v0 import LLaDOUModelLM, sample
from networks.value_critic import ValueCritic
from networks.vgr_sampler import vgr_sample
from dataloaders.collate_fn_math import collate_fn_math, extract_answer_gsm8k, collate_fn_gsm8k
from evaluate.grader import math_equal
from evaluate.parser import extract_answer

def judge_answer_MATH(answers: Sequence[str], responses: Sequence[str], counts):
    ext_ans = [extract_answer(ans) for ans in answers]
    ext_res = [extract_answer(res) for res in responses]

    # stat acc
    counts[1] += len(ext_ans)
    for ans, res in zip(ext_ans, ext_res):
        if math_equal(ans, res, timeout=True):
            counts[0] += 1
    
    return counts

def judge_answer_GSM8K(answers: Sequence[str], responses: Sequence[str], counts):
    ext_ans = [extract_answer_gsm8k(ans) for ans in answers]
    ext_res = [extract_answer(res) for res in responses]

    # stat acc
    counts[1] += len(ext_ans)
    for ans, res in zip(ext_ans, ext_res):
        if math_equal(ans, res):
            counts[0] += 1
    
    return counts


def reward_from_responses_gsm8k(answers: Sequence[str], responses: Sequence[str]):
    ext_ans = [extract_answer_gsm8k(ans) for ans in answers]
    ext_res = [extract_answer(res) for res in responses]
    rewards = []
    for ans, res in zip(ext_ans, ext_res):
        rewards.append(1.0 if math_equal(ans, res) else 0.0)
    return rewards


def reward_from_responses_math(answers: Sequence[str], responses: Sequence[str]):
    ext_ans = [extract_answer(ans) for ans in answers]
    ext_res = [extract_answer(res) for res in responses]
    rewards = []
    for ans, res in zip(ext_ans, ext_res):
        rewards.append(1.0 if math_equal(ans, res, timeout=True) else 0.0)
    return rewards


@click.command()
@click.option("--ckpt_path", type=str, default="")
@click.option('--local_data_path', type=str, default="datasets/gsm8k")
@click.option('--batch_size', type=int, default=1)
@click.option('--num_workers', type=int, default=1)
@click.option('--steps', type=int, default=256)
@click.option('--gen_length', type=int, default=256)
@click.option('--block_length', type=int, default=8)
@click.option('--task', type=str, default="gsm8k")
@click.option('--seed', type=int, default=113)
@click.option('--no_sample', type=bool, default=True)
@click.option('--critic_path', type=str, default="")
@click.option('--gate_start_step', type=int, default=32)
@click.option('--retry_m', type=int, default=8)
@click.option('--max_backtracks_total', type=int, default=32)
@click.option('--min_value_improve', type=float, default=0.0)
def main(
    ckpt_path, 
    local_data_path, 
    batch_size, 
    num_workers, 
    steps, 
    gen_length, 
    block_length, 
    no_sample,
    seed,
    critic_path,
    gate_start_step,
    retry_m,
    max_backtracks_total,
    min_value_improve,
    **kwargs,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    
    print(f"Running on device: {device} (Single GPU Mode)")

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 126081
        
    model = LLaDOUModelLM.from_pretrained(
        pretrained_model_name_or_path=ckpt_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval().requires_grad_(False).to(device)

    critic = None
    if critic_path:
        resolved_path = critic_path
        if os.path.isdir(critic_path):
            candidates = sorted(glob.glob(os.path.join(critic_path, "critic_head*.pt")))
            if not candidates:
                raise FileNotFoundError(f"No critic_head*.pt found in {critic_path}")
            resolved_path = candidates[-1]
        critic_state = torch.load(resolved_path, map_location="cpu")
        state_dict = critic_state.get("state_dict", critic_state)
        critic = ValueCritic(model)
        critic.load_head_state_dict(state_dict)
        critic.eval().to(device)
    
    # load data 
    if 'MATH' in local_data_path:
        ds = load_dataset(local_data_path, split='test').with_format('torch')
        task = 'MATH500' if '500' in local_data_path else 'MATH'
        current_collate_fn = collate_fn_math 
    elif 'gsm8k' in local_data_path:
        ds = load_dataset(local_data_path, split='test', data_dir='main').with_format('torch')
        task = 'gsm8k'
        current_collate_fn = collate_fn_gsm8k
    else:
        raise ValueError(f"Invalid data path: {local_data_path}")
        
    dl = DataLoader(
        ds, 
        batch_size=batch_size, 
        collate_fn=current_collate_fn, 
        num_workers=num_workers, 
        pin_memory=True, 
        shuffle=False 
    )
    
    pbar = tqdm(dl)
    counts = [0, 0] 
    total_backtracks = 0

    for ix, batch in enumerate(pbar):
        answers = batch['answers']
        if critic is None:
            inputs = sample(
                model,
                batch,
                tokenizer,
                device=device,
                inference=no_sample,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
            )
        else:
            inputs = vgr_sample(
                model,
                critic,
                batch,
                tokenizer,
                device=device,
                inference=no_sample,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                gate_start_step=gate_start_step,
                retry_m=retry_m,
                max_backtracks_total=max_backtracks_total,
                min_value_improve=min_value_improve,
            )
            total_backtracks += inputs["backtrack_counts"].sum().item()
        
        responses = tokenizer.batch_decode(inputs['trajectory_outputs'][-1], skip_special_tokens=True)
        
        if 'MATH' in local_data_path:
            counts = judge_answer_MATH(answers, responses, counts)
        elif 'gsm8k' in local_data_path:
            counts = judge_answer_GSM8K(answers, responses, counts)

        total_correct = counts[0]
        total_samples = counts[1]
        if total_samples > 0:
            acc = total_correct / total_samples
            pbar.set_description(f"acc: {acc * 100:.2f}%")

    print("Final Counts:", counts)
    if counts[1] > 0:
        print(f"Final Acc: {counts[0]/counts[1]:.4f}")
        if critic is not None:
            avg_backtracks = total_backtracks / counts[1]
            print(f"Avg Backtracks: {avg_backtracks:.2f}")

if __name__ == "__main__":
    main()
