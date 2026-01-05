import os, json
import click 
import numpy as np
from tqdm import tqdm
from typing import Sequence
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

from networks.lladou_v0 import LLaDOUModelLM, sample
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

    for ix, batch in enumerate(pbar):
        answers = batch['answers']

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

if __name__ == "__main__":
    main()