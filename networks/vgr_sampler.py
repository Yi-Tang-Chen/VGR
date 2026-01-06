import torch
import torch.nn.functional as F


def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    if method == "max":
        return categorical_probs.argmax(dim=-1)
    raise ValueError(f"Method {method} for sampling categorical variables is not valid.")


@torch.no_grad()
def vgr_sample(
    model,
    critic,
    batch,
    tokenizer,
    device,
    reward_fn=None,
    num_generations=1,
    repeat_times=1,
    temperature=1.0,
    steps=256,
    gen_length=256,
    block_length=8,
    mask_id=126336,
    eos_id=126081,
    inference=False,
    gate_start_step=32,
    retry_m=8,
    max_backtracks_total=32,
    min_value_improve=0.0,
):
    if isinstance(batch, str):
        batch = {
            "problems": [batch],
        }
    if block_length is None:
        block_length = gen_length
    assert gen_length % block_length == 0
    steps_per_block = steps * block_length // gen_length

    prob_dtype = torch.float32
    problems = batch["problems"]
    m = [[{"role": "user", "content": prompt}] for prompt in problems]
    prompts = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    prompt = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    prompt_len = attention_mask.sum(dim=1)

    attention_mask = torch.cat(
        [
            torch.ones((len(problems), gen_length), device=attention_mask.device, dtype=attention_mask.dtype),
            attention_mask,
        ],
        dim=1,
    )
    attention_mask = attention_mask.repeat(num_generations, 1)

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(device)
    x[:, : prompt.shape[1]] = prompt.clone()
    for i in range(x.shape[0]):
        x[i, prompt_len[i] + gen_length :] = eos_id

    x = x.repeat(num_generations, 1)
    prompt_len = prompt_len.repeat(num_generations)

    trajectory_inputs = []
    trajectory_outputs = []
    update_flags = []
    current_blocks = []
    sample_orders = []
    vgr_remask_flags = []

    batch_size = x.shape[0]
    backtrack_counts = torch.zeros(batch_size, device=x.device, dtype=torch.long)
    prev_values = None

    current_block = torch.zeros((x.shape[0], gen_length), device=x.device, dtype=torch.bool)
    current_block[:, :block_length] = True

    for step in range(steps):
        trajectory_inputs.append(x.clone())
        current_blocks.append(current_block)

        mask_index = x == mask_id
        use_autocast = str(device).startswith("cuda")
        with torch.autocast(device_type="cuda", enabled=use_autocast, dtype=torch.bfloat16):
            outputs = model(x, return_last_hidden_state=True, attention_mask=attention_mask)

        merge_hidden_states = outputs.last_hidden_state
        last_hidden_states = torch.stack(
            [f[int(prompt_len[i]) : int(prompt_len[i]) + gen_length] for i, f in enumerate(merge_hidden_states)]
        )

        logits = outputs.logits / temperature if temperature > 0.0 else outputs.logits
        p = F.softmax(logits.to(prob_dtype), dim=-1)
        pred_out = sample_categorical(p, "hard" if not inference else "max")
        pred_out = torch.where(mask_index, pred_out, x)

        timestep = torch.full(
            (last_hidden_states.shape[0],),
            float(step) / float(steps),
            device=last_hidden_states.device,
        )

        mask_index_gen = torch.stack(
            [im[int(prompt_len[i]) : int(prompt_len[i]) + gen_length] for i, im in enumerate(mask_index)]
        )
        remask_logits = model(
            last_hidden_states,
            pred_mask_prob=True,
            timestep=timestep,
            mask_index=mask_index_gen,
            current_block=current_block,
        )
        remask_logits = remask_logits.masked_fill(~mask_index_gen, -torch.inf)
        remask_logits = remask_logits.masked_fill(~current_block, -torch.inf)
        remask_prob = remask_logits.softmax(-1)
        remask_prob = torch.nan_to_num(remask_prob, nan=0.0, posinf=0.0, neginf=0.0)
        remask_prob = remask_prob.clamp(min=0)
        row_sum = remask_prob.sum(dim=-1, keepdim=True)
        if (row_sum == 0).any():
            fallback = current_block.float()
            fallback = fallback / fallback.sum(dim=-1, keepdim=True).clamp(min=1.0)
            remask_prob = torch.where(row_sum == 0, fallback, remask_prob)

        if inference:
            samples = remask_prob.topk(gen_length // steps).indices
        else:
            samples = torch.multinomial(remask_prob, num_samples=gen_length // steps, replacement=False)
        bs_idx = torch.arange(batch_size, dtype=samples.dtype, device=samples.device).unsqueeze(1)
        update_flag = torch.zeros_like(remask_logits).bool()
        update_flag[bs_idx, samples] = True
        update_index = torch.zeros_like(x).bool()
        update_index[bs_idx, prompt_len.unsqueeze(1) + samples] = True
        sample_orders.append(samples)

        x0 = torch.where(update_index, pred_out, x)

        if step % steps_per_block == steps_per_block - 1:
            current_block = current_block.roll(block_length, 1)

        remask_flag = torch.zeros_like(update_flag).bool()
        if critic is not None and step >= max(gate_start_step - 1, 0):
            with torch.autocast(device_type="cuda", enabled=use_autocast, dtype=torch.bfloat16):
                values = critic(
                    input_ids=x0,
                    attention_mask=attention_mask,
                    prompt_len=prompt_len,
                    gen_length=gen_length,
                    timestep=timestep,
                )
            values = values.float()

            if (
                step >= gate_start_step
                and prev_values is not None
                and retry_m > 0
                and max_backtracks_total > 0
            ):
                gate_mask = values <= (prev_values + min_value_improve)
                gate_mask = gate_mask & (backtrack_counts < max_backtracks_total)

                if gate_mask.any():
                    token_conf = p.max(dim=-1)
                    token_conf_gen = torch.stack(
                        [
                            conf[int(prompt_len[i]) : int(prompt_len[i]) + gen_length]
                            for i, conf in enumerate(token_conf)
                        ]
                    )
                    for i in torch.nonzero(gate_mask, as_tuple=False).flatten().tolist():
                        candidate_mask = update_flag[i]
                        if not candidate_mask.any():
                            continue
                        positions = torch.nonzero(candidate_mask, as_tuple=False).flatten()
                        candidate_uncert = (1.0 - token_conf_gen[i])[candidate_mask]
                        k = min(retry_m, candidate_uncert.numel())
                        if k <= 0:
                            continue
                        topk = candidate_uncert.topk(k).indices
                        chosen = positions[topk]
                        for pos in chosen.tolist():
                            x0[i, int(prompt_len[i]) + pos] = mask_id
                            remask_flag[i, pos] = True
                        backtrack_counts[i] += 1

            prev_values = values

        trajectory_outputs.append(x0.clone())
        update_flags.append(update_flag)
        vgr_remask_flags.append(remask_flag)
        x = x0

    responses = tokenizer.batch_decode(x0, skip_special_tokens=True)
    rewards = reward_fn(batch, responses, num_generations, device).float() if reward_fn is not None else torch.zeros(
        batch_size, device=x.device
    )

    output_dict = {
        "trajectory_inputs": trajectory_inputs,
        "trajectory_outputs": trajectory_outputs,
        "current_blocks": current_blocks,
        "update_flags": update_flags,
        "prompt_len": prompt_len,
        "rewards": rewards,
        "sample_orders": sample_orders,
        "attention_mask": attention_mask,
        "vgr_remask_flags": vgr_remask_flags,
        "backtrack_counts": backtrack_counts,
    }

    return output_dict
