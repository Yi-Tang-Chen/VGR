import json
import os
import random
import time

import click
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from networks.value_critic import ValueCritic


# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_feature_dataset(path: str) -> Dict:
    data = torch.load(path, map_location="cpu")
    for key in ("features", "timesteps", "rewards"):
        if key not in data:
            raise KeyError(f"Missing '{key}' in dataset: {path}")
    return data


# -----------------------------
# Main
# -----------------------------

@click.command()
@click.option("--dataset_path", type=str, default="datasets_cache/critic_train.pt")
@click.option("--critic_out_dir", type=str, default="critic_ckpt")
@click.option("--batch_size", type=int, default=512)
@click.option("--num_workers", type=int, default=2)
@click.option("--epochs", type=int, default=5)
@click.option("--max_steps", type=int, default=0)
@click.option("--lr", type=float, default=2e-4)
@click.option("--weight_decay", type=float, default=0.01)
@click.option("--loss_type", type=str, default="bce")
@click.option("--mlp_hidden_mult", type=float, default=1.0)
@click.option("--dropout", type=float, default=0.1)
@click.option("--seed", type=int, default=113)
@click.option("--log_every", type=int, default=50)
@click.option("--save_every", type=int, default=0)
def main(
    dataset_path,
    critic_out_dir,
    batch_size,
    num_workers,
    epochs,
    max_steps,
    lr,
    weight_decay,
    loss_type,
    mlp_hidden_mult,
    dropout,
    seed,
    log_every,
    save_every,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    os.makedirs(critic_out_dir, exist_ok=True)

    data = load_feature_dataset(dataset_path)
    features = data["features"]
    timesteps = data["timesteps"]
    rewards = data["rewards"]
    meta = data.get("meta", {})

    if features.ndim != 2:
        raise ValueError(f"Expected features with shape (N, D), got {features.shape}")

    hidden_size = features.shape[1]
    use_timestep = meta.get("use_timestep", True)
    pool = meta.get("pool", "mean")

    dataset = TensorDataset(features, timesteps, rewards)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    critic = ValueCritic(
        actor=None,
        hidden_size=hidden_size,
        pool=pool,
        use_timestep=use_timestep,
        mlp_hidden_mult=mlp_hidden_mult,
        dropout=dropout,
    ).to(device)

    params = [p for p in critic.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    if loss_type == "bce":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif loss_type == "mse":
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    total_steps = max_steps if max_steps > 0 else epochs * len(dataloader)
    global_step = 0

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"epoch {epoch+1}/{epochs}")
        for feats, ts, r in pbar:
            if max_steps and global_step >= max_steps:
                break

            feats = feats.to(device, non_blocking=True).float()
            ts = ts.to(device, non_blocking=True).float()
            r = r.to(device, non_blocking=True).float()

            t0 = time.perf_counter()
            critic.train()
            optimizer.zero_grad(set_to_none=True)

            values = critic.forward_from_pooled(feats, ts)
            loss = criterion(values.float(), r.float())
            loss.backward()
            optimizer.step()

            train_time = time.perf_counter() - t0

            with torch.no_grad():
                probs = torch.sigmoid(values.float())
                preds = (probs >= 0.5).float()
                acc = (preds == r.float()).float().mean().item()
                reward_mean = r.mean().item()
                pred_mean = preds.mean().item()

            if log_every > 0 and global_step % log_every == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{acc:.3f}",
                    rmean=f"{reward_mean:.3f}",
                    pmean=f"{pred_mean:.3f}",
                    t=f"{train_time:.4f}s",
                )

            if save_every > 0 and global_step % save_every == 0 and global_step > 0:
                save_path = os.path.join(critic_out_dir, f"critic_head_step{global_step}.pt")
                torch.save({"state_dict": critic.head_state_dict(), "step": global_step}, save_path)

            global_step += 1
            if global_step >= total_steps:
                break

        if max_steps and global_step >= max_steps:
            break

        save_path = os.path.join(critic_out_dir, f"critic_head_epoch{epoch+1}.pt")
        torch.save({"state_dict": critic.head_state_dict(), "step": global_step}, save_path)

    config = {
        "dataset_path": dataset_path,
        "epochs": epochs,
        "batch_size": batch_size,
        "loss_type": loss_type,
        "lr": lr,
        "weight_decay": weight_decay,
        "mlp_hidden_mult": mlp_hidden_mult,
        "dropout": dropout,
        "seed": seed,
        "hidden_size": hidden_size,
        "use_timestep": use_timestep,
        "pool": pool,
        "meta": meta,
    }
    with open(os.path.join(critic_out_dir, "critic_config.json"), "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
