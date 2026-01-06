import json
from typing import Optional

import torch
import torch.nn as nn

from networks.layers import TimestepEmbedder


class ValueCritic(nn.Module):
    def __init__(
        self,
        actor: Optional[nn.Module],
        hidden_size: Optional[int] = None,
        pool: str = "mean",
        use_timestep: bool = True,
        mlp_hidden_mult: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.actor = actor
        if self.actor is not None:
            self.actor.eval().requires_grad_(False)
        self.pool = pool
        self.use_timestep = use_timestep
        if hidden_size is None and actor is not None:
            hidden_size = getattr(actor.config, "hidden_size", None)
        self.hidden_size = hidden_size
        if self.hidden_size is None:
            raise ValueError("hidden_size is required when actor has no config.hidden_size.")

        if self.use_timestep:
            self.timestep_embedder = TimestepEmbedder(hidden_size=self.hidden_size)
        else:
            self.timestep_embedder = None

        self.norm = nn.LayerNorm(self.hidden_size)
        mlp_hidden = int(self.hidden_size * mlp_hidden_mult)
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, mlp_hidden),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(mlp_hidden, 1),
        )

    def _pool_hidden(self, hidden_states: torch.Tensor, prompt_len: torch.Tensor, gen_length: int) -> torch.Tensor:
        pooled = []
        for i in range(hidden_states.shape[0]):
            start = int(prompt_len[i])
            end = start + gen_length
            token_states = hidden_states[i, start:end]
            if self.pool == "mean":
                pooled.append(token_states.mean(dim=0))
            elif self.pool == "last":
                pooled.append(token_states[-1])
            else:
                raise ValueError(f"Unknown pool type: {self.pool}")
        return torch.stack(pooled, dim=0)

    def forward_from_pooled(self, pooled: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if self.use_timestep:
            pooled = pooled + self.timestep_embedder(timestep)
        pooled = self.norm(pooled)
        value = self.value_head(pooled).squeeze(-1)
        return value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: torch.Tensor,
        gen_length: int,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if self.actor is None:
            raise ValueError("ValueCritic forward requires actor; use forward_from_pooled for offline training.")
        if not torch.is_tensor(prompt_len):
            prompt_len = torch.tensor(prompt_len, device=input_ids.device)

        with torch.no_grad():
            outputs = self.actor(
                input_ids,
                return_last_hidden_state=True,
                attention_mask=attention_mask,
            )
            hidden_states = outputs.last_hidden_state
            if hidden_states is None and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1]

        pooled = self._pool_hidden(hidden_states, prompt_len, gen_length)
        return self.forward_from_pooled(pooled, timestep)

    def head_state_dict(self) -> dict:
        return {
            "timestep_embedder": self.timestep_embedder.state_dict() if self.timestep_embedder is not None else None,
            "norm": self.norm.state_dict(),
            "value_head": self.value_head.state_dict(),
            "config": {
                "pool": self.pool,
                "use_timestep": self.use_timestep,
                "hidden_size": self.hidden_size,
                "mlp_hidden_mult": self.value_head[0].out_features / float(self.hidden_size),
            },
        }

    def load_head_state_dict(self, state_dict: dict) -> None:
        if state_dict.get("timestep_embedder") is not None and self.timestep_embedder is not None:
            self.timestep_embedder.load_state_dict(state_dict["timestep_embedder"])
        if "norm" in state_dict:
            self.norm.load_state_dict(state_dict["norm"])
        if "value_head" in state_dict:
            self.value_head.load_state_dict(state_dict["value_head"])

    def save_head(self, path: str) -> None:
        torch.save(self.head_state_dict(), path)

    @staticmethod
    def load_head(path: str) -> dict:
        return torch.load(path, map_location="cpu")

    @staticmethod
    def save_config(config: dict, path: str) -> None:
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
