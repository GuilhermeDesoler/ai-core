from __future__ import annotations

import torch
import torch.nn as nn


class LPKTTopicH3Model(nn.Module):
    def __init__(
        self,
        num_h3: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.h3_embedding = nn.Embedding(num_h3, embedding_dim, padding_idx=0)

        self.interaction_proj = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_dim),
            nn.ReLU(),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
        )

        self.learn_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid()
        )
        self.learn_candidate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh()
        )
        self.forget_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.Sigmoid()
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        h3 = batch["h3_ids"]
        nh3 = batch["next_h3_ids"]
        c = batch["corrects"].unsqueeze(-1)
        dt = torch.log1p(batch["delta_ts"].unsqueeze(-1))
        tr = torch.log1p(batch["time_responses"].unsqueeze(-1))

        h3_emb = self.h3_embedding(h3)
        nh3_emb = self.h3_embedding(nh3)

        interaction = torch.cat([h3_emb, c], dim=-1)
        interaction_state = self.interaction_proj(interaction)

        time_state = self.time_proj(torch.cat([dt, tr], dim=-1))

        batch_size, seq_len, _ = interaction_state.shape
        hidden = torch.zeros(
            batch_size, interaction_state.size(-1), device=interaction_state.device
        )
        outputs = []

        for t in range(seq_len):
            # if t == 0:
            #     print(f"[LPKT] Processando timesteps de 0 a {seq_len-1}")

            i_t = interaction_state[:, t, :]
            t_t = time_state[:, t, :]

            learn_in = torch.cat([i_t, t_t], dim=-1)
            learn_gate = self.learn_gate(learn_in)
            learn_candidate = self.learn_candidate(learn_in)

            forget_in = torch.cat([hidden, i_t, t_t], dim=-1)
            forget_gate = self.forget_gate(forget_in)

            hidden = forget_gate * hidden + learn_gate * learn_candidate

            out_in = torch.cat([hidden, nh3_emb[:, t, :], t_t], dim=-1)
            outputs.append(self.output(out_in).squeeze(-1))

        return torch.stack(outputs, dim=1)
