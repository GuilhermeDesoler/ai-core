from __future__ import annotations

import torch
import torch.nn as nn


class LPKTTopicH2Model(nn.Module):
    def __init__(self, num_h2: int, embedding_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()

        self.h2_embedding = nn.Embedding(num_h2, embedding_dim, padding_idx=0)

        self.interaction_proj = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_dim),
            nn.ReLU(),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
        )

        self.learn_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.learn_candidate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh())
        self.forget_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())

        self.output = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        h2 = batch["h2_ids"]
        nh2 = batch["next_h2_ids"]
        c = batch["corrects"].unsqueeze(-1)
        dt = batch["delta_ts"].unsqueeze(-1)
        tr = batch["time_responses"].unsqueeze(-1)

        h2_emb = self.h2_embedding(h2)
        nh2_emb = self.h2_embedding(nh2)

        interaction = torch.cat([h2_emb, c], dim=-1)
        interaction_state = self.interaction_proj(interaction)

        time_state = self.time_proj(torch.cat([dt, tr], dim=-1))

        batch_size, seq_len, _ = interaction_state.shape
        hidden = torch.zeros(batch_size, interaction_state.size(-1), device=interaction_state.device)
        outputs = []

        for t in range(seq_len):
            i_t = interaction_state[:, t, :]
            t_t = time_state[:, t, :]

            learn_in = torch.cat([i_t, t_t], dim=-1)
            learn_gate = self.learn_gate(learn_in)
            learn_candidate = self.learn_candidate(learn_in)

            forget_in = torch.cat([hidden, t_t], dim=-1)
            forget_gate = self.forget_gate(forget_in)

            hidden = forget_gate * hidden + learn_gate * learn_candidate

            out_in = torch.cat([hidden, nh2_emb[:, t, :], t_t], dim=-1)
            outputs.append(self.output(out_in).squeeze(-1))

        return torch.stack(outputs, dim=1)
