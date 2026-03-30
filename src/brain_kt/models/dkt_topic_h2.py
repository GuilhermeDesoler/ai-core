from __future__ import annotations

import torch
import torch.nn as nn


class DKTTopicH2Model(nn.Module):
    def __init__(self, num_h2, emb_dim=128, hidden_dim=256, dropout=0.2):
        super().__init__()

        self.h2_emb = nn.Embedding(num_h2, emb_dim, padding_idx=0)
        self.next_h2_emb = nn.Embedding(num_h2, emb_dim, padding_idx=0)

        self.input_proj = nn.Linear(emb_dim + 3, emb_dim)

        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.output = nn.Sequential(
            nn.Linear(hidden_dim + emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch):
        h2 = batch["h2_ids"]
        nh2 = batch["next_h2_ids"]
        c = batch["corrects"].unsqueeze(-1)
        dt = batch["delta_ts"].unsqueeze(-1)
        tr = batch["time_responses"].unsqueeze(-1)

        h2_emb = self.h2_emb(h2)
        x = torch.cat([h2_emb, c, dt, tr], dim=-1)
        x = self.input_proj(x)

        h, _ = self.lstm(x)
        h = self.dropout(h)

        nh2_emb = self.next_h2_emb(nh2)

        out = torch.cat([h, nh2_emb], dim=-1)
        logits = self.output(out).squeeze(-1)

        return logits
