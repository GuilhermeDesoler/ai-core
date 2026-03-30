from __future__ import annotations

import torch
import torch.nn as nn


class DKTNextItemModel(nn.Module):
    def __init__(self, num_questions, num_skills, emb_dim=128, hidden_dim=256, dropout=0.2):
        super().__init__()

        self.q_emb = nn.Embedding(num_questions, emb_dim, padding_idx=0)
        self.s_emb = nn.Embedding(num_skills, emb_dim, padding_idx=0)

        self.next_q_emb = nn.Embedding(num_questions, emb_dim, padding_idx=0)
        self.next_s_emb = nn.Embedding(num_skills, emb_dim, padding_idx=0)

        self.input_proj = nn.Linear(emb_dim * 2 + 3, emb_dim)

        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.output = nn.Sequential(
            nn.Linear(hidden_dim + emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch):
        q = batch["question_ids"]
        s = batch["skill_ids"]
        nq = batch["next_question_ids"]
        ns = batch["next_skill_ids"]
        c = batch["corrects"].unsqueeze(-1)
        dt = batch["delta_ts"].unsqueeze(-1)
        tr = batch["time_responses"].unsqueeze(-1)

        q_emb = self.q_emb(q)
        s_emb = self.s_emb(s)

        x = torch.cat([q_emb, s_emb, c, dt, tr], dim=-1)
        x = self.input_proj(x)

        h, _ = self.lstm(x)
        h = self.dropout(h)

        nq_emb = self.next_q_emb(nq)
        ns_emb = self.next_s_emb(ns)

        out = torch.cat([h, nq_emb, ns_emb], dim=-1)
        logits = self.output(out).squeeze(-1)

        return logits
