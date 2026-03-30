from __future__ import annotations

import torch
import torch.nn as nn


class DKTModel(nn.Module):
    def __init__(
        self,
        num_questions: int,
        num_skills: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.question_embedding = nn.Embedding(num_questions, embedding_dim, padding_idx=0)
        self.skill_embedding = nn.Embedding(num_skills, embedding_dim, padding_idx=0)

        self.input_proj = nn.Linear(embedding_dim * 2 + 3, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        q = batch["question_ids"]
        s = batch["skill_ids"]
        c = batch["corrects"].unsqueeze(-1)
        dt = batch["delta_ts"].unsqueeze(-1)
        tr = batch["time_responses"].unsqueeze(-1)

        q_emb = self.question_embedding(q)
        s_emb = self.skill_embedding(s)

        x = torch.cat([q_emb, s_emb, c, dt, tr], dim=-1)

        x = self.input_proj(x)

        h, _ = self.lstm(x)

        logits = self.output(h).squeeze(-1)

        return logits
