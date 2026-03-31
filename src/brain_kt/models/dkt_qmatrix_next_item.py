from __future__ import annotations

import torch
import torch.nn as nn


class DKTQMatrixModel(nn.Module):
    def __init__(
        self,
        num_concepts: int,
        num_skills: int,
        emb_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.concept_emb = nn.Embedding(num_concepts, emb_dim, padding_idx=0)
        self.skill_emb = nn.Embedding(num_skills, emb_dim, padding_idx=0)

        input_dim = emb_dim * 3 + 3  # concept + next concept + skill + features

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def pool_concepts(self, concept_ids, concept_mask):
        emb = self.concept_emb(concept_ids)
        mask = concept_mask.unsqueeze(-1)
        summed = (emb * mask).sum(dim=2)
        count = mask.sum(dim=2).clamp(min=1)
        return summed / count

    def forward(self, batch):
        c = self.pool_concepts(batch["concept_ids"], batch["concept_mask"])
        nc = self.pool_concepts(batch["next_concept_ids"], batch["next_concept_mask"])
        s = self.skill_emb(batch["skill_ids"])

        x = torch.cat([
            c,
            nc,
            s,
            batch["corrects"].unsqueeze(-1),
            batch["delta_ts"].unsqueeze(-1),
            batch["time_responses"].unsqueeze(-1),
        ], dim=-1)

        out, _ = self.lstm(x)
        out = self.dropout(out)
        logits = self.fc(out).squeeze(-1)
        return logits
