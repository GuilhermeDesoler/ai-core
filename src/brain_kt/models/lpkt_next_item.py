from __future__ import annotations

import torch
import torch.nn as nn


class LPKTNextItemModel(nn.Module):
    def __init__(
        self,
        num_questions: int,
        num_skills: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.question_embedding = nn.Embedding(num_questions, embedding_dim, padding_idx=0)
        self.skill_embedding = nn.Embedding(num_skills, embedding_dim, padding_idx=0)
        self.next_question_embedding = nn.Embedding(num_questions, embedding_dim, padding_idx=0)
        self.next_skill_embedding = nn.Embedding(num_skills, embedding_dim, padding_idx=0)

        self.interaction_proj = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
        )

        self.learn_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.learn_candidate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
        )
        self.forget_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        q = batch["question_ids"]
        s = batch["skill_ids"]
        nq = batch["next_question_ids"]
        ns = batch["next_skill_ids"]
        c = batch["corrects"].unsqueeze(-1)
        dt = batch["delta_ts"].unsqueeze(-1)
        tr = batch["time_responses"].unsqueeze(-1)

        q_emb = self.question_embedding(q)
        s_emb = self.skill_embedding(s)
        nq_emb = self.next_question_embedding(nq)
        ns_emb = self.next_skill_embedding(ns)

        interaction = torch.cat([q_emb, s_emb, c], dim=-1)
        interaction_state = self.interaction_proj(interaction)

        time_features = torch.cat([dt, tr], dim=-1)
        time_state = self.time_proj(time_features)

        batch_size, seq_len, _ = interaction_state.shape
        hidden = torch.zeros(batch_size, interaction_state.size(-1), device=interaction_state.device)
        logits_steps = []

        for t in range(seq_len):
            interaction_t = interaction_state[:, t, :]
            time_t = time_state[:, t, :]

            learn_input = torch.cat([interaction_t, time_t], dim=-1)
            learn_gate = self.learn_gate(learn_input)
            learn_candidate = self.learn_candidate(learn_input)

            forget_input = torch.cat([hidden, time_t], dim=-1)
            forget_gate = self.forget_gate(forget_input)

            hidden = forget_gate * hidden + learn_gate * learn_candidate

            pred_input = torch.cat([hidden, nq_emb[:, t, :], ns_emb[:, t, :], time_t], dim=-1)
            logits_t = self.output_head(pred_input).squeeze(-1)
            logits_steps.append(logits_t)

        return torch.stack(logits_steps, dim=1)
