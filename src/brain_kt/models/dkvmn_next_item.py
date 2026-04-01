from __future__ import annotations

import torch
import torch.nn as nn


class DKVMNNextItemModel(nn.Module):
    def __init__(
        self,
        num_questions: int,
        num_skills: int,
        memory_size: int = 50,
        key_dim: int = 64,
        value_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.q_emb = nn.Embedding(num_questions, key_dim, padding_idx=0)
        self.s_emb = nn.Embedding(num_skills, key_dim, padding_idx=0)
        self.next_q_emb = nn.Embedding(num_questions, key_dim, padding_idx=0)
        self.next_s_emb = nn.Embedding(num_skills, key_dim, padding_idx=0)

        self.key_memory = nn.Parameter(torch.randn(memory_size, key_dim) * 0.1)
        self.init_value_memory = nn.Parameter(torch.randn(memory_size, value_dim) * 0.1)

        self.interaction_proj = nn.Sequential(
            nn.Linear(key_dim + 3, value_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.erase_layer = nn.Linear(value_dim, value_dim)
        self.add_layer = nn.Linear(value_dim, value_dim)

        self.output = nn.Sequential(
            nn.Linear(value_dim + key_dim, value_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(value_dim, 1),
        )

    def _attention(self, query: torch.Tensor) -> torch.Tensor:
        # query: [B, key_dim]
        logits = torch.matmul(query, self.key_memory.t())
        return torch.softmax(logits, dim=-1)

    def _read(self, value_memory: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # value_memory: [B, M, V], weights: [B, M]
        return torch.sum(value_memory * weights.unsqueeze(-1), dim=1)

    def _write(
        self,
        value_memory: torch.Tensor,
        weights: torch.Tensor,
        interaction: torch.Tensor,
    ) -> torch.Tensor:
        erase = torch.sigmoid(self.erase_layer(interaction))
        add = torch.tanh(self.add_layer(interaction))

        erase_term = 1.0 - weights.unsqueeze(-1) * erase.unsqueeze(1)
        add_term = weights.unsqueeze(-1) * add.unsqueeze(1)
        return value_memory * erase_term + add_term

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        q = batch["question_ids"]
        s = batch["skill_ids"]
        nq = batch["next_question_ids"]
        ns = batch["next_skill_ids"]
        c = batch["corrects"].unsqueeze(-1)
        dt = batch["delta_ts"].unsqueeze(-1)
        tr = batch["time_responses"].unsqueeze(-1)

        batch_size, seq_len = q.shape
        device = q.device

        value_memory = (
            self.init_value_memory.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        )
        logits_steps = []

        for t in range(seq_len):
            current_query = self.q_emb(q[:, t]) + self.s_emb(s[:, t])
            next_query = self.next_q_emb(nq[:, t]) + self.next_s_emb(ns[:, t])

            interaction_input = torch.cat(
                [current_query, c[:, t], dt[:, t], tr[:, t]], dim=-1
            )
            interaction_state = self.interaction_proj(interaction_input)

            write_weights = self._attention(current_query)
            value_memory = self._write(value_memory, write_weights, interaction_state)

            read_weights = self._attention(next_query)
            read_content = self._read(value_memory, read_weights)

            pred_input = torch.cat([read_content, next_query], dim=-1)
            logits_t = self.output(pred_input).squeeze(-1)
            logits_steps.append(logits_t)

        return torch.stack(logits_steps, dim=1)
