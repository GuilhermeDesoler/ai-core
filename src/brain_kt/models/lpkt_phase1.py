import torch
import torch.nn as nn

class LPKTPhase1(nn.Module):
    def __init__(self, n_q, n_s, hidden_dim=128):
        super().__init__()
        self.q_emb = nn.Embedding(n_q, hidden_dim)
        self.s_emb = nn.Embedding(n_s, hidden_dim)

        self.fc = nn.Linear(hidden_dim * 2 + 5, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, batch):
        q = self.q_emb(batch["question_ids"])
        s = self.s_emb(batch["skill_ids"])

        extra = torch.stack([
            batch["delta_ts"],
            batch["time_responses"],
            batch.get("skill_attempts", torch.zeros_like(batch["delta_ts"])),
            batch.get("rolling_acc", torch.zeros_like(batch["delta_ts"])),
            batch.get("streak", torch.zeros_like(batch["delta_ts"])),
        ], dim=-1)

        x = torch.cat([q, s, extra], dim=-1)
        x = self.fc(x)

        h, _ = self.lstm(x)
        logits = self.out(h).squeeze(-1)
        return logits
