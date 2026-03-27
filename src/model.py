import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class InsiderThreatLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 16,
        embed_dim: int = 32,
        lstm1_hidden: int = 128,
        lstm2_hidden: int = 64,
        num_features: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm1 = nn.LSTM(
            embed_dim, lstm1_hidden,
            batch_first=True, dropout=dropout,
        )
        self.lstm2 = nn.LSTM(
            lstm1_hidden, lstm2_hidden,
            batch_first=True, dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm2_hidden + num_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, seq_input: torch.Tensor, feat_input: torch.Tensor) -> torch.Tensor:
        # seq_input : (batch, 200) long
        # feat_input: (batch, 6)  float32
        x = self.embedding(seq_input)           # (batch, 200, 32)
        x, _ = self.lstm1(x)                    # (batch, 200, 128)
        x, _ = self.lstm2(x)                    # (batch, 200, 64)
        x = x[:, -1, :]                         # (batch, 64)  last hidden state
        x = torch.cat([x, feat_input], dim=1)   # (batch, 70)
        x = self.classifier(x)                  # (batch, 1)
        return x.squeeze(1)                     # (batch,)  raw logits


class ThreatDataset(Dataset):
    def __init__(
        self,
        seq_array: np.ndarray,
        feat_array: np.ndarray,
        label_array: np.ndarray,
    ):
        self.seq   = torch.from_numpy(seq_array).long()
        self.feat  = torch.from_numpy(feat_array).float()
        self.label = torch.from_numpy(label_array.astype(np.float32))

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, idx: int):
        return self.seq[idx], self.feat[idx], self.label[idx]


if __name__ == "__main__":
    model = InsiderThreatLSTM()
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"\nTotal params    : {total:,}")
    print(f"Trainable params: {trainable:,}")

    # Smoke-test forward pass
    batch = 4
    seq  = torch.randint(0, 16, (batch, 200))
    feat = torch.randn(batch, 6)
    out  = model(seq, feat)
    print(f"\nForward pass OK — output shape: {out.shape}  (expected [{batch}])")
