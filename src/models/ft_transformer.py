# src/models/ft_transformer.py

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.data.preprocessing import prepare_data


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    embedding_dim: int = 32
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    random_seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class FTTransformerLike(nn.Module):
    """
    A simple tabular Transformer-style classifier.

    Input: processed feature matrix of shape (batch_size, num_features)
    Idea:
      - treat each scalar feature as a token
      - project each feature into an embedding
      - add a learnable CLS token
      - pass through Transformer encoder
      - use CLS representation for binary classification
    """

    def __init__(
        self,
        num_features: int,
        embedding_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.embedding_dim = embedding_dim

        # Each scalar feature gets projected into embedding_dim dimensions
        self.feature_projection = nn.Linear(1, embedding_dim)

        # Learnable per-feature embeddings so feature identity is preserved
        self.feature_embeddings = nn.Parameter(
            torch.randn(num_features, embedding_dim)
        )

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, num_features)
        returns logits shape: (batch_size,)
        """
        batch_size, num_features = x.shape

        # Convert each scalar feature into shape (batch_size, num_features, 1)
        x = x.unsqueeze(-1)

        # Project each feature token
        x = self.feature_projection(x)  # (batch_size, num_features, embedding_dim)

        # Add learned feature identity embeddings
        x = x + self.feature_embeddings.unsqueeze(0)

        # Add CLS token at the start
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Transformer encoding
        x = self.transformer(x)

        # Use CLS output
        cls_output = x[:, 0, :]
        logits = self.classifier(cls_output).squeeze(1)

        return logits


def make_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader, int]:
    prepared = prepare_data()

    X_train = prepared.X_train_p.toarray().astype(np.float32)
    X_test = prepared.X_test_p.toarray().astype(np.float32)
    y_train = prepared.y_train.to_numpy().astype(np.float32)
    y_test = prepared.y_test.to_numpy().astype(np.float32)

    train_dataset = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test),
        torch.tensor(y_test),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    num_features = X_train.shape[1]
    return train_loader, test_loader, num_features


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_targets.extend(y_batch.numpy().astype(int).tolist())

    auc = roc_auc_score(all_targets, all_probs)
    f1 = f1_score(all_targets, all_preds)

    return auc, f1


def main() -> None:
    config = TrainConfig()
    set_seed(config.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, test_loader, num_features = make_dataloaders(config)
    print("Number of processed input features:", num_features)

    model = FTTransformerLike(
        num_features=num_features,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        auc, f1 = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test AUC: {auc:.4f} | "
            f"Test F1: {f1:.4f}"
        )


if __name__ == "__main__":
    main()