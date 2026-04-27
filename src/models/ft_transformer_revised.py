from __future__ import annotations

import copy
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) >= 3 else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.preprocessing_revised import prepare_data

'''
first try
@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    embedding_dim: int = 32
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    random_seed: int = 42
    train_size_val: float = 0.15
    test_size: float = 0.15
    early_stopping_patience: int = 8
    gradient_clip_norm: float = 1.0
'''
@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    embedding_dim: int = 64
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.35
    random_seed: int = 42
    train_size_val: float = 0.15
    test_size: float = 0.15
    early_stopping_patience: int = 15
    gradient_clip_norm: float = 1.0

@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_features: int
    pos_weight: torch.Tensor


@dataclass
class EvalResult:
    auc: float
    f1: float
    threshold: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class FTTransformerLike(nn.Module):
    """
    A simple Transformer-style classifier for processed tabular features.
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

        self.feature_projection = nn.Linear(1, embedding_dim)
        self.feature_embeddings = nn.Parameter(torch.randn(num_features, embedding_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _ = x.shape
        x = x.unsqueeze(-1)
        x = self.feature_projection(x)
        x = x + self.feature_embeddings.unsqueeze(0)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.transformer(x)

        cls_output = x[:, 0, :]
        logits = self.classifier(cls_output).squeeze(1)
        return logits


def _to_dense_float32(matrix: object) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def make_dataloaders(config: TrainConfig) -> DataBundle:
    prepared = prepare_data(
        test_size=config.test_size,
        val_size=config.train_size_val,
        random_state=config.random_seed,
        save_artifacts=False,
    )

    x_train = _to_dense_float32(prepared.x_train_p)
    x_val = _to_dense_float32(prepared.x_val_p)
    x_test = _to_dense_float32(prepared.x_test_p)

    y_train = prepared.y_train.to_numpy(dtype=np.float32)
    y_val = prepared.y_val.to_numpy(dtype=np.float32)
    y_test = prepared.y_test.to_numpy(dtype=np.float32)

    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    positive_count = max(float(y_train.sum()), 1.0)
    negative_count = max(float(len(y_train) - y_train.sum()), 1.0)
    pos_weight = torch.tensor(negative_count / positive_count, dtype=torch.float32)

    sample_applicant = torch.tensor(x_test[0], dtype=torch.float32).unsqueeze(0)
    torch.save(sample_applicant, 'sample_applicant_data.pt')
    print("Applicant data successfully saved")


    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_features=x_train.shape[1],
        pos_weight=pos_weight,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()

        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)

        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)

    return total_loss / len(loader.dataset)


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs: list[float] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs.tolist())
            all_targets.extend(y_batch.numpy().astype(int).tolist())

    return np.asarray(all_probs), np.asarray(all_targets)


def select_best_threshold(probs: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.linspace(0.10, 0.90, 81):
        preds = (probs >= threshold).astype(int)
        score = f1_score(targets, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    tune_threshold: bool = False,
) -> EvalResult:
    probs, targets = collect_predictions(model, loader, device)

    auc = roc_auc_score(targets, probs)
    if tune_threshold:
        best_threshold, best_f1 = select_best_threshold(probs, targets)
        return EvalResult(auc=auc, f1=best_f1, threshold=best_threshold)

    preds = (probs >= threshold).astype(int)
    f1 = f1_score(targets, preds, zero_division=0)
    return EvalResult(auc=auc, f1=f1, threshold=threshold)


def main() -> None:
    config = TrainConfig()
    set_seed(config.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data = make_dataloaders(config)
    print("Number of processed input features:", data.num_features)

    model = FTTransformerLike(
        num_features=data.num_features,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    model_config = {
    'num_features': data.num_features,
    'embedding_dim':config.embedding_dim,
    'num_heads': config.num_heads,
    'num_layers': config.num_layers,
    'dropout': config.dropout,
    }

    torch.save(model_config, 'model_config.pt')
    print("Model blueprint successfully saved!")

    criterion = nn.BCEWithLogitsLoss(pos_weight=data.pos_weight.to(device))
    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
    )

    best_state = None
    best_val_auc = -float("inf")
    best_threshold = 0.5
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=data.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_clip_norm=config.gradient_clip_norm,
        )

        val_result = evaluate(
            model=model,
            loader=data.val_loader,
            device=device,
            tune_threshold=True,
        )

        scheduler.step(val_result.auc)

        improved = val_result.auc > best_val_auc
        if improved:
            best_val_auc = val_result.auc
            best_threshold = val_result.threshold
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Save the trained weights to a file
        torch.save(model.state_dict(), 'ft_transformer_weights.pth')
        print("Model weights successfully saved!")

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val AUC: {val_result.auc:.4f} | "
            f"Val F1: {val_result.f1:.4f} | "
            f"Val Threshold: {val_result.threshold:.2f}"
        )

        if epochs_without_improvement >= config.early_stopping_patience:
            print(
                f"Early stopping triggered after {epoch} epochs "
                f"(no validation AUC improvement for {config.early_stopping_patience} epochs)."
            )
            break

    if best_state is None:
        raise RuntimeError("Training finished without saving a best model state.")

    # Load best model weights back into the model
    model.load_state_dict(best_state)

    # Save best model weights for SHAP
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    best_model_path = artifacts_dir / "ft_transformer_best.pt"
    torch.save(best_state, best_model_path)
    print(f"Saved best model to: {best_model_path}")

    test_result = evaluate(
        model=model,
        loader=data.test_loader,
        device=device,
        threshold=best_threshold,
        tune_threshold=False,
    )

    print("\nFinal test results using best validation checkpoint:")
    print(f"Test AUC: {test_result.auc:.4f}")
    print(f"Test F1: {test_result.f1:.4f}")
    print(f"Threshold used: {best_threshold:.2f}")
    
    #Collect the raw probabilities and true labels from the test set
    probs, targets = collect_predictions(model, data.test_loader, device)
    
    #Convert probabilities to strict 0 or 1 predictions using your best threshold
    preds = (probs >= best_threshold).astype(int)
    
    #Generate and print the metrics
    print("\nConfusion Matrix:\n", confusion_matrix(targets, preds))
    print("\nClassification Report:\n", classification_report(targets, preds))

if __name__ == "__main__":
    main()