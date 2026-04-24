import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score

from src.data.preprocessing_revised import prepare_data
from src.models.ft_transformer_revised import (
    FTTransformerLike, TrainConfig, make_dataloaders, collect_predictions
)

print("Loading data...")
prepared = prepare_data()

# --- Logistic Regression ---
print("Training Logistic Regression...")
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(prepared.x_train_p, prepared.y_train)
lr_probs = lr.predict_proba(prepared.x_test_p)[:, 1]
y_test = prepared.y_test.to_numpy()

# --- Random Forest ---
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(prepared.x_train_p, prepared.y_train)
rf_probs = rf.predict_proba(prepared.x_test_p)[:, 1]

# --- FT-Transformer ---
print("Training FT-Transformer (this may take a minute)...")
config = TrainConfig()
device = torch.device("cpu")
data = make_dataloaders(config)

model = FTTransformerLike(
    num_features=data.num_features,
    embedding_dim=config.embedding_dim,
    num_heads=config.num_heads,
    num_layers=config.num_layers,
    dropout=config.dropout,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=data.pos_weight.to(device))

best_val_auc = 0
best_state = None
patience_counter = 0

for epoch in range(1, config.epochs + 1):
    model.train()
    for x_batch, y_batch in data.train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x_batch).squeeze(), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_probs, val_targets = collect_predictions(model, data.val_loader, device)
    val_auc = roc_auc_score(val_targets, val_probs)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    ft_probs, ft_targets = collect_predictions(model, data.test_loader, device)

# --- Plot ROC curves ---
fig, ax = plt.subplots(figsize=(8, 6))

models = [
    (y_test,     lr_probs, "Logistic Regression", "#185FA5"),
    (y_test,     rf_probs, "Random Forest",        "#1D9E75"),
    (ft_targets, ft_probs, "FT-Transformer",       "#D85A30"),
]

for targets, probs, label, color in models:
    fpr, tpr, _ = roc_curve(targets, probs)
    score = roc_auc_score(targets, probs)
    ax.plot(fpr, tpr, label=f"{label} (AUC = {score:.2f})", linewidth=2, color=color)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Random chance (AUC = 0.50)")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve Comparison — All Models", fontsize=14)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("roc_comparison.png", dpi=150)
print("\nDone! Chart saved as roc_comparison.png in your project folder.")
