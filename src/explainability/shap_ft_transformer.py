from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

# Allow this file to run either as part of the src package or directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) >= 3 else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from src.data.preprocessing_revised import prepare_data
    from src.models.ft_transformer_revised import FTTransformerLike, TrainConfig, _to_dense_float32, set_seed
except ModuleNotFoundError:
    from preprocessing_revised import prepare_data
    from ft_transformer_revised import FTTransformerLike, TrainConfig, _to_dense_float32, set_seed


def load_trained_model(model_path: str | Path, device: torch.device, config: TrainConfig, num_features: int) -> FTTransformerLike:
    model = FTTransformerLike(
        num_features=num_features,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_feature_names(prepared, num_features):
    x_train_p = prepared.x_train_p

    # Case 1: pandas DataFrame
    if hasattr(x_train_p, "columns"):
        return list(x_train_p.columns)

    # Case 2: sklearn pipeline
    if hasattr(prepared, "preprocessor"):
        try:
            return list(prepared.preprocessor.get_feature_names_out())
        except:
            pass

    # Fallback
    return [f"feature_{i}" for i in range(num_features)]


def make_model_predict(model: FTTransformerLike, device: torch.device):
    def model_predict(X_numpy, batch_size: int = 256):
        """
        SHAP wrapper: numpy array -> numpy probabilities for positive class.
        """
        X_numpy = np.asarray(X_numpy, dtype=np.float32)
        outputs = []

        for i in range(0, len(X_numpy), batch_size):
            batch = X_numpy[i:i + batch_size]
            X_tensor = torch.tensor(batch, dtype=torch.float32).to(device)

            with torch.no_grad():
                logits = model(X_tensor)
                probs = torch.sigmoid(logits)
                outputs.append(probs.cpu().numpy())

        return np.concatenate(outputs, axis=0)

    return model_predict


def main() -> None:
    config = TrainConfig()
    set_seed(config.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    prepared = prepare_data(
        test_size=config.test_size,
        val_size=config.train_size_val,
        random_state=config.random_seed,
        save_artifacts=False,
    )

    x_train = _to_dense_float32(prepared.x_train_p)
    x_test = _to_dense_float32(prepared.x_test_p)
    num_features = x_train.shape[1]
    feature_names = get_feature_names(prepared, num_features)

    print("Number of processed input features:", num_features)

    # Change this path if your saved model is elsewhere.
    model_path = PROJECT_ROOT / "artifacts" / "ft_transformer_best.pt"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find saved model at: {model_path}\n"
            "Save the best model weights first, then run this SHAP script."
        )

    model = load_trained_model(
        model_path=model_path,
        device=device,
        config=config,
        num_features=num_features,
    )

    model_predict = make_model_predict(model, device)

    # Keep these modest at first because KernelExplainer can be slow.
    background_size = min(50, len(x_train))
    explain_size = min(20, len(x_test))

    background_idx = np.random.choice(len(x_train), background_size, replace=False)
    background = x_train[background_idx]
    x_explain = x_test[:explain_size]

    print(f"Background sample size: {background_size}")
    print(f"Rows to explain: {explain_size}")

    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(x_explain)

    # Summary bar plot
    shap.summary_plot(
        shap_values,
        x_explain,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig("shap_summary_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Summary beeswarm plot
    shap.summary_plot(
        shap_values,
        x_explain,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig("shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved SHAP plots:")
    print("- shap_summary_bar.png")
    print("- shap_summary_beeswarm.png")


if __name__ == "__main__":
    main()
