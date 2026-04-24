from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report

import sys
from pathlib import Path

# Allow Python to find the 'src' folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.preprocessing_revised import prepare_data

print("Loading data...")
prepared = prepare_data()

# Initialize the Random Forest Model
# We still use class_weight='balanced' to handle your imbalanced dataset constraint
rf_model = RandomForestClassifier(
    n_estimators=100, 
    class_weight='balanced', 
    random_state=42,
    n_jobs=-1 # Uses all your CPU cores to train faster
)

print("Training Random Forest baseline...")
rf_model.fit(prepared.x_train_p, prepared.y_train)

# Make Predictions
y_pred = rf_model.predict(prepared.x_test_p)
# Get probabilities for the AUC-ROC curve
y_pred_proba = rf_model.predict_proba(prepared.x_test_p)[:, 1] 

# Evaluate Success Metrics
auc_roc = roc_auc_score(prepared.y_test, y_pred_proba)
f1 = f1_score(prepared.y_test, y_pred)
conf_matrix = confusion_matrix(prepared.y_test, y_pred)

print("\n--- Random Forest Evaluation ---")
print(f"Primary Metric (AUC-ROC): {auc_roc:.4f}")
print(f"Secondary Metric (F1 Score): {f1:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(prepared.y_test, y_pred))