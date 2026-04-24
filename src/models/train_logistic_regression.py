from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report
from src.data.preprocessing_revised import prepare_data
# Load prepared data using  function

prepared = prepare_data(save_artifacts=True)

# Initialize the Logistic Regression Model

baseline_model = LogisticRegression(
    class_weight='balanced', 
    max_iter=1000,           # Increased to ensure the model converges
    random_state=42
)

# 3. Train the Model
print("Training Logistic Regression baseline...")
baseline_model.fit(prepared.x_train_p, prepared.y_train)

#Make Predictions

y_pred = baseline_model.predict(prepared.x_test_p)

# Need probability predictions for the AUC-ROC curve
# [:, 1] gets the probability of the positive class (which is mapped as 1 = Bad/Rejected)
y_pred_proba = baseline_model.predict_proba(prepared.x_test_p)[:, 1]

# Evaluate the Model against Success Metrics
auc_roc = roc_auc_score(prepared.y_test, y_pred_proba)
f1 = f1_score(prepared.y_test, y_pred)
conf_matrix = confusion_matrix(prepared.y_test, y_pred)

print("\n--- Baseline Model Evaluation ---")
print(f"Primary Metric (AUC-ROC): {auc_roc:.4f}")
print(f"Secondary Metric (F1 Score): {f1:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(prepared.y_test, y_pred))