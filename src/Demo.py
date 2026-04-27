import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) >= 3 else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.ft_transformer_revised import FTTransformerLike

config = torch.load('model_config.pt')

#Initialize model architecture using the loaded blueprint
model = FTTransformerLike(
    num_features=config['num_features'],
    embedding_dim=config['embedding_dim'],
    num_heads=config['num_heads'],
    num_layers=config['num_layers'],
    dropout=config['dropout']
)

#Load the weights
model.load_state_dict(torch.load('ft_transformer_weights.pth'))
model.eval()

#Load the sample
sample_applicant = torch.load('sample_applicant_data.pt')

#Make the Prediction
print("Feeding applicant data into the FT-Transformer...")

with torch.no_grad():
    raw_logit = model(sample_applicant)
    probability = torch.sigmoid(raw_logit).item() * 100

print(f"LIVE SYSTEM OUTPUT: The FT-Transformer predicts a {probability:.1f}% risk of default for this applicant.")