import os
import yaml
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import DG_Dataset
from model import BinaryResNet18

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_id"]

# Data Paths
csv_file = config["csv_file"] 
data_dir = config["data_dir"]
load_path = "./checkpoints/DG-Origin/Test-Run/E_15_Loss_0.7268.pt"

test_domains = config["test_domains"]
print(f"Testing on domains: {test_domains}")
transform = transforms.Compose([transforms.ToTensor()])
test_ds = DG_Dataset(csv_file = csv_file, domains = test_domains, data_dir = data_dir, shift_feature = "Origin", transform = transform)
print(f"Test samples: {len(test_ds)}")
test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=4)

# Print dataset information
print("Class to index mapping:", test_ds.class_to_idx)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BinaryResNet18(dropout_p=0.5).to(device)
model.load_state_dict(torch.load(load_path, map_location=device))

# Make predictions
model.eval()
predictions = []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="> Testing"):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        logits = logits.view(-1)
        preds = torch.sigmoid(logits) > 0.5
        predictions.extend(preds.cpu().numpy().tolist())


# Calculate confusion matrix and classification report
y_true = [label for _, label in test_loader.dataset]
y_pred = [int(pred) for pred in predictions]
conf_matrix = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names= test_ds.class_to_idx.keys())

print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(report)