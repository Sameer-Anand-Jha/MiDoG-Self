import os
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import DG_Dataset
from model import BinaryResNet18
from train_utils import train_epoch, validate_epoch, EarlyStopping

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_id"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# W&B setup (only if enabled in config)
use_wandb = config["log_wandb"]
if use_wandb:
    import wandb
    wandb_run = wandb.init(entity = config["entity_name"], project = config["project_name"], name = config["run_name"], config = config)
else:
    wandb_run = None

# Data Paths
csv_file = config["csv_file"] 
data_dir = config["data_dir"]
ckpt_path = config["ckpt_path"]
save_path = os.path.join(ckpt_path, config["project_name"], config["run_name"])
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Hyperparameters
lr = config["lr"]
bs = config["batch_size"]
num_epochs = config["epochs"]
val_every = config["val_every"]
lr_factor = config["lr_factor"]
lr_patience = config["lr_patience"]
es_patience = config["es_patience"]

train_domains = config["train_domains"]
#test_domains = config["test_domains"]

transform = transforms.Compose([transforms.ToTensor()])
train_ds = DG_Dataset(csv_file = csv_file, domains = train_domains, data_dir = data_dir, shift_feature = "Origin", transform = transform)
#test_ds = DG_Dataset(csv_file = csv_file, domains = test_domains, data_dir = data_dir, shift_feature = "Origin", transform = transform)

# Print dataset information
print("Class to index mapping:", train_ds.class_to_idx)

# Split the train dataset into training and validation sets
train_size = int(0.8 * len(train_ds))
val_size = len(train_ds) - train_size
train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])
print(f"Training samples (In Domain): {len(train_ds)}, Validation samples (In Domain): {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4)
#test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4)

model = BinaryResNet18(dropout_p=0.5).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Schedulers
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience)
early_stopper = EarlyStopping(patience=es_patience, verbose=True, save_path=save_path)

# Training
print("\nStarting training...")
for epoch in range(num_epochs):
    # Training
    train_loss, train_acc = train_epoch(device, model, train_loader, criterion, optimizer, epoch, num_epochs)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

    if use_wandb:
        wandb.log({"train_loss": train_loss, "train_accuracy": train_acc}, step=epoch + 1)

    # Validation
    if epoch % val_every == 0:
        val_loss, val_acc = validate_epoch(device, model, val_loader, criterion)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        if use_wandb:
            wandb.log({"val_loss": val_loss, "val_accuracy": val_acc}, step=epoch + 1)

        if epoch > config["warmup_epochs"]:
            
            # Adjust learning rate
            old_lr = optimizer.param_groups[0]['lr']
            lr_scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                print(f"LR decreased: {old_lr:.6f} → {new_lr:.6f}")
            
            # Early stopping check
            early_stopper(epoch, val_loss, model)
            if early_stopper.early_stop:
                print("Early stopping triggered. Stopping training.")
                break

print("Training complete. Best model saved to:", early_stopper.save_path)

# Finish W&B run
if use_wandb:
    wandb_run.finish()