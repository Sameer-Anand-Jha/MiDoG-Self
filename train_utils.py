import os
import torch
import numpy as np
from tqdm import tqdm

def train_epoch(device, model, train_loader, criterion, optimizer, epoch, num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        logits = logits.view(-1)     
        loss = criterion(logits, labels.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
        optimizer.step()
        epoch_loss += loss.item()

        # Calculate accuracy
        preds = torch.sigmoid(logits) > 0.5
        epoch_accuracy += (preds == labels).float().mean().item()

    return epoch_loss / len(train_loader), epoch_accuracy / len(train_loader)


def validate_epoch(device, model, val_loader, criterion):
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="> Validation"):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            logits = logits.view(-1)
            loss = criterion(logits, labels.float())
            val_loss += loss.item()

            # Calculate accuracy
            preds = torch.sigmoid(logits) > 0.5
            val_accuracy += (preds == labels).float().mean().item()
    
    return val_loss / len(val_loader), val_accuracy / len(val_loader)


class EarlyStopping:

    def __init__(self, patience=7, verbose=False, delta=0.0, save_path=None):

        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, epoch, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            if self.verbose:
                print(f'Validation loss decreased ({self.best_loss:.6f} → {val_loss:.6f}).  Model Saved.')
            run_name = f"E_{epoch}_Loss_{val_loss:.4f}.pt"
            torch.save(model.state_dict(), os.path.join(self.save_path, run_name))
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'No improvement in validation loss for {self.counter}/{self.patience} epochs.')
            if self.counter >= self.patience:
                self.early_stop = True

