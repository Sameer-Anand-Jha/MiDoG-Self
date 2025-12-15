import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# Pretrained ResNet18 model for binary classification
class BinaryResNet18(nn.Module):
    def __init__(self, dropout_p=0.3):
        super().__init__()
        # load ResNet-18 backbone
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        # replace final fc: output single logit
        num_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_feats, 1))
    
    def forward(self, x):
        # returns raw logit
        return self.backbone(x)