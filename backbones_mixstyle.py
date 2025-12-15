# backbones_mixstyle.py
import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    densenet121, DenseNet121_Weights,
    densenet169, DenseNet169_Weights
)
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Origin_DG.model_spatial import CBAM  # as in your code

class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def forward(self, x):
        if not self.training or not self._activated or torch.rand(1).item() > self.p:
            return x
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + 1e-6).sqrt()
        x_norm = (x - mu) / sig
        lmda = self.beta.sample((B, 1, 1, 1)).to(x.device)
        perm = torch.randperm(B, device=x.device)  # random mix
        mu2, sig2 = mu[perm], sig[perm]
        mu_m = mu * lmda + mu2 * (1 - lmda)
        sg_m = sig * lmda + sig2 * (1 - lmda)
        return x_norm * sg_m + mu_m


def _insert_mixstyle_in_sequential(seq: nn.Sequential, p=0.5, alpha=0.1):
    """Split a Sequential into three chunks and insert MixStyle after chunk1 and chunk2.
    Works robustly across torchvision models with `.features` (length varies by model)."""
    layers = list(seq.children())
    L = len(layers)
    # pick safe split points
    s1 = max(1, L // 3)
    s2 = max(s1 + 1, (2 * L) // 3)
    return nn.Sequential(
        *layers[:s1],
        MixStyle(p=p, alpha=alpha),
        *layers[s1:s2],
        MixStyle(p=p, alpha=alpha),
        *layers[s2:],
    )


def _infer_out_channels(module: nn.Module, img_size=224, device='cpu'):
    with torch.no_grad():
        x = torch.zeros(1, 3, img_size, img_size, device=device)
        y = module(x)
        return y.shape[1]


class BinaryTVBackboneMixStyle(nn.Module):
    """
    Generic wrapper for torchvision backbones that expose `.features` (Sequential).
    Examples: MobileNetV3-Large, EfficientNet-B0, DenseNet-121, MNASNet, ShuffleNetV2.
    """
    def __init__(self,
                 backbone_name: str = 'mobilenet_v3_large',
                 pretrained: bool = True,
                 dropout_p: float = 0.3,
                 mixstyle_p: float = 0.5,
                 mixstyle_alpha: float = 0.1,
                 img_size: int = 224,
                 add_relu_after_features: bool = False):
        super().__init__()

        # 1) Build the backbone
        if backbone_name == 'mobilenet_v3_large':
            bb = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
            features = bb.features  # nn.Sequential
        elif backbone_name == 'efficientnet_b0':
            bb = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            features = bb.features  # nn.Sequential
        elif backbone_name == 'densenet121':
            bb = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
            features = bb.features  # nn.Sequential (ends at norm5)
            # DenseNet conventionally applies a ReLU after features; expose a flag to keep parity
            add_relu_after_features = False
        elif backbone_name == 'densenet169':
            bb = densenet169(weights=DenseNet169_Weights.DEFAULT if pretrained else None)
            features = bb.features  # nn.Sequential (ends at norm5)
            # DenseNet conventionally applies a ReLU after features; expose a flag to keep parity
            add_relu_after_features = False
        else:
            raise ValueError(f'Unsupported backbone: {backbone_name}. '
                             f'Pick one of: mobilenet_v3_large, efficientnet_b0, densenet121.')

        # 2) Insert MixStyle at two interior points
        self.features = _insert_mixstyle_in_sequential(
            features, p=mixstyle_p, alpha=mixstyle_alpha
        )

        # Optional post-features ReLU for backbones like DenseNet
        self.post_act = nn.ReLU(inplace=True) if add_relu_after_features else nn.Identity()

        # 3) Work out output channels robustly by probing once
        device = next(self.features.parameters()).device if any(p.requires_grad for p in self.features.parameters()) else 'cpu'
        self.num_feats = _infer_out_channels(nn.Sequential(self.features, self.post_act), img_size=img_size, device=device)

        # 4) Keep your CBAM + head
        self.cbam = CBAM(in_planes=self.num_feats)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Flatten(),
            nn.Linear(self.num_feats, 1)
        )

    def forward(self, x):
        feat_map = self.features(x)      # [B, C, H, W]
        feat_map = self.post_act(feat_map)
        feat_map = self.cbam(feat_map)
        pooled = self.pool(feat_map)
        logits = self.classifier(pooled)
        return feat_map, logits


model = BinaryTVBackboneMixStyle(backbone_name='densenet121')
print(model)