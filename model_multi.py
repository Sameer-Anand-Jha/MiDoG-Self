import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

class CustomFasterRCNN(nn.Module):
    """
    A Faster R-CNN wrapper whose backbone is an FPN built on top of any CNN (e.g. ResNet50).
    We support multiple training “methods.” The default is ERM, which just returns
    self.detector(images, targets). Other methods compute an additional loss on backbone features.
    """

    def __init__(self,
                 backbone_cnn: nn.Module,
                 return_layers: dict,
                 in_channels_list: list,
                 out_channels: int,
                 num_classes: int):
        """
        Args:
            backbone_cnn (nn.Module):
                The original CNN (e.g. torchvision.models.resnet50(pretrained=True)).
                BackboneWithFPN will slice out the layers in return_layers.

            return_layers (dict[str→str]):
                Mapping from layer names in backbone_cnn → FPN keys.
                For ResNet50:
                    {"layer2": "feat2",
                     "layer3": "feat3",
                     "layer4": "feat4"}

            in_channels_list (list[int]):
                Channel counts for each tapped layer, in order of return_layers.values().
                E.g. [512, 1024, 2048].

            out_channels (int):
                Number of channels in each FPN output (commonly 256).

            num_classes (int):
                Total number of object classes (including background).
        """
        super().__init__()

        # 1) Validate that backbone_cnn has the requested return_layers
        backbone_names = {name for name, _ in backbone_cnn.named_children()}
        missing = [k for k in return_layers if k not in backbone_names]
        if missing:
            raise ValueError(
                f"Backbone is missing these keys: {missing}. "
                f"Available top-level modules: {sorted(backbone_names)}"
            )

        # 2) Build FPN on top of backbone_cnn. BackboneWithFPN internally calls IntermediateLayerGetter.
        self.fpn_backbone = BackboneWithFPN(
            backbone=backbone_cnn,
            return_layers=return_layers,
            in_channels_list=in_channels_list,
            out_channels=out_channels
        )
        # Outputs an OrderedDict with keys list(return_layers.values()) + ["pool"]

        # 3) AnchorGenerator: number of pyramid levels = len(in_channels_list) + 1
        num_returned = len(in_channels_list)           # e.g. 3 tapped layers
        num_pyramid_levels = num_returned + 1          # FPN adds one extra (“pool”)
        sizes = tuple([(32 * (2 ** i),) for i in range(num_pyramid_levels)])       # [(32,), (64,), (128,), (256,)]
        aspect_ratios = tuple([(0.5, 1.0, 2.0) for _ in range(num_pyramid_levels)])
        anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)

        # 4) Build the Box ROI pooler: must match the FPN output keys
        featmap_names = list(return_layers.values()) + ["pool"]
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=featmap_names,
            output_size=7,
            sampling_ratio=2
        )

        # 5) Build Faster R-CNN head using our FPN backbone
        self.detector = FasterRCNN(
            backbone=self.fpn_backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=box_roi_pool
        )

        # 6) Replace the box predictor to match num_classes exactly
        in_feats = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)

        # Adaptor to convert features from (B, C, H, W) to (B, C, 1, 1)
        #self.adaptor_conv = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        #self.adaptor_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, images, targets=None):

        # Compute losses for detection module
        loss_dict = self.detector(images, targets)

        # 2) Extract FPN features for custom loss:
        #    We need to run the same image preprocessing that FasterRCNN expects:
        #    - Under the hood, detector(images, targets) has already called model.transform, but we can't easily recover that.
        #    So we reapply it: call detector.transform() to get normalized tensors and image_sizes
        images_norm, _ = self.detector.transform(images, targets)
        # images_norm.tensors: Tensor[B,3,H',W'] ready for backbone
        features = self.fpn_backbone(images_norm.tensors)
        # features is an OrderedDict: {"feat2": Tensor, "feat3": Tensor, "feat4": Tensor, "pool": Tensor} - ResNet50 FE

        #adaptor_features = [self.adaptor_pool(self.adaptor_conv(features[k])) for k in features.keys()]  # Apply the adaptor conv
        #adaptor_features = self.adaptor_pool(adaptor_features)  # Apply adaptive pooling to get (B, C, 1, 1)
        
        return features, loss_dict


def make_resnet50_fpn_backbone(num_classes: int):
    """
    Creates a CustomFasterRCNN whose backbone is ResNet-50 up to layer4, wrapped in FPN.
    """

    # 1) Load a fresh ResNet-50
    resnet = torchvision.models.resnet50(weights='IMAGENET1K_V1')

    # 2) Print top-level children to confirm available layers
    #print(">>> ResNet50 top-level children:", [n for n, _ in resnet.named_children()])
    # Expect: ['conv1','bn1','relu','maxpool','layer1','layer2','layer3','layer4','avgpool','fc']

    # 3) Choose which ResNet layers to tap for FPN:
    return_layers = {
        "layer2": "feat2",  # stride=8, channels=512
        "layer3": "feat3",  # stride=16, channels=1024
        "layer4": "feat4",  # stride=32, channels=2048
    }
    in_channels_list = [512, 1024, 2048]
    out_channels = 256  # each FPN output has 256 channels

    # 4) Build the CustomFasterRCNN
    model = CustomFasterRCNN(
        backbone_cnn=resnet,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        num_classes=num_classes
    )
    return model


# ---------------- Example usage ----------------
if __name__ == "__main__":
    num_classes = 2  # background + 1 object class
    model = make_resnet50_fpn_backbone(num_classes)

    # Dummy data
    images = [torch.randn(3, 800, 800), torch.randn(3, 800, 800)]
    targets = [
        {
            "boxes": torch.tensor([[100, 150, 400, 500]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64)
        },
        {
            "boxes": torch.tensor([[200, 200, 600, 650]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.int64)
        }
    ]

    model.train()
    features, loss_dict_erm = model(images, targets)
    
    print("Extracted features:", features.shape)        #  torch.Size([2, 256, 25, 25])
    print("Losses:", loss_dict_erm)


