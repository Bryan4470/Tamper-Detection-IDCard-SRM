import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNetStream(nn.Module):
    """ResNet50 backbone wrapped to expose the same staged feature interface
    as the original Xception TransferModel:
        fea_part1_0 → fea_part1_1 → fea_part2 → fea_part3 → fea_part4 → fea_part5
    followed by a classifier() method returning (logits, features).

    Feature map shapes for 256×256 input:
        fea_part1_0  →  (B,   64, 128, 128)  conv1+bn+relu
        fea_part1_1  →  (B,   64,  64,  64)  maxpool
        fea_part2    →  (B, 1024,  16,  16)  layer1+layer2+layer3[:3]
        fea_part3    →  (B, 1024,  16,  16)  layer3[3:]
        fea_part4    →  (B, 2048,   8,   8)  layer4[:2]
        fea_part5    →  (B, 2048,   8,   8)  layer4[-1]
    """

    def __init__(self, num_out_classes=2, dropout=0.4):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        self.part1_0 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.part1_1 = base.maxpool

        layer3_blocks = list(base.layer3.children())
        self.part2 = nn.Sequential(base.layer1, base.layer2, *layer3_blocks[:3])
        self.part3 = nn.Sequential(*layer3_blocks[3:])

        layer4_blocks = list(base.layer4.children())
        self.part4 = nn.Sequential(*layer4_blocks[:2])
        self.part5 = layer4_blocks[-1]

        self.classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(2048, num_out_classes),
        )

    def fea_part1_0(self, x):
        return self.part1_0(x)

    def fea_part1_1(self, x):
        return self.part1_1(x)

    def fea_part2(self, x):
        return self.part2(x)

    def fea_part3(self, x):
        return self.part3(x)

    def fea_part4(self, x):
        return self.part4(x)

    def fea_part5(self, x):
        return self.part5(x)

    def classifier(self, fea):
        """Match TransferModel return signature: (logits, features)."""
        logits = self.classifier_head(fea)
        return logits, fea
