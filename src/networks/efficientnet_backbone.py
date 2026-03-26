import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights


class EfficientNetStream(nn.Module):
    """EfficientNet-B4 backbone wrapped to expose the same staged feature interface
    as ResNetStream:
        fea_part1_0 → fea_part1_1 → fea_part2 → fea_part3 → fea_part4 → fea_part5

    Feature map shapes for 256×256 input:
        fea_part1_0  →  (B,   48, 128, 128)  features[0] stem conv
        fea_part1_1  →  (B,   24, 128, 128)  features[1] MBConv stride=1
        fea_part2    →  (B,  112,  16,  16)  features[2:5]
        fea_part3    →  (B,  160,  16,  16)  features[5]
        fea_part4    →  (B,  272,   8,   8)  features[6]
        fea_part5    →  (B, 1792,   8,   8)  features[7:9]

    SE modules at every block give per-channel attention — useful for weighting
    which SRM noise residual channel carries the most tamper signal.
    """

    def __init__(self, num_out_classes=2, dropout=0.4):
        super().__init__()
        base = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)

        self.part1_0 = base.features[0]
        self.part1_1 = base.features[1]
        self.part2   = nn.Sequential(*list(base.features[2:5]))
        self.part3   = base.features[5]
        self.part4   = base.features[6]
        self.part5   = nn.Sequential(*list(base.features[7:9]))

        self.classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(1792, num_out_classes),
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
