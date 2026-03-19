"""
Loss Functions for RGB+Cb Tamper Detection

Implements:
1. CbConsistencyLoss: Enforces Cb feature consistency
2. CombinedLoss: Multi-task learning with classification + consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class CbConsistencyLoss(nn.Module):
    """
    Cb Consistency Loss for background region patches.

    Key idea:
    - Genuine images: All region patches should have consistent Cb features
    - Tampered images: At least ONE region should show inconsistency
    """

    def __init__(self, margin: float = 0.5, patches_per_region: int = 16,
                 use_region_level: bool = True):
        super().__init__()
        self.margin = margin
        self.patches_per_region = patches_per_region
        self.use_region_level = use_region_level

    def forward(self, cb_features: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            cb_features: (B, N, D) - Cb patch features
            labels: (B,) - 0=genuine, 1=tampered
        """
        if self.use_region_level:
            return self._forward_region_level(cb_features, labels)
        return self._forward_global(cb_features, labels)

    def _forward_region_level(self, cb_features: torch.Tensor,
                               labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, D = cb_features.shape
        num_regions = N // self.patches_per_region

        cb_grouped = cb_features.view(B, num_regions, self.patches_per_region, D)
        region_scores = []

        for r in range(num_regions):
            region_patches = cb_grouped[:, r, :, :]
            dists = torch.cdist(region_patches, region_patches, p=2)

            triu_idx = torch.triu_indices(self.patches_per_region, self.patches_per_region,
                                          offset=1, device=dists.device)
            pairwise_dists = dists[:, triu_idx[0], triu_idx[1]]
            region_scores.append(pairwise_dists.mean(dim=1))

        region_scores = torch.stack(region_scores, dim=1)

        genuine_mask = (labels == 0)
        tampered_mask = (labels == 1)

        if genuine_mask.sum() > 0:
            genuine_loss = region_scores[genuine_mask].max(dim=1)[0].mean()
        else:
            genuine_loss = torch.tensor(0.0, device=cb_features.device)

        if tampered_mask.sum() > 0:
            max_scores = region_scores[tampered_mask].max(dim=1)[0]
            tampered_loss = F.relu(self.margin - max_scores).mean()
        else:
            tampered_loss = torch.tensor(0.0, device=cb_features.device)

        return {
            'total': genuine_loss + tampered_loss,
            'genuine': genuine_loss,
            'tampered': tampered_loss
        }

    def _forward_global(self, cb_features: torch.Tensor,
                        labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, D = cb_features.shape
        dists = torch.cdist(cb_features, cb_features, p=2)

        triu_idx = torch.triu_indices(N, N, offset=1, device=dists.device)
        pairwise_dists = dists[:, triu_idx[0], triu_idx[1]]

        genuine_mask = (labels == 0)
        tampered_mask = (labels == 1)

        if genuine_mask.sum() > 0:
            genuine_loss = pairwise_dists[genuine_mask].mean()
        else:
            genuine_loss = torch.tensor(0.0, device=cb_features.device)

        if tampered_mask.sum() > 0:
            max_dists = pairwise_dists[tampered_mask].max(dim=1)[0]
            tampered_loss = F.relu(self.margin - max_dists).mean()
        else:
            tampered_loss = torch.tensor(0.0, device=cb_features.device)

        return {
            'total': genuine_loss + tampered_loss,
            'genuine': genuine_loss,
            'tampered': tampered_loss
        }


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        labels_one_hot = F.one_hot(labels, num_classes=logits.shape[1])
        pt = (probs * labels_one_hot).sum(dim=1)

        focal_weight = (1 - pt) ** self.gamma
        loss = -self.alpha * focal_weight * torch.log(pt + 1e-8)
        return loss.mean()


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Shapes the feature space so that same-class samples cluster tightly and
    cross-class samples are pushed apart on the unit hypersphere.

    Reference: CFL-Net WACV 2023, SeeABLE ICCV 2023.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) — un-normalised feature vectors
            labels:   (B,)   — integer class labels
        Returns:
            Scalar contrastive loss, or 0.0 if no valid anchor exists.
        """
        B = features.size(0)
        if B < 2:
            return torch.tensor(0.0, device=features.device)

        # L2-normalise onto unit hypersphere
        features = F.normalize(features, dim=1)

        # Cosine similarity matrix scaled by temperature  (B, B)
        sim = torch.matmul(features, features.T) / self.temperature

        # Masks
        labels = labels.view(-1, 1)                           # (B, 1)
        pos_mask = (labels == labels.T).float()               # same class
        eye = torch.eye(B, device=features.device)
        pos_mask = pos_mask * (1 - eye)                       # exclude self
        neg_mask = 1 - eye                                    # all pairs except self

        # Numerically stable: subtract max per row before exp
        sim_max = sim.detach().max(dim=1, keepdim=True).values
        sim_exp = torch.exp(sim - sim_max)

        # For each anchor: sum positives / sum all negatives
        num_positives = pos_mask.sum(dim=1)                   # (B,)
        valid = num_positives > 0                             # anchors that have a positive

        if valid.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        numerator = (sim_exp * pos_mask).sum(dim=1)           # (B,)
        denominator = (sim_exp * neg_mask).sum(dim=1)         # (B,)

        # Avoid log(0)
        loss_per_anchor = -torch.log(numerator / (denominator + 1e-8) + 1e-8)
        loss_per_anchor = loss_per_anchor / num_positives.clamp(min=1)

        return loss_per_anchor[valid].mean()


class CombinedLoss(nn.Module):
    """Combined loss: Classification + Cb Consistency + (optional) Supervised Contrastive."""

    def __init__(self, cls_weight: float = 1.0, cb_weight: float = 0.3,
                 cb_margin: float = 0.5, use_focal_loss: bool = False,
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0,
                 patches_per_region: int = 16, use_region_level: bool = True,
                 contrastive_weight: float = 0.0, temperature: float = 0.07):
        super().__init__()
        self.cls_weight = cls_weight
        self.cb_weight = cb_weight
        self.contrastive_weight = contrastive_weight

        self.cls_loss = FocalLoss(focal_alpha, focal_gamma) if use_focal_loss else nn.CrossEntropyLoss()
        self.cb_loss = CbConsistencyLoss(cb_margin, patches_per_region, use_region_level)
        self.contrastive_loss = SupervisedContrastiveLoss(temperature)

    def forward(self, logits: torch.Tensor, cb_features: torch.Tensor,
                labels: torch.Tensor,
                fused_features: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        cls_loss = self.cls_loss(logits, labels)
        cb_loss_dict = self.cb_loss(cb_features, labels)

        total_loss = self.cls_weight * cls_loss + self.cb_weight * cb_loss_dict['total']

        if self.contrastive_weight > 0 and fused_features is not None:
            con_loss = self.contrastive_loss(fused_features, labels)
            total_loss = total_loss + self.contrastive_weight * con_loss
        else:
            con_loss = torch.tensor(0.0, device=logits.device)

        return {
            'total': total_loss,
            'classification': cls_loss,
            'cb_consistency': cb_loss_dict['total'],
            'cb_genuine': cb_loss_dict['genuine'],
            'cb_tampered': cb_loss_dict['tampered'],
            'contrastive': con_loss,
        }
