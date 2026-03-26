"""
CB Consistency Loss for Three-Stream Tamper Detection.

Key idea:
- Genuine images: All region patches should have consistent Cb features
- Tampered images: At least ONE region should show inconsistency

Adapted from face-tamper-inference/src/models/losses.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class CbConsistencyLoss(nn.Module):
    """
    Cb Consistency Loss for background region patches.

    For genuine images: Minimize pairwise distances between patch features
    For tampered images: Maximize at least one region's inconsistency (margin loss)
    """

    def __init__(
        self,
        margin: float = 0.5,
        patches_per_region: int = 16,
        use_region_level: bool = True,
    ):
        """
        Args:
            margin: Margin for tampered images (they should have distance > margin)
            patches_per_region: Number of patches per region (grid_size^2)
            use_region_level: If True, compute consistency per region then aggregate
        """
        super().__init__()
        self.margin = margin
        self.patches_per_region = patches_per_region
        self.use_region_level = use_region_level

    def forward(
        self,
        cb_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CB consistency loss.

        Args:
            cb_features: (B, N, D) - Cb patch features from CbEncoder
            labels: (B,) - 0=genuine, 1=tampered

        Returns:
            Dictionary with 'total', 'genuine', 'tampered' losses
        """
        if self.use_region_level:
            return self._forward_region_level(cb_features, labels)
        return self._forward_global(cb_features, labels)

    def _forward_region_level(
        self,
        cb_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss at region level then aggregate."""
        B, N, D = cb_features.shape
        num_regions = N // self.patches_per_region

        # Group patches by region
        cb_grouped = cb_features.view(B, num_regions, self.patches_per_region, D)
        region_scores = []

        for r in range(num_regions):
            region_patches = cb_grouped[:, r, :, :].contiguous()  # (B, patches_per_region, D)

            # Compute pairwise L2 distances within region
            dists = torch.cdist(region_patches, region_patches, p=2)  # (B, K, K)

            # Get upper triangle (unique pairs)
            triu_idx = torch.triu_indices(
                self.patches_per_region,
                self.patches_per_region,
                offset=1,
                device=dists.device
            )
            pairwise_dists = dists[:, triu_idx[0], triu_idx[1]]  # (B, num_pairs)

            # Mean distance per region
            region_scores.append(pairwise_dists.mean(dim=1))  # (B,)

        # Stack region scores: (B, num_regions)
        region_scores = torch.stack(region_scores, dim=1)

        # Split by class
        genuine_mask = (labels == 0)
        tampered_mask = (labels == 1)

        # Genuine loss: minimize max region inconsistency
        if genuine_mask.sum() > 0:
            genuine_loss = region_scores[genuine_mask].max(dim=1)[0].mean()
        else:
            genuine_loss = torch.tensor(0.0, device=cb_features.device)

        # Tampered loss: maximize at least one region's inconsistency
        if tampered_mask.sum() > 0:
            max_scores = region_scores[tampered_mask].max(dim=1)[0]
            tampered_loss = F.relu(self.margin - max_scores).mean()
        else:
            tampered_loss = torch.tensor(0.0, device=cb_features.device)

        return {
            'total': genuine_loss + tampered_loss,
            'genuine': genuine_loss,
            'tampered': tampered_loss,
        }

    def _forward_global(
        self,
        cb_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss globally across all patches."""
        B, N, D = cb_features.shape

        # Ensure contiguous for cdist backward pass
        cb_features = cb_features.contiguous()

        # Compute pairwise distances
        dists = torch.cdist(cb_features, cb_features, p=2)  # (B, N, N)

        # Get upper triangle
        triu_idx = torch.triu_indices(N, N, offset=1, device=dists.device)
        pairwise_dists = dists[:, triu_idx[0], triu_idx[1]]  # (B, num_pairs)

        # Split by class
        genuine_mask = (labels == 0)
        tampered_mask = (labels == 1)

        # Genuine loss: minimize mean pairwise distance
        if genuine_mask.sum() > 0:
            genuine_loss = pairwise_dists[genuine_mask].mean()
        else:
            genuine_loss = torch.tensor(0.0, device=cb_features.device)

        # Tampered loss: maximize max pairwise distance
        if tampered_mask.sum() > 0:
            max_dists = pairwise_dists[tampered_mask].max(dim=1)[0]
            tampered_loss = F.relu(self.margin - max_dists).mean()
        else:
            tampered_loss = torch.tensor(0.0, device=cb_features.device)

        return {
            'total': genuine_loss + tampered_loss,
            'genuine': genuine_loss,
            'tampered': tampered_loss,
        }


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        labels_one_hot = F.one_hot(labels, num_classes=logits.shape[1]).float()
        pt = (probs * labels_one_hot).sum(dim=1)

        focal_weight = (1 - pt) ** self.gamma
        loss = -self.alpha * focal_weight * torch.log(pt + 1e-8)
        return loss.mean()


class CombinedThreeStreamLoss(nn.Module):
    """
    Combined loss for Three-Stream Network:
    - Classification loss (CrossEntropy or Focal)
    - CB Consistency loss

    Total = cls_weight * cls_loss + cb_weight * cb_loss
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        cb_weight: float = 0.3,
        cb_margin: float = 0.5,
        use_focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        patches_per_region: int = 16,
        use_region_level: bool = True,
    ):
        """
        Args:
            cls_weight: Weight for classification loss
            cb_weight: Weight for CB consistency loss
            cb_margin: Margin for CB loss
            use_focal_loss: Use Focal Loss instead of CrossEntropy
            focal_alpha: Alpha for Focal Loss
            focal_gamma: Gamma for Focal Loss
            patches_per_region: Patches per region for CB loss
            use_region_level: Use region-level CB loss computation
        """
        super().__init__()
        self.cls_weight = cls_weight
        self.cb_weight = cb_weight

        if use_focal_loss:
            self.cls_loss = FocalLoss(focal_alpha, focal_gamma)
        else:
            self.cls_loss = nn.CrossEntropyLoss()

        self.cb_loss = CbConsistencyLoss(
            margin=cb_margin,
            patches_per_region=patches_per_region,
            use_region_level=use_region_level,
        )

    def forward(
        self,
        logits: torch.Tensor,
        cb_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            logits: (B, num_classes) classification logits
            cb_features: (B, N, D) CB patch features
            labels: (B,) ground truth labels

        Returns:
            Dictionary with all loss components
        """
        # Classification loss
        cls_loss = self.cls_loss(logits, labels)

        # CB consistency loss
        cb_loss_dict = self.cb_loss(cb_features, labels)

        # Combined
        total_loss = self.cls_weight * cls_loss + self.cb_weight * cb_loss_dict['total']

        return {
            'total': total_loss,
            'classification': cls_loss,
            'cb_consistency': cb_loss_dict['total'],
            'cb_genuine': cb_loss_dict['genuine'],
            'cb_tampered': cb_loss_dict['tampered'],
        }


if __name__ == '__main__':
    # Test losses
    B, N, D = 4, 80, 128  # 5 regions * 16 patches = 80
    num_classes = 2

    logits = torch.randn(B, num_classes)
    cb_features = torch.randn(B, N, D)
    labels = torch.tensor([0, 1, 0, 1])  # 2 genuine, 2 tampered

    # Test CbConsistencyLoss
    cb_loss_fn = CbConsistencyLoss(patches_per_region=16)
    cb_losses = cb_loss_fn(cb_features, labels)
    print(f"CB Loss - total: {cb_losses['total']:.4f}, "
          f"genuine: {cb_losses['genuine']:.4f}, "
          f"tampered: {cb_losses['tampered']:.4f}")

    # Test CombinedThreeStreamLoss
    combined_loss_fn = CombinedThreeStreamLoss()
    losses = combined_loss_fn(logits, cb_features, labels)
    print(f"Combined Loss - total: {losses['total']:.4f}, "
          f"cls: {losses['classification']:.4f}, "
          f"cb: {losses['cb_consistency']:.4f}")
