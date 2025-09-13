#   Copyright 2025 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import torch.nn.functional as F

class MonaiUNet3DSeg(nn.Module):
    """3D U-Net model for segmentation using MONAI.
    Parameters
    ----------
        num_classes (int): Number of output classes for segmentation.
        0: background
        1: ICA-L
        2: ICA-R
        3: MCA-L
        4: MCA-R
        5: ACA-L
        6: ACA-R
    """
    def __init__(self, num_classes=7):
        super().__init__()
        self.unet = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH
        )

    def forward(self, x):
        return self.unet(x)

def load_trained_model_seg(model_path, device):
    model = MonaiUNet3DSeg(num_classes=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

###while training will not be conducted, this is inference-ready, it doesn't hurt to have it
def combined_loss(preds, mask, dice_weight=1.0, ce_weight=1.0):
    """
    Combined loss function that computes the weighted sum of Dice loss and Cross-Entropy loss.
    Args:
        preds (torch.Tensor): Predicted logits from the model.
        mask (torch.Tensor): Ground truth segmentation mask.
        dice_weight (float): Weight for the Dice loss.
        ce_weight (float): Weight for the Cross-Entropy loss.
    Returns:
        torch.Tensor: Combined loss value.
    """
    ce_loss = F.cross_entropy(preds, mask)
    probs = torch.softmax(preds, dim=1)
    dice_loss = 0.0
    num_classes = preds.shape[1]
    for class_idx in range(1, num_classes):
        p = probs[:, class_idx].flatten()
        t = (mask == class_idx).float().flatten()
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        dice = (2. * intersection + 1e-5) / (union + 1e-5)
        dice_loss += (1 - dice) / (num_classes - 1)
    return dice_weight * dice_loss + ce_weight * ce_loss

