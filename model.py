import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import random

# OPZIONALE: import di qualche backbone pre-addestrato 3D o “inflated” da 2D
# Per esempio da segmentation_models_pytorch_3d
from torchvision.models import efficientnet_b0

# --- Block base: (Conv3d -> BN -> ReLU) * 2
class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# --- Downsample (MaxPool -> DoubleConv)
class Down3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_ch, out_ch)
        )
    def forward(self, x):
        return self.down(x)

# --- Upsample (Upsample -> Concat -> DoubleConv)
class Up3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = DoubleConv3D(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # padding se dimensioni diverse
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2,
                        diffY//2, diffY-diffY//2,
                        diffZ//2, diffZ-diffZ//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# --- Output conv
class OutConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_ch=1, num_classes_head1=14, num_classes_head2=14, base_ch=16):
        super().__init__()
        self.inc = DoubleConv3D(in_ch, base_ch)
        self.down1 = Down3D(base_ch, base_ch*2)
        self.down2 = Down3D(base_ch*2, base_ch*4)
        self.down3 = Down3D(base_ch*4, base_ch*8)
        self.bottom = DoubleConv3D(base_ch*8, base_ch*16)
        self.up1 = Up3D(base_ch*16 + base_ch*8, base_ch*8)
        self.up2 = Up3D(base_ch*8 + base_ch*4, base_ch*4)
        self.up3 = Up3D(base_ch*4 + base_ch*2, base_ch*2)
        self.up4 = Up3D(base_ch*2 + base_ch, base_ch)

        # Due teste separate per segmentazione
        self.head1 = OutConv3D(base_ch, num_classes_head1)
        self.head2 = OutConv3D(base_ch, num_classes_head2)

        # Global classification: max pooling senza Linear
        self.global_pool = nn.AdaptiveMaxPool3d(1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottom(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # due teste segmentazione
        out1 = self.head1(x)  # [B,14,D,H,W]
        out2 = self.head2(x)  # [B,14,D,H,W]

        # global classification: max pooling sui voxel
        pred = out1.clone()
        pred[:,0,:,:,:] = -pred[:,0,:,:,:]
        pooled = self.global_pool(out1).view(out1.size(0), -1)  # [B,14]

        # Copia i valori 1-13, metti somma in 0
        vec = pooled
    
        return {
                'seg_vessels': out1,
                'seg_aneurysms': out2,
                'class': vec
            }
    