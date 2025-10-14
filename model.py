"""
Standalone 3D U-Net-like implementation with pluggable encoder-like builder.

This module provides a lightweight, standalone implementation of a 3D U-Net
architecture that mirrors the structure and layer counts of segmentation_models_pytorch's
UNet but without any external dependencies or pretrained weights.

Provided classes
- Conv3DBlock: two Conv3d + BN + ReLU block
- Encoder3D: simple configurable encoder that returns feature maps (shallow->deep)
- UnetDecoder3D: decoder with upsampling and skip concatenation
- SegmentationHead3D: final conv + activation
- ClassificationHead3D: optional aux classification head built on top of encoder's deepest feature
- Unet3D: the full model; constructor arguments try to match the familiar API but the
  implementation is standalone and simple.

Notes
- No pretrained weights are supported.
- Attention modules are not implemented; hook points are provided if you want to add them.

Example
-------
>>> model = Unet3D(encoder_depth=5, in_channels=1, classes=1)
>>> x = torch.randn(1,1,64,128,128)
>>> y = model(x)
>>> y.shape  # -> (1, classes, 64, 128, 128)

Author: ChatGPT (GPT-5 Thinking mini)
"""

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DBlock(nn.Module):
    """(Conv3d -> BN -> ReLU) x 2

    Keeps spatial dims if padding = kernel_size // 2
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, groups: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, padding=pad, bias=False, groups=groups),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size, padding=pad, bias=False, groups=groups),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder3D(nn.Module):
    """Simple encoder that produces feature maps at multiple stages.

    Produces a list of feature tensors from shallow->deep. The number of stages is
    equal to `depth`. Each stage halves spatial dims according to `strides` (which can be
    a sequence of tuples like ((2,2,2),(2,2,2),...)) or a single int/tuple used for pooling.
    """

    def __init__(self,
                 in_channels: int = 1,
                 features: Sequence[int] = (32, 64, 128, 256, 512),
                 strides: Optional[Sequence[Tuple[int, int, int]]] = None):
        super().__init__()
        self.features = list(features)
        self.depth = len(self.features)

        if strides is None:
            strides = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2)]
        # allow a single tuple or int
        if isinstance(strides, tuple) and not isinstance(strides[0], tuple):
            strides = [strides] * (self.depth - 1)

        self.strides = list(strides)

        self.stages = nn.ModuleList()
        prev_ch = in_channels
        for i, ch in enumerate(self.features):
            block = Conv3DBlock(prev_ch, ch)
            self.stages.append(block)
            prev_ch = ch

        # pooling ops between stages
        self.pools = nn.ModuleList()
        for s in self.strides:
            self.pools.append(nn.MaxPool3d(kernel_size=s, stride=s))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs: List[torch.Tensor] = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            outs.append(x)
            if i < len(self.pools):
                x = self.pools[i](x)
        return outs  # shallow -> deep


class UpBlock3D(nn.Module):
    """Upsampling block: ConvTranspose3d (or interpolate) + Conv3DBlock after concat with skip"""

    def __init__(self, in_ch: int, out_ch: int, up_mode: str = "transpose"):
        super().__init__()
        if up_mode == "transpose":
            self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        elif up_mode == "interp":
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(in_ch, out_ch, kernel_size=1),
            )
        else:
            raise ValueError("Unknown up_mode " + str(up_mode))

        # after concatenation channels = out_ch + skip_ch -> however we will construct decoder
        # with known channels, so here we just accept in_ch = channels from previous layer
        # and expect to receive a skip with matching channels to concat. We'll perform conv on 2*out_ch
        self.conv = Conv3DBlock(out_ch * 2, out_ch)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.up(x)
        if skip is None:
            # if no skip provided just pass through conv (duplicate to have correct channels)
            # pad/crop to match spatial dims if needed
            if x.shape[2:] != skip.shape[2:]:
                pass
        # If spatial dims differ, try to crop skip or pad
        if skip is not None and x.shape[2:] != skip.shape[2:]:
            # crop skip to x
            slices = []
            for sx, tx in zip(skip.shape[2:], x.shape[2:]):
                if sx == tx:
                    slices.append(slice(None))
                elif sx > tx:
                    start = (sx - tx) // 2
                    slices.append(slice(start, start + tx))
                else:
                    slices.append(slice(None))
            skip = skip[(...,) + tuple(slices)]
            if skip.shape[2:] != x.shape[2:]:
                # pad skip
                pad_sizes = []
                for sx, tx in zip(skip.shape[2:], x.shape[2:]):
                    total = tx - sx
                    before = total // 2
                    after = total - before
                    pad_sizes.extend([before, after])
                skip = F.pad(skip, pad_sizes[::-1])

        if skip is None:
            # replicate channels so conv input matches expected 2*out_ch
            x = torch.cat([x, x], dim=1)
        else:
            x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UnetDecoder3D(nn.Module):
    """Decoder module for UNet3D. Accepts encoder_channels (shallow->deep) and
    decoder_channels (same length) and performs upsampling from deepest to shallowest.
    """

    def __init__(self,
                 encoder_channels: Sequence[int],
                 decoder_channels: Sequence[int],
                 n_blocks: int,
                 up_mode: str = "transpose",
                 center: bool = False):
        super().__init__()
        if len(encoder_channels) != len(decoder_channels):
            raise ValueError("encoder_channels and decoder_channels must have same length")

        self.n_blocks = n_blocks
        # copy lists
        enc_ch = list(encoder_channels)
        dec_ch = list(decoder_channels)

        # build up blocks from deep -> shallow
        self.up_blocks = nn.ModuleList()
        in_ch = enc_ch[-1]
        for i in range(len(dec_ch)):
            out_ch = dec_ch[i]
            self.up_blocks.append(UpBlock3D(in_ch, out_ch, up_mode=up_mode))
            in_ch = out_ch

    def forward(self, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        # encoder_features: shallow -> deep
        x = encoder_features[-1]
        skips = encoder_features[:-1]
        # iterate up blocks and corresponding skips in reverse order
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            x = up_block(x, skip)
        return x


class SegmentationHead3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, activation: Optional[Union[str, callable]] = None):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=pad)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.activation is None:
            return x
        if isinstance(self.activation, str):
            if self.activation == "sigmoid":
                return torch.sigmoid(x)
            if self.activation == "softmax":
                # softmax over channel dim
                return torch.softmax(x, dim=1)
            if self.activation == "tanh":
                return torch.tanh(x)
            if self.activation == "identity":
                return x
            raise ValueError("Unknown activation " + str(self.activation))
        if callable(self.activation):
            return self.activation(x)
        return x


class ClassificationHead3D(nn.Module):
    def __init__(self, in_channels: int, classes: int = 1, pooling: str = "avg", dropout: float = 0.0, activation: Optional[str] = None):
        super().__init__()
        if pooling not in ("avg", "max"):
            raise ValueError("pooling must be 'avg' or 'max'")
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_channels, classes)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, C, D, H, W) -> pool to (N, C)
        if self.pooling == "avg":
            x = F.adaptive_avg_pool3d(x, 1).view(x.size(0), -1)
        else:
            x = F.adaptive_max_pool3d(x, 1).view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        if self.activation == "softmax":
            return torch.softmax(x, dim=1)
        return x


class Unet3D(nn.Module):
    """Standalone UNet-like 3D model.

    Parameters (kept similar to the familiar API):
    - encoder_depth: number of encoder stages (3..6)
    - encoder_features: tuple specifying channel sizes for encoder stages (shallow->deep)
    - decoder_channels: tuple specifying decoder channel sizes (same length as encoder_features)
    - in_channels, classes, activation, aux_params
    - strides: pooling strides between stages; accepts sequence or single tuple

    The encoder is a simple stack of Conv3DBlock + pooling. The decoder uses ConvTranspose3d
    upsampling and concatenation with encoder skip features.
    """

    def __init__(self,
                 encoder_depth: int = 5,
                 encoder_features: Optional[Sequence[int]] = None,
                 decoder_channels: Optional[Sequence[int]] = None,
                 encoder_weights: Optional[str] = None,  # ignored, for API parity
                 decoder_use_batchnorm: bool = True,  # ignored (BN always used)
                 in_channels: int = 1,
                 classes: int = 14,
                 activation: Optional[Union[str, callable]] = None,
                 aux_params: Optional[dict] = None,
                 strides: Optional[Sequence[Tuple[int, int, int]]] = None):
        super().__init__()

        if encoder_features is None:
            # default features similar to common UNet sizes but scaled for 3D
            encoder_features = [32, 64, 128, 256, 512][:encoder_depth]
        else:
            encoder_features = list(encoder_features)[:encoder_depth]

        if decoder_channels is None:
            # decoder channels typically reverse of encoder but can be custom
            decoder_channels = list(reversed(encoder_features))
            decoder_channels = decoder_channels[1:] + [decoder_channels[-1]]

        if len(encoder_features) != len(decoder_channels):
            # make equal by trimming/padding decoder_channels
            min_len = min(len(encoder_features), len(decoder_channels))
            encoder_features = encoder_features[:min_len]
            decoder_channels = decoder_channels[:min_len]

        self.encoder = Encoder3D(in_channels=in_channels, features=encoder_features, strides=strides)
        self.encoder_2 = Encoder3D(in_channels=14, features=[14,14,14,14,14], strides=strides)
        self.encoder_channels = encoder_features
        self.decoder_channels = decoder_channels
        self.decoder = UnetDecoder3D(
            encoder_channels=self.encoder_channels,
            decoder_channels=self.decoder_channels,
            n_blocks=len(self.encoder_channels),
            up_mode="transpose",
        )

        self.segmentation_head_1 = SegmentationHead3D(in_channels=self.decoder_channels[-1], out_channels=classes, activation=None)
        self.segmentation_head_2 = SegmentationHead3D(in_channels=self.decoder_channels[-1], out_channels=classes, activation=None)

        self.linear = nn.Linear(14, 14)

        if aux_params is not None:
            # aux_params keys: classes, pooling, dropout, activation
            aux_classes = aux_params.get("classes", 1)
            pooling = aux_params.get("pooling", "avg")
            dropout = aux_params.get("dropout", 0.0)
            act = aux_params.get("activation", None)
            self.classification_head = ClassificationHead3D(in_channels=self.encoder_channels[-1], classes=aux_classes, pooling=pooling, dropout=dropout, activation=act)
        else:
            self.classification_head = None

        # simple name
        self.name = f"unet3d_d{len(self.encoder_channels)}"

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        # encoder returns shallow->deep list
        feats = self.encoder(x)
        x = self.decoder(feats)
        ane = self.segmentation_head_1(x)
        ves = self.segmentation_head_2(x)
        c = self.encoder_2(ane)[-1]
        c = c.amax(dim=(2, 3, 4))
        c = self.linear(c)
        return {"seg_vessels": ves, "seg_aneurysms": ane, "class": c}


if __name__ == "__main__":
    # quick smoke test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet3D(encoder_depth=4, in_channels=1, classes=14).to(device)
    x = torch.randn(4, 1, 16, 224, 224, device=device)
    out = model(x)
    print("input:", x.shape)
    if isinstance(out, tuple):
        print("seg:", out[0].shape, "cls:", out[1].shape)
    else:
        print("out:", out.shape)

    total = sum(p.numel() for p in model.parameters())
    print(f"params: {total:,}")
