import torch
import torch.nn as nn
import torchvision.models as models


class MultiTaskEfficientNet(nn.Module):
    def __init__(self, num_labels=14, backbone_name="efficientnet_b0", pretrained=True):
        super().__init__()
        
        # Backbone EfficientNet
        backbone = getattr(models, backbone_name)(weights="IMAGENET1K_V1" if pretrained else None)
        
        # Rimuovo il classifier finale
        self.feature_extractor = backbone.features
        self.avgpool = backbone.avgpool
        in_features = backbone.classifier[1].in_features
        
        # Classification head
        self.classifier = nn.Linear(in_features, num_labels)

        # Segmentation head (tipo decoder semplice, attaccato a feature intermedie)
        # Prendo feature a 1/16 della risoluzione
        self.seg_in_channels = backbone.features[6][-1].out_channels
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(self.seg_in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)  # mask binaria
        )
        
    def forward(self, x):
        feats = []
        out = x
        for i, block in enumerate(self.feature_extractor):
            out = block(out)
            if i == 6:  # salviamo feature a 1/16
                seg_features = out
        
        # Classification
        pooled = self.avgpool(out)
        pooled = torch.flatten(pooled, 1)
        class_logits = self.classifier(pooled)
        
        # Segmentation
        seg_logits = self.segmentation_head(seg_features)
        seg_logits = nn.functional.interpolate(seg_logits, size=x.shape[2:], mode="bilinear", align_corners=False)
        
        return class_logits, seg_logits