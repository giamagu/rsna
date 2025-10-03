import torch
import torch.nn as nn
import torch.nn.functional as F

# OPZIONALE: import di qualche backbone pre-addestrato 3D o “inflated” da 2D
# Per esempio da segmentation_models_pytorch_3d
from segmentation_models_pytorch_3d import Unet as Unet3D  # encoder+decoder base

class MultiTask3DNet(nn.Module):
    def __init__(self, 
                 in_channels=1,
                 num_vessel_classes=14,    # 0–13
                 num_aneurysm_classes=14,   # 0–13
                 num_classification_classes=14,
                 pretrained=True,
                 freeze_backbone=False):
        super().__init__()
        
        # Backbone base UNet 3D
        # Questo Unet restituisce output con canale = classes, tipico smp style
        # Però lo useremo solo per l'encoder + decoder fino al bottleneck delle caratteristiche (o anche decoder se vuoi)
        self.base_unet = Unet3D(
            in_channels=in_channels,
            classes=1,    # dummy, solo per ottenere feature; eventualmente cambiare
            encoder_name='efficientnet-b0',  # se supportato
            encoder_weights='imagenet',      # o altro pretraining
            activation=None
        )
        # NOTA: se l’Unet3D qui restituisce un solo canale, potresti usarlo solo per encoder e definire tu i decoder.

        # Se vuoi usare il decoder della base per risparmiare lavoro:
        # altrimenti costruisci i tuoi decoder

        # Heads di segmentazione
        # Vasi sanguigni
        self.seg_vessels_head = nn.Conv3d(
            in_channels=self.base_unet.decoder.blocks[-1].conv2[0].out_channels,
            out_channels=num_vessel_classes,
            kernel_size=1
        )

        # Aneurismi
        self.seg_aneurysms_head = nn.Conv3d(
            in_channels=self.base_unet.decoder.blocks[-1].conv2[0].out_channels,
            out_channels=num_aneurysm_classes,
            kernel_size=1
        )

        # Head classificazione globale
        # Prendiamo feature via global pooling dal “bottleneck” o da ultima feature dell’encoder/decoder
        self.global_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.classifier = nn.Linear(self.base_unet.decoder.blocks[-1].conv2[0].out_channels, num_classification_classes)

        # Flag se congelare backbone
        if freeze_backbone:
            for param in self.base_unet.parameters():
                param.requires_grad = False
            # Però lascia attivi i parametri dei nuovi head
            for p in self.seg_vessels_head.parameters():
                p.requires_grad = True
            for p in self.seg_aneurysms_head.parameters():
                p.requires_grad = True
            for p in self.classifier.parameters():
                p.requires_grad = True

    def forward(self, x):
        """
        x: tensor di shape (B, in_channels, D, H, W)
        """

        # Ottieni features dal backbone
        # per Unet3D lo standard è che ritorna l’output segmentazione, ma vogliamo le feature "intermediate"
        # Qui assumiamo di poter ottenere le feature dell’ultimo decoder layer
        features = self.base_unet.encoder(x)  
        # Questa parte dipende fortemente da come è implementato Unet3D: potresti dover modificare per ottenere il "bottleneck" o "feature map"
        
        # Se base_unet ha già un decoder che produce feature di dimensione spaziale D, H, W
        # altrimenti costruisci uno “decoder custom”

        # Useremo una parte del decoder per "espandere" features fino a quella dimensione spaziale finale
        decoded = self.base_unet.decoder(*features)  # dipende da implementazione

        # Ultimo layer "decoded" è un feature map con shape (B, C_f, D, H, W)
        # dove C_f = self.base_unet.decoder_channels[-1] (numero di canali finali del decoder)

        # Heads
        seg_vessels_logits = self.seg_vessels_head(decoded)        # (B, 14, D, H, W)
        seg_aneurysms_logits = self.seg_aneurysms_head(decoded)    # (B, 14, D, H, W)

        # Classificazione globale
        pooled = self.global_pool(decoded)  # (B, C_f, 1,1,1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, C_f)
        class_logits = self.classifier(pooled)    # (B, 14)

        return {
            'seg_vessels': seg_vessels_logits,
            'seg_aneurysms': seg_aneurysms_logits,
            'class': class_logits
        }
