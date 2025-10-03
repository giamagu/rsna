import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from dataset import RSNA3DDataset
from model import MultiTask3DNet

def main():

    # ==============================
    # CONFIG
    # ==============================
    DATASET_DIR = "resampled_dataset"
    SPLIT_FILE = os.path.join(DATASET_DIR, "split.json")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 2
    EPOCHS = 50
    LR = 1e-4

    # ==============================
    # CREA O CARICA SPLIT
    # ==============================
    if not os.path.exists(SPLIT_FILE):
        # tutte le cartelle disponibili (serie)
        series_dirs = sorted(os.listdir(os.path.join(DATASET_DIR, "series")))
        random.shuffle(series_dirs)

        split_idx = int(0.85 * len(series_dirs))
        train_series = series_dirs[:split_idx]
        val_series = series_dirs[split_idx:]

        split = {"train": train_series, "val": val_series}

        with open(SPLIT_FILE, "w") as f:
            json.dump(split, f, indent=2)
        print(f"Creato split.json con {len(train_series)} train e {len(val_series)} val")
    else:
        with open(SPLIT_FILE, "r") as f:
            split = json.load(f)
        train_series = split["train"]
        val_series = split["val"]
        print(f"Caricato split.json con {len(train_series)} train e {len(val_series)} val")

    # ==============================
    # DATALOADERS
    # ==============================
    train_dataset = RSNA3DDataset(DATASET_DIR, series_ids=train_series)
    val_dataset   = RSNA3DDataset(DATASET_DIR, series_ids=val_series)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # ==============================
    # MODEL
    # ==============================
    model = MultiTask3DNet(
        in_channels=1,
        num_vessel_classes=14,
        num_aneurysm_classes=14,
        num_classification_classes=14,
        pretrained=True,
        freeze_backbone=False
    ).to(DEVICE)

    # ==============================
    # LOSSES & OPTIMIZER
    # ==============================
    criterion_seg = nn.CrossEntropyLoss()
    criterion_vec = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ==============================
    # TRAINING LOOP
    # ==============================
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            inputs = batch["input"].to(DEVICE, dtype=torch.float32)
            seg_vessels_gt = batch["cowseg"].to(DEVICE, dtype=torch.long)
            seg_aneurysm_gt = batch["aneurysm_seg"].to(DEVICE, dtype=torch.long)
            vec_gt = batch["aneurysm_vector"].to(DEVICE, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss_vessels = criterion_seg(outputs["seg_vessels"], seg_vessels_gt)
            loss_aneurysm = criterion_seg(outputs["seg_aneurysms"], seg_aneurysm_gt)
            loss_vec = criterion_vec(outputs["class"], vec_gt)

            loss = loss_vessels + loss_aneurysm + loss_vec
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # VALIDATION
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(DEVICE, dtype=torch.float32)
                seg_vessels_gt = batch["cowseg"].to(DEVICE, dtype=torch.long)
                seg_aneurysm_gt = batch["aneurysm_seg"].to(DEVICE, dtype=torch.long)
                vec_gt = batch["aneurysm_vector"].to(DEVICE, dtype=torch.float32)

                outputs = model(inputs)

                loss_vessels = criterion_seg(outputs["seg_vessels"], seg_vessels_gt)
                loss_aneurysm = criterion_seg(outputs["seg_aneurysms"], seg_aneurysm_gt)
                loss_vec = criterion_vec(outputs["class"], vec_gt)

                loss = loss_vessels + loss_aneurysm + loss_vec
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Salvataggio checkpoint ogni 10 epoche
        if (epoch + 1) % 10 == 0:
            ckpt_path = f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint salvato: {ckpt_path}")


if __name__ == "__main__":
    
    main()