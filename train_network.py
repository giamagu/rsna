import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataset import AneurysmDatasetBuilder
from model import MultiTaskEfficientNet


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        """
        Args:
            patience: numero massimo di epoche senza miglioramento
            delta: miglioramento minimo considerato
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=10,
    lr=1e-3,
    alpha=1.0,
    finetune=False,
    patience=5
):
    """
    Train loop per modello multitask classificazione + segmentazione con early stopping.
    """

    model = model.to(device)

    # Congela parametri se non facciamo finetune
    if not finetune:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
        for p in model.segmentation_head.parameters():
            p.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_seg = nn.BCEWithLogitsLoss()

    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        # ---- TRAIN ----
        model.train()
        train_loss = 0.0

        for imgs, labels, masks in tqdm(train_loader, desc=f"[Train Epoch {epoch+1}]"):
            imgs, labels = imgs.to(device), labels.to(device).float()

            optimizer.zero_grad()
            class_logits, seg_logits = model(imgs)

            loss_cls = criterion_cls(class_logits, labels)

            loss_seg = 0.0
            if masks is not None:
                masks = masks.to(device).float()
                loss_seg = criterion_seg(seg_logits, masks)

            loss = loss_cls + alpha * loss_seg
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels, masks in tqdm(val_loader, desc=f"[Val Epoch {epoch+1}]"):
                imgs, labels = imgs.to(device), labels.to(device).float()

                class_logits, seg_logits = model(imgs)
                loss_cls = criterion_cls(class_logits, labels)

                loss_seg = 0.0
                if masks is not None:
                    masks = masks.to(device).float()
                    loss_seg = criterion_seg(seg_logits, masks)

                loss = loss_cls + alpha * loss_seg
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}"
        )

        # ---- EARLY STOPPING ----
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break


# ====================
# Esempio d'uso
# ====================
if __name__ == "__main__":
    # Supponendo che tu abbia già il builder con split train/val
    # dataset = builder.build_dataset()
    # train_set, val_set = builder.split_train_val()

    builder = AneurysmDatasetBuilder(
        img_dir="preprocessed/images",
        mask_dir="preprocessed/masks",
        csv_path="rsna-intracranial-aneurysm-detection/train.csv"
    )

    dataset = builder.build_dataset()
    train_set, val_set = builder.split_train_val()

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    model = MultiTaskEfficientNet()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fase 1: addestramento solo dei layer nuovi
    train_model(model, train_loader, val_loader, device, num_epochs=5, lr=1e-3, finetune=False, patience=3)

    # Fase 2: fine-tuning completo
    train_model(model, train_loader, val_loader, device, num_epochs=10, lr=1e-4, finetune=True, patience=3)

