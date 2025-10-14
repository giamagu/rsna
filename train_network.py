
import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np

from dataset import RSNA3DDataset
from model import Unet3D
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import SimpleITK as sitk
from save_dataset_png import convert_to_float32, resample_image, sitk_to_uint8

model = Unet3D().to(DEVICE)
ckpt_path = "checkpoint_epoch_15.pth"
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
model.eval()

def predict(path):
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_files)
    image = reader.Execute()

    # ----- Gestisci 4D -----
    if image.GetDimension() == 4:
        size = list(image.GetSize())
        size[3] = 0  # tieni solo un frame
        index = [0, 0, 0, 0]
        image = sitk.Extract(image, size, index)
        image = convert_to_float32(image)

    # ----- Resample immagine -----
    original_size = image.GetSize()  # (X,Y,Z)
    image_resampled = resample_image(image, target_size=(224,224))
    new_size = image_resampled.GetSize()  # (224,224,Z)

    # ----- Salvataggio immagine come PNG -----
    arr_img = sitk_to_uint8(image_resampled, is_mask=False)
    arr_img = arr_img.astype(np.float32) / 255

    out = model(arr_img)["class"]


def evaluate_score(labels, preds):
    # Converti labels e preds in array NumPy
    labels_array = np.array(labels)  # Shape: (N, 14)
    preds_array = np.array(preds)    # Shape: (N, 14)

    aucs = []

    # Calcola l'AUC per la prima componente (AUC_0)

    # Calcola l'AUC per le altre componenti (AUC_1, ..., AUC_13)
    auc_rest = 0
    meaningful_classes = 0
    for i in range(14):
        try:
            aucs.append(roc_auc_score(labels_array[:, i], preds_array[:, i]))
        except:
            aucs.append(-1)
            
    # Calcola l'AUC totale
    return aucs

def calculate_auc_from_number(positive_counts, negative_counts):
    # Numero totale di campioni positivi e negativi
    total_positives = np.sum(positive_counts)
    total_negatives = np.sum(negative_counts)
    
    # Inizializza l'AUC
    auc = 0.0
    
    # Itera sugli score (da 0 a 10000)
    cum_negatives = 0  # Cumulativo dei negativi
    for i in range(len(positive_counts)):
        # Contributo dell'intervallo all'AUC
        auc += positive_counts[i] * (cum_negatives + 0.5 * negative_counts[i])
        cum_negatives += negative_counts[i]
    
    # Normalizza l'AUC dividendo per il totale di positivi e negativi
    auc /= (total_positives * total_negatives)
    
    return auc

def evaluate_score_seg(labels, preds):
    # Converti labels e preds in array NumPy
    labels_array = np.array(labels)  # Shape: (N, 14)
    preds_array = np.array(preds)    # Shape: (N, 14)

    # Calcola l'AUC per la prima componente (AUC_0)
    # Calcola l'AUC per le altre componenti (AUC_1, ..., AUC_13)
    auc_rest = 0
    meaningful_classes = 0

    aucs = []
    for i in range(14):
        if True:
        #try:
            aucs.append(calculate_auc_from_number(labels_array[i], preds_array[i]))
        #except:
        #    aucs.append(-1)

    return aucs

def main():

    # ==============================
    # CONFIG
    # ==============================
    DATASET_DIR = "resampled_dataset"
    SPLIT_FILE = os.path.join(DATASET_DIR, "split.json")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 1
    EPOCHS = 40
    LR = 4e-3

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

    image_size = 224
    train_transforms = A.Compose(
        [
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, border_mode=0, p=0.3),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            #A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit = 0.1, p=0.3),
        ],
        additional_targets={
            "vessels": "mask",  # Le segmentazioni dei vasi sono trattate come maschere
            "aneurysms": "mask"  # Le segmentazioni degli aneurismi sono trattate come maschere
        }
    )

    train_dataset = RSNA3DDataset(DATASET_DIR, series_ids=train_series, only_vessels = True, transform=train_transforms)
    val_dataset   = RSNA3DDataset(DATASET_DIR, series_ids=val_series, only_vessels = True, minimum_slices = 16, maximum_slices=1000000)


    #tot = np.array(train_dataset[0]["aneurysm_vector"])
    #for x in train_dataset:
    #    tot += np.array(x["aneurysm_vector"])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last =True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    # ==============================
    # MODEL
    # ==============================
    model = Unet3D().to(DEVICE)

    '''ckpt_path = "checkpoint_epoch_30.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    labels = []
    preds = []

    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        leftover = sample["image"].shape[1]%16
        left_over_left = int(leftover/2)
        left_over_right = leftover - left_over_left
        if leftover > 0:
            out = model(sample["image"][:, left_over_left:-left_over_right, :, :].unsqueeze(0).to(DEVICE))
        else:
            out = model(sample["image"].unsqueeze(0).to(DEVICE))
        labels.append(sample["aneurysm_vector"].tolist())
        preds.append(torch.sigmoid(out["class"].to(DEVICE).cpu()).detach().numpy().tolist()[0])
        a = 1

    # Calculate AUC
    auc_score = evaluate_score(labels, preds)
    print(f"AUC Score: {auc_score:.4f}")'''

    # ==============================
    # LOSSES & OPTIMIZER
    # ==============================
    weights = np.array([1.0] + [100000.0]*13, dtype = np.float32)
    weights = weights / weights.sum() * 14
    criterion_seg_an = nn.CrossEntropyLoss(weight = torch.Tensor(weights).to(DEVICE))

    weights = np.array([1.0] + [1000.0]*13,  dtype = np.float32)
    weights = weights / weights.sum() * 14
    criterion_seg_ves = nn.CrossEntropyLoss(weight = torch.Tensor(weights).to(DEVICE))

    criterion_vec = nn.BCEWithLogitsLoss()

    # ==============================
    # TRAINING LOOP
    # ==============================
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):

        if epoch == 20:

            train_dataset = RSNA3DDataset(DATASET_DIR, series_ids=train_series, only_vessels = False, transform=train_transforms)
            val_dataset   = RSNA3DDataset(DATASET_DIR, series_ids=val_series, only_vessels = False, minimum_slices = 16, maximum_slices=1000000)

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last =True)
            val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

            LR = 1e-3
            optimizer = optim.Adam(model.parameters(), lr=LR)
        
        total_loss = 0.0

        i = 0

        for batch in train_loader:

            inputs = batch["image"].to(DEVICE, dtype=torch.float32)
            seg_vessels_gt = batch["vessels"].to(DEVICE, dtype=torch.long)
            seg_aneurysm_gt = batch["aneurysms"].to(DEVICE, dtype=torch.long)
            vec_gt = batch["aneurysm_vector"].to(DEVICE, dtype=torch.float32)
            vec_gt[:, 0] = 1 - vec_gt[:, 0]


            optimizer.zero_grad()
            outputs = model(inputs)

            loss_vessels = torch.tensor(0.0, device=DEVICE)
            for k in range(seg_vessels_gt.shape[0]):
                if seg_vessels_gt[k].min().item() > -0.5:
                    outputs["seg_vessels"][k:k+1], seg_vessels_gt[k:k+1]
                    loss_vessels += criterion_seg_ves(outputs["seg_vessels"][k:k+1], seg_vessels_gt[k:k+1])
            loss_aneurysm = criterion_seg_an(outputs["seg_aneurysms"], seg_aneurysm_gt)
            loss_vec = criterion_vec(outputs["class"], vec_gt)

            if epoch > 30:
                loss = loss_vessels + loss_aneurysm + loss_vec
            else:
                loss = loss_vessels + loss_aneurysm

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print(f"Epoch {epoch+1}/{EPOCHS} | Batch {i+1}/{len(train_loader)} | Loss vec: {loss_vec.item():.4f} | Loss ane: {loss_aneurysm.item():.4f} | Loss vessels: {loss_vessels.item():.4f}")

            i += 1

        avg_train_loss = total_loss / len(train_loader)



        # VALIDATION
        #model.eval()
        #val_loss = 0.0
        #with torch.no_grad():
        #    for batch in val_loader:
        #        inputs = batch["image"].to(DEVICE, dtype=torch.float32)
        #        seg_vessels_gt = batch["vessels"].to(DEVICE, dtype=torch.long)
        #        seg_aneurysm_gt = batch["aneurysms"].to(DEVICE, dtype=torch.long)
        #        vec_gt = batch["aneurysm_vector"].to(DEVICE, dtype=torch.float32)

        #        start = 0
        #        while start < inputs.shape[2]:

        #            outputs = model(inputs[:,:,start:start+256,:,:])

                    #loss_vessels = criterion_seg(outputs["seg_vessels"], seg_vessels_gt)
        #            loss_aneurysm = criterion_seg(outputs["seg_aneurysms"], seg_aneurysm_gt[:,start:start+256,:,:])
        #            loss_vec = criterion_vec(outputs["class"], vec_gt) / 3

        #            loss = loss_vessels + loss_aneurysm + loss_vec
        #            val_loss += loss.item()

        #            start += 256

        #avg_val_loss = val_loss / len(val_loader)

        #print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Salvataggio checkpoint ogni 10 epoche
        if (epoch + 1) % 4 == 0:
            ckpt_path = f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint salvato: {ckpt_path}")

    ckpt_path = "checkpoint_epoch_28.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    labels_ane = np.zeros([14, 0, 224, 224])
    preds_ane = np.zeros([14, 0, 224, 224])
    labels_seg = np.zeros([14, 0, 224, 224])
    preds_seg = np.zeros([14, 0, 224, 224])
    labels = []
    preds = []

    positive_preds_counter_ane = [np.zeros(10001) for i in range(14)]
    negative_preds_counter_ane = [np.zeros(10001) for i in range(14)]
    positive_preds_counter_seg = [np.zeros(10001) for i in range(14)]
    negative_preds_counter_seg = [np.zeros(10001) for i in range(14)]

    val_dataset = RSNA3DDataset(DATASET_DIR, series_ids=val_series, only_vessels = False, minimum_slices = 16, maximum_slices=1000000)

    for i in range(50):#len(val_dataset)):
        print(i)
        sample = val_dataset[i]
        inp = sample["image"].unsqueeze(0).to(DEVICE)
        
        start = 0
        pred_out = [0]*14
        must_stop = False
        while start < inp.shape[2] and not must_stop:
            if start + 64 > inp.shape[2]:
                start = inp.shape[2] - 64
                must_stop = True
            out = model(inp[:,:,start:start+64,:,:])
            l = torch.sigmoid(out["class"].to(DEVICE).cpu()).detach().numpy().tolist()[0]
            pred_out = [max(pred_out[j], l[j]) for j in range(14)]

            preds_ane = torch.softmax(out["seg_aneurysms"].to(DEVICE).cpu(), axis = 1).detach().numpy()[0]
            label = np.array([sample["aneurysms"][start:start+64,:,:].detach().numpy() == i for i in range(14)])
            #labels_ane = np.concatenate([labels_ane, label], axis = 1)
            preds_ane = (preds_ane * 10000).astype(int)

            for c in range(14):
                negative_preds = preds_ane[c][label[c] < 0.5]
                positive_preds = preds_ane[c][label[c] > 0.5]

                negative_preds_counter_ane[c] += np.bincount(negative_preds, minlength=10001)
                positive_preds_counter_ane[c] += np.bincount(positive_preds, minlength=10001)

            preds_seg = torch.softmax(out["seg_vessels"].to(DEVICE).cpu(), axis = 1).detach().numpy()[0]
            label = np.array([sample["vessels"][start:start+64,:,:].detach().numpy() == i for i in range(14)])
            #labels_seg = np.concatenate([labels_seg, label], axis = 1)
            preds_seg = (preds_seg * 10000).astype(int)

            for c in range(14):
                negative_preds = preds_seg[c][label[c] < 0.5]
                positive_preds = preds_seg[c][label[c] > 0.5]

                negative_preds_counter_seg[c] += np.bincount(negative_preds, minlength=10001)
                positive_preds_counter_seg[c] += np.bincount(positive_preds, minlength=10001)

            start += 48
        
        labels.append(sample["aneurysm_vector"].tolist())
        preds.append(pred_out)
        a = 1

    # Save labels and predictions
    np.save('labels.npy', np.array(labels))
    np.save('preds.npy', np.array(preds))
    np.save('positive_preds_counter_ane.npy', np.array(positive_preds_counter_ane))
    np.save('negative_preds_counter_ane.npy', np.array(negative_preds_counter_ane))
    np.save('positive_preds_counter_seg.npy', np.array(positive_preds_counter_seg))
    np.save('negative_preds_counter_seg.npy', np.array(negative_preds_counter_seg))

    # Calculate AUC
    auc_score = evaluate_score(labels, preds)
    auc_score_ane = evaluate_score_seg(positive_preds_counter_ane, negative_preds_counter_ane)
    auc_score_ves = evaluate_score_seg(positive_preds_counter_seg, negative_preds_counter_seg)
    print(f"AUC Score: {auc_score:.4f}")


if __name__ == "__main__":
    
    main()