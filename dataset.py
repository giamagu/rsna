import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

# ===== Funzione di utilità per il plot =====
import matplotlib.pyplot as plt

def plot_x_y_slice(x, y, idx=274):
    """
    Plotta fianco a fianco x[idx,:,:] e y[idx,:,:].
    """
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(x[idx,:,:], cmap='gray')
    plt.title(f'x[{idx}]')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(y[idx,:,:], cmap='gray')
    plt.title(f'y[{idx}]')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


class RSNA3DDataset(Dataset):
    def __init__(self, root_dir, series_ids, maximum_slices = 16, minimum_slices = 16, transform=None, only_vessels = False):
        """
        Args:
            root_dir (str): directory principale del dataset resampled_dataset
                deve contenere:
                - series/
                - aneurysm_labels/
                - segmentations_cowseg/
            transform: trasformazioni opzionali (es. augmentazioni, normalizzazioni)
        """
        self.root_dir = root_dir
        self.series_dir = os.path.join(root_dir, "series")
        self.aneurysm_dir = os.path.join(root_dir, "aneurysm_labels")
        self.vessels_dir = os.path.join(root_dir, "segmentations_cowseg")
        self.transform = transform
        self.series_ids = series_ids
        self.maximum_slices = maximum_slices
        self.minimum_slices = minimum_slices
        self.only_vessels = only_vessels

        # Lista delle cartelle → assumiamo che abbiano lo stesso nome
        self.ids = []
        for dir in os.listdir(self.series_dir):
            if dir in self.series_ids:
                if self.only_vessels and not os.path.exists(os.path.join(self.vessels_dir, dir)):
                    continue
                self.ids.append(dir)
        self.ids.sort()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        series_id = self.ids[idx]

        # Path ai file npy (ogni cartella contiene un solo npy)
        series_path = os.path.join(self.series_dir, series_id)
        aneurysm_path = os.path.join(self.aneurysm_dir, series_id)
        vessels_path = os.path.join(self.vessels_dir, series_id)

        # Carica i volumi# Ogni cartella contiene un solo npy → lo prendiamo
        series_file = [f for f in os.listdir(series_path) if f.endswith(".npy")][0]
        x = np.load(os.path.join(series_path, series_file))        # input
        try:
            aneurysm_file = [f for f in os.listdir(aneurysm_path) if f.endswith(".npy")][0]
            y_aneurysm = np.load(os.path.join(aneurysm_path, aneurysm_file))  # mask aneurismi
        except:
            y_aneurysm = np.zeros_like(x, dtype=np.uint8)
        try:
            vessels_file = [f for f in os.listdir(vessels_path) if f.endswith(".npy")][0]
            y_vessels = np.load(os.path.join(vessels_path, vessels_file))     # mask vasi
        except:
            y_vessels = np.zeros_like(x, dtype=np.float32) - 1

        # Trova i canali in cui c'è almeno un 1 in y_aneurysm
        channels_with_aneurysm = np.any(y_aneurysm > 0, axis=(1, 2))

        # Crea un array con gli indici dei canali
        aneurysm_channels = np.where(channels_with_aneurysm)[0]

        if x.shape[0] > self.maximum_slices:

            if random.random() < 0.75 and len(aneurysm_channels) > 0:
                selected_channel = random.choice(aneurysm_channels)
                start_idx = max(0, selected_channel - random.randint(2, self.maximum_slices - 3))
            else:
                start_idx = np.random.randint(0, x.shape[0] - self.maximum_slices + 1)

            # Estrai un sottoslice di dimensione 16xHxW
            if x.shape[0] > self.maximum_slices:
                x = x[start_idx:start_idx + self.maximum_slices]
                y_aneurysm = y_aneurysm[start_idx:start_idx + self.maximum_slices]
                y_vessels = y_vessels[start_idx:start_idx + self.maximum_slices]
            if x.shape[0] > self.minimum_slices and x.shape[0] % self.minimum_slices != 0:
                # Riduci a un multiplo di minimum_slices
                new_depth = (x.shape[0] // self.minimum_slices) * self.minimum_slices
                leftover = x.shape[0] - new_depth
                start_crop = leftover // 2
                x = x[start_crop:start_crop + new_depth]
                y_aneurysm = y_aneurysm[start_crop:start_crop + new_depth]
                y_vessels = y_vessels[start_crop:start_crop + new_depth]

        if np.max(y_vessels) > 200:
            a = 1
            

        # Crea vettore 14-D da aneurysm label
        # posizione 0 = generico (1 se c’è almeno un aneurisma in qualunque vaso)
        # posizione 1..13 = 1 se c’è almeno un pixel con valore == i
        label_vec = np.zeros(14, dtype=np.float32)
        unique_vals = np.unique(y_aneurysm)

        # 1..13
        for i in range(1, 14):
            if i in unique_vals:
                label_vec[i] = 1.0
        # 0 = generico
        if np.any(label_vec[1:]):
            label_vec[0] = 1.0

        x = x.astype(np.float32) / 255

        # Cast a torch tensor
        sample = {
            "id": series_id,
            "image": x,  # Mantieni x come NumPy array
            "vessels": y_vessels,  # Mantieni y_vessels come NumPy array
            "aneurysms": y_aneurysm,  # Mantieni y_aneurysm come NumPy array
            "aneurysm_vector": label_vec  # Mantieni label_vec come NumPy array
        }

        # Ruota le immagini e le maschere per Albumentations (da (D, H, W) a (H, W, D))
        sample["image"] = np.transpose(sample["image"], (1, 2, 0))  # (H, W, D)
        sample["vessels"] = np.transpose(sample["vessels"], (1, 2, 0))  # (H, W, D)
        sample["aneurysms"] = np.transpose(sample["aneurysms"], (1, 2, 0))  # (H, W, D)


        if self.transform:
            # Applica le trasformazioni
            augmented = self.transform(
                image=sample["image"],
                vessels=sample["vessels"],
                aneurysms=sample["aneurysms"]
            )

            # Aggiorna solo i campi trasformati
            sample["image"] = augmented["image"]
            sample["vessels"] = augmented["vessels"]
            sample["aneurysms"] = augmented["aneurysms"]

        # Ruota le immagini e le maschere indietro (da (H, W, D) a (D, H, W))
        sample["image"] = np.transpose(sample["image"], (2, 0, 1))  # (D, H, W)
        sample["vessels"] = np.transpose(sample["vessels"], (2, 0, 1))  # (D, H, W)
        sample["aneurysms"] = np.transpose(sample["aneurysms"], (2, 0, 1))  # (D, H, W)

        if sample["image"].shape[0] < self.minimum_slices:
            # Padding per avere sempre dimensione (16, H, W)
            pad_width = self.minimum_slices - sample["image"].shape[0]
            pad_before = pad_width // 2
            pad_after = pad_width - pad_before

            sample["image"] = np.pad(sample["image"], ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant', constant_values=0)
            sample["vessels"] = np.pad(sample["vessels"], ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant', constant_values=0)
            sample["aneurysms"] = np.pad(sample["aneurysms"], ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant', constant_values=0)

        if np.max(sample["vessels"]) > 0.5:
            a = 1
        if np.max(sample["aneurysms"]) > 200:
            a = 1

        # Converti in tensori dopo le trasformazioni
        sample["image"] = torch.from_numpy(sample["image"]).unsqueeze(0).float()  # (1, D, H, W)
        sample["vessels"] = torch.from_numpy(sample["vessels"]).long()  # (D, H, W)
        sample["aneurysms"] = torch.from_numpy(sample["aneurysms"]).long()  # (D, H, W)
        sample["aneurysm_vector"] = torch.from_numpy(sample["aneurysm_vector"]).float()  # (14,)
        return sample