import os
import numpy as np
import torch
from torch.utils.data import Dataset

class RSNA3DDataset(Dataset):
    def __init__(self, root_dir, series_ids, transform=None):
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
        self.maximum_slices = 16

        # Lista delle cartelle → assumiamo che abbiano lo stesso nome
        self.ids = []
        for dir in os.listdir(self.series_dir):
            if dir in self.series_ids:
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

        # Estrai un sottoslice di dimensione 8xHxW
        if x.shape[0] > self.maximum_slices:
            start_idx = np.random.randint(0, x.shape[0] - self.maximum_slices + 1)
            x = x[start_idx:start_idx + self.maximum_slices]
            y_aneurysm = y_aneurysm[start_idx:start_idx + self.maximum_slices]
            y_vessels = y_vessels[start_idx:start_idx + self.maximum_slices]

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

        if sample["image"].shape[0] < self.maximum_slices:
            # Padding per avere sempre dimensione (16, H, W)
            pad_width = self.maximum_slices - sample["image"].shape[0]
            pad_before = pad_width // 2
            pad_after = pad_width - pad_before

            sample["image"] = np.pad(sample["image"], ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant', constant_values=0)
            sample["vessels"] = np.pad(sample["vessels"], ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant', constant_values=-1)
            sample["aneurysms"] = np.pad(sample["aneurysms"], ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant', constant_values=0)

        # Converti in tensori dopo le trasformazioni
        sample["image"] = torch.from_numpy(sample["image"]).unsqueeze(0).float()  # (1, D, H, W)
        sample["vessels"] = torch.from_numpy(sample["vessels"]).long()  # (D, H, W)
        sample["aneurysms"] = torch.from_numpy(sample["aneurysms"]).long()  # (D, H, W)
        sample["aneurysm_vector"] = torch.from_numpy(sample["aneurysm_vector"]).float()  # (14,)

        return sample
