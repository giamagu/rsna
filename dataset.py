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

        # Ogni cartella contiene un solo npy → lo prendiamo
        series_file = [f for f in os.listdir(series_path) if f.endswith(".npy")][0]
        aneurysm_file = [f for f in os.listdir(aneurysm_path) if f.endswith(".npy")][0]
        vessels_file = [f for f in os.listdir(vessels_path) if f.endswith(".npy")][0]

        # Carica i volumi
        x = np.load(os.path.join(series_path, series_file))        # input
        y_aneurysm = np.load(os.path.join(aneurysm_path, aneurysm_file))  # mask aneurismi
        y_vessels = np.load(os.path.join(vessels_path, vessels_file))     # mask vasi

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

        # Cast a torch tensor
        x = torch.from_numpy(x).unsqueeze(0).float()           # (1, D, H, W)
        y_aneurysm = torch.from_numpy(y_aneurysm).long()       # (D, H, W)
        y_vessels = torch.from_numpy(y_vessels).long()         # (D, H, W)
        label_vec = torch.from_numpy(label_vec).float()        # (14,)

        sample = {
            "id": series_id,
            "image": x,
            "vessels": y_vessels,
            "aneurysms": y_aneurysm,
            "aneurysm_vector": label_vec
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
