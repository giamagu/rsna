import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class AneurysmDatasetBuilder:
    def __init__(self, img_dir, mask_dir, csv_path, img_size=224, distance_threshold=5, test_size=0.2, random_state=42):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.csv_path = csv_path
        self.img_size = img_size
        self.distance_threshold = distance_threshold
        self.test_size = test_size
        self.random_state = random_state

        # Le 13 colonne label (ordine importante!)
        self.LABEL_COLUMNS = [
            "Anterior_communicating_artery",
            "Left_middle_cerebral_artery",
            "Right_middle_cerebral_artery",
            "Left_posterior_communicating_artery",
            "Right_posterior_communicating_artery",
            "Left_posterior_cerebral_artery",
            "Right_posterior_cerebral_artery",
            "Basilar_artery",
            "Left_internal_carotid_artery",
            "Right_internal_carotid_artery",
            "Vertebral_artery",
            "Cerebellar_artery",
            "Other"
        ]

        # Carica labels dal CSV
        self.df = pd.read_csv(self.csv_path)
        self.labels_dict = {
            row["SeriesInstanceUID"]: row[self.LABEL_COLUMNS].values.astype(int)
            for _, row in self.df.iterrows()
        }

    def build_dataset(self):
        dataset = []

        for series_uid, base_labels in self.labels_dict.items():
            img_dir = os.path.join(self.img_dir, series_uid)
            mask_dir = os.path.join(self.mask_dir, series_uid)

            if not os.path.exists(img_dir):
                continue

            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
            mask_files = sorted([f for f in os.listdir(mask_dir)]) if os.path.exists(mask_dir) else []

            has_aneurysm = base_labels.sum() > 0
            extended_labels = np.concatenate([base_labels, [int(has_aneurysm)]])

            if not has_aneurysm:
                # Serie senza aneurismi
                for img_file in img_files:
                    dataset.append({
                        "series": series_uid,
                        "slice": img_file,
                        "img_path": os.path.join(img_dir, img_file),
                        "mask_path": None,
                        "label": np.zeros(14, dtype=int)
                    })
            else:
                # Serie con aneurismi
                if not mask_files:
                    continue

                mask_slices = []
                for mfile in mask_files:
                    mpath = os.path.join(mask_dir, mfile)
                    mask = cv2.imread(mpath, cv2.IMREAD_UNCHANGED)
                    if mask is not None and np.any(mask > 0):
                        idx = int(mfile.split("slice")[-1].split(".")[0])
                        mask_slices.append(idx)

                if not mask_slices:
                    continue

                mask_slices = np.array(mask_slices)

                for img_file in img_files:
                    idx = int(img_file.split("slice")[-1].split(".")[0])

                    dist = np.min(np.abs(mask_slices - idx))
                    if dist == 0:
                        label = extended_labels
                        mask_path = os.path.join(mask_dir, f"mask_slice{idx:03d}.png")
                    elif dist >= self.distance_threshold:
                        label = np.zeros(14, dtype=int)
                        mask_path = None
                    else:
                        continue  # slice troppo vicina a una mask

                    dataset.append({
                        "series": series_uid,
                        "slice": img_file,
                        "img_path": os.path.join(img_dir, img_file),
                        "mask_path": mask_path if mask_path and os.path.exists(mask_path) else None,
                        "label": label
                    })

        self.dataset = dataset
        return dataset

    def split_train_val(self):
        if not hasattr(self, "dataset"):
            raise ValueError("Devi prima chiamare build_dataset()")

        series_ids = list(set([d["series"] for d in self.dataset]))
        train_ids, val_ids = train_test_split(
            series_ids, test_size=self.test_size, random_state=self.random_state
        )

        train_set = [d for d in self.dataset if d["series"] in train_ids]
        val_set = [d for d in self.dataset if d["series"] in val_ids]

        return train_set, val_set


builder = AneurysmDatasetBuilder(
    img_dir="preprocessed/images",
    mask_dir="preprocessed/masks",
    csv_path="rsna-intracranial-aneurysm-detection/train.csv"
)

dataset = builder.build_dataset()
train_set, val_set = builder.split_train_val()

print(f"Totale immagini: {len(dataset)}")
print(f"Train: {len(train_set)}, Val: {len(val_set)}")
