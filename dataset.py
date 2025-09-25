import os
import numpy as np
import pandas as pd
from PIL import Image
import ast

class AneurysmDataset:
    def __init__(self, img_dir, csv_path, n=1):
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.n = n  # Numero di immagini vicine da includere nelle label
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
        self.num_labels = len(self.LABEL_COLUMNS) + 1  # 13 + 1 per "almeno un aneurisma"
        self.localizers = self._load_localizers()

    def _load_localizers(self):
        """Carica il file train_localizers.csv e crea un dizionario con le informazioni sugli aneurismi."""
        df = pd.read_csv(self.csv_path)
        localizers = {}

        for _, row in df.iterrows():
            sop_uid = row["SOPInstanceUID"]
            coordinates = ast.literal_eval(row["coordinates"])
            location = row["location"]

            # Trova l'indice della vena corrispondente
            if location in self.LABEL_COLUMNS:
                label_idx = self.LABEL_COLUMNS.index(location)
            else:
                continue

            # Aggiungi al dizionario
            if sop_uid not in localizers:
                localizers[sop_uid] = {"labels": np.zeros(len(self.LABEL_COLUMNS)), "slices": []}
            localizers[sop_uid]["labels"][label_idx] = 1
            if "f" in row:  # Se c'è un'informazione sulla slice
                localizers[sop_uid]["slices"].append(int(row["f"]))

        return localizers

    # Modifica della funzione build_dataset
    def build_dataset(self):
        dataset = []

        for series_uid in os.listdir(self.img_dir):
            series_path = os.path.join(self.img_dir, series_uid)
            if not os.path.isdir(series_path):
                continue

            img_files = sorted([f for f in os.listdir(series_path) if f.endswith(".png")])

            for i, img_file in enumerate(img_files):
                sop_uid = img_file.split(".")[0]  # Sop UID è il nome del file senza estensione
                img_path = os.path.join(series_path, img_file)

                # Carica l'immagine
                img = Image.open(img_path)

                # Inizializza la label
                label = np.zeros(self.num_labels)

                # Inizializza la maschera
                mask = np.zeros((224, 224), dtype=np.uint8)

                # Se l'immagine è in localizers, aggiorna la label e la maschera
                if sop_uid in self.localizers:
                    aneurysm_info = self.localizers[sop_uid]
                    label[:len(self.LABEL_COLUMNS)] = aneurysm_info["labels"]

                    # Genera la maschera per le coordinate dell'aneurisma
                    for coord in aneurysm_info.get("coordinates", []):
                        x, y = int(coord['x']), int(coord['y'])
                        if 0 <= x < 224 and 0 <= y < 224:  # Assicurati che le coordinate siano valide
                            mask[y, x] = 1

                    # Estendi la label alle immagini vicine
                    for slice_offset in range(-self.n, self.n + 1):
                        neighbor_idx = i + slice_offset
                        if 0 <= neighbor_idx < len(img_files):
                            neighbor_file = img_files[neighbor_idx]
                            neighbor_uid = neighbor_file.split(".")[0]
                            if neighbor_uid in self.localizers:
                                label[:len(self.LABEL_COLUMNS)] |= self.localizers[neighbor_uid]["labels"]

                # Calcola la 14esima entrata (almeno un aneurisma)
                label[-1] = int(label[:len(self.LABEL_COLUMNS)].sum() > 0)

                # Aggiungi al dataset l'immagine, la label e la maschera
                dataset.append((img, label, mask))

        return dataset