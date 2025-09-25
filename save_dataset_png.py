import os
import cv2
import numpy as np
import pydicom
from tqdm import tqdm

# =========================
# CONFIG
# =========================
BASE_PATH = "rsna-intracranial-aneurysm-detection"
SERIES_DIR = os.path.join(BASE_PATH, "series")
OUTPUT_IMG_DIR = "preprocessed/images"
IMG_SIZE = 224

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# =========================
# Funzione utilità
# =========================
def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalizza immagine in [0, 255] e converte a uint8."""
    img = img.astype(np.float32)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    return (img * 255).astype(np.uint8)

# =========================
# Processing
# =========================
series_list = sorted(os.listdir(SERIES_DIR))
print(f"Totale series: {len(series_list)}")

for series_uid in tqdm(series_list):
    series_path = os.path.join(SERIES_DIR, series_uid)
    if not os.path.isdir(series_path):
        continue

    out_series_dir = os.path.join(OUTPUT_IMG_DIR, series_uid)
    os.makedirs(out_series_dir, exist_ok=True)

    dicom_files = sorted(os.listdir(series_path))
    for dcm_file in dicom_files:
        dcm_path = os.path.join(series_path, dcm_file)

        try:
            dcm = pydicom.dcmread(dcm_path)
            img = dcm.pixel_array
        except Exception as e:
            print(f"Errore con {dcm_path}: {e}")
            continue

        # Caso 2D
        if img.ndim == 2:
            out_path = os.path.join(out_series_dir, dcm_file.replace(".dcm", ".png"))
            if os.path.isfile(out_path):
                continue  # Salta se già esiste
            img = normalize_to_uint8(img)
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(out_path, img_resized)

        # Caso 3D (più slice in un DICOM)
        elif img.ndim == 3:
            num_slices = img.shape[0]
            for i in range(num_slices):
                out_path = os.path.join(out_series_dir, f"{dcm_file.replace('.dcm','')}_slice{i:03d}.png")
                if os.path.isfile(out_path):
                    continue  # Salta se già esiste
                slice_img = normalize_to_uint8(img[i])
                slice_resized = cv2.resize(slice_img, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(out_path, slice_resized)

        else:
            print(f"Shape inattesa {img.shape} in {dcm_path}")
            continue

