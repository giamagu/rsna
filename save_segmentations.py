import os
import cv2
import numpy as np
import nibabel as nib
from tqdm import tqdm

# =========================
# CONFIG
# =========================
BASE_PATH = "rsna-intracranial-aneurysm-detection"
SEG_DIR = os.path.join(BASE_PATH, "segmentations")
OUTPUT_MASK_DIR = "preprocessed/masks"
IMG_SIZE = 224

os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

# =========================
# Processing
# =========================
seg_files = [f for f in os.listdir(SEG_DIR) if f.endswith(".nii")]
print(f"Totale segmentazioni trovate: {len(seg_files)}")

for seg_file in tqdm(seg_files):
    seg_path = os.path.join(SEG_DIR, seg_file)

    # Ricava SeriesInstanceUID dal nome (prima di eventuale _cowseg)
    series_uid = seg_file.split("_")[0].replace(".nii", "")

    out_series_dir = os.path.join(OUTPUT_MASK_DIR, series_uid)
    os.makedirs(out_series_dir, exist_ok=True)

    try:
        nii = nib.load(seg_path)
        mask_data = nii.get_fdata().astype(np.uint8)
    except Exception as e:
        print(f"Errore con {seg_path}: {e}")
        continue

    # Caso 3D
    if mask_data.ndim == 3:
        num_slices = mask_data.shape[2]
        for i in range(num_slices):
            out_path = os.path.join(out_series_dir, f"mask_slice{i:03d}.png")
            if os.path.isfile(out_path):
                continue  # Salta se già esiste
            slice_mask = mask_data[:, :, i]

            # Resize
            slice_resized = cv2.resize(
                slice_mask, (IMG_SIZE, IMG_SIZE),
                interpolation=cv2.INTER_NEAREST  # importante per maschere
            )

            cv2.imwrite(out_path, slice_resized)

    # Caso 4D (alcune segmentazioni potrebbero avere extra dimensione)
    elif mask_data.ndim == 4:
        num_slices = mask_data.shape[2]
        for i in range(num_slices):
            out_path = os.path.join(out_series_dir, f"mask_slice{i:03d}.png")
            if os.path.isfile(out_path):
                continue  # Salta se già esiste
            slice_mask = mask_data[:, :, i, 0]

            slice_resized = cv2.resize(
                slice_mask, (IMG_SIZE, IMG_SIZE),
                interpolation=cv2.INTER_NEAREST
            )

            cv2.imwrite(out_path, slice_resized)

    else:
        print(f"Shape inattesa {mask_data.shape} in {seg_path}")
        continue

