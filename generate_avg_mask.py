import os
import cv2
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation

# ======================
# CONFIG
# ======================
SEG_DIR = "/kaggle/working/preprocessed/masks"
IMG_SIZE = 224
NUM_LABELS = 13
DILATE_PIXELS = 10

# ======================
# UTILS
# ======================
def resize_mask(mask, size=(IMG_SIZE, IMG_SIZE)):
    return cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)

def dilate_mask(mask, pixels):
    struct = np.ones((pixels*2+1, pixels*2+1), dtype=bool)
    return binary_dilation(mask, structure=struct)

# ======================
# MAIN
# ======================
sum_masks = np.zeros((NUM_LABELS, IMG_SIZE, IMG_SIZE), dtype=np.float32)
count_masks = np.zeros(NUM_LABELS, dtype=np.int32)

for series_uid in os.listdir(SEG_DIR):
    series_path = os.path.join(SEG_DIR, series_uid)
    if not os.path.isdir(series_path):
        continue

    mask_files = sorted([f for f in os.listdir(series_path) if f.endswith(".png")])

    for fname in mask_files:
        mask = cv2.imread(os.path.join(series_path, fname), cv2.IMREAD_UNCHANGED)

        if mask is None:
            continue

        # Resize
        mask = resize_mask(mask)

        # Etichette presenti
        labels_present = np.unique(mask)
        labels_present = labels_present[labels_present > 0]  # rimuovi background

        if len(labels_present) != 1:
            continue  # ignora slice con più di una label

        label_id = labels_present[0]
        if not (1 <= label_id <= NUM_LABELS):
            continue

        # Crea mask binaria
        binary = (mask == label_id)

        # Dilata
        dilated = dilate_mask(binary, DILATE_PIXELS)

        # Accumula
        sum_masks[label_id - 1] += dilated.astype(np.float32)
        count_masks[label_id - 1] += 1

# ======================
# AVERAGE MASKS
# ======================
avg_masks = []
for i in range(NUM_LABELS):
    if count_masks[i] > 0:
        avg = sum_masks[i] / count_masks[i]
    else:
        avg = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    avg_masks.append(avg)

avg_masks = np.array(avg_masks)  # shape = (13, 224, 224)

print("Shape finali:", avg_masks.shape)

out_dir = "/kaggle/working/avg_masks"
os.makedirs(out_dir, exist_ok=True)

for i, avg in enumerate(avg_masks):
    cv2.imwrite(os.path.join(out_dir, f"avg_mask_label{i+1}.png"),
                (avg * 255).astype(np.uint8))
