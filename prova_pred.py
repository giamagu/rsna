import os
import shutil
import numpy as np
import pandas as pd
import polars as pl
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from model import Unet3D


ID_TO_LABEL = {
    1:  "Other Posterior Circulation",
    2:  "Basilar Tip",
    3:  "Right Posterior Communicating Artery",
    4:  "Left Posterior Communicating Artery",
    5:  "Right Infraclinoid Internal Carotid Artery",
    6:  "Left Infraclinoid Internal Carotid Artery",
    7:  "Right Supraclinoid Internal Carotid Artery",
    8:  "Left Supraclinoid Internal Carotid Artery",
    9:  "Right Middle Cerebral Artery",
    10: "Left Middle Cerebral Artery",
    11: "Right Anterior Cerebral Artery",
    12: "Left Anterior Cerebral Artery",
    13: "Anterior Communicating Artery"
}

LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}

# ===== 1. Funzioni utili =====
def resample_image(img, target_size=(224, 224)):
    """
    Resample image in XY a target_size, keeping Z
    Linear interpolation for images
    """
    original_size = img.GetSize()  # (X, Y, Z)
    original_spacing = img.GetSpacing()

    # nuovo spacing XY
    new_spacing = [
        original_spacing[0] * (original_size[0] / target_size[0]),
        original_spacing[1] * (original_size[1] / target_size[1]),
        original_spacing[2]
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([target_size[0], target_size[1], original_size[2]])
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(img)

def resample_mask(mask_img, reference_img):
    """Resample mask using nearest neighbor to match reference_img"""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return resampler.Execute(mask_img)

def convert_to_float32(img):
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    img_float = sitk.GetImageFromArray(arr)
    img_float.SetSpacing(img.GetSpacing())
    img_float.SetOrigin(img.GetOrigin())
    img_float.SetDirection(img.GetDirection())
    return img_float

def save_slices(arr, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(arr.shape[0]):
        out_path = os.path.join(out_dir, f"{prefix}_slice_{i}.png")
        imageio.imwrite(out_path, arr[i])

def save_volume(arr, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    
    # Assicuriamoci che sia uint8
    arr_uint8 = arr.astype(np.uint8)
    
    out_path = os.path.join(out_dir, f"{prefix}.npy")
    np.save(out_path, arr_uint8)
    print(f"Salvato: {out_path} con shape {arr_uint8.shape} e dtype {arr_uint8.dtype}")

def sitk_to_uint8(img, is_mask=False):
    arr = sitk.GetArrayFromImage(img) # Z,H,W
    if is_mask:
        return arr.astype(np.uint8)
    else:
        arr = arr.astype(np.float32)
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)
        return (arr * 255).astype(np.uint8)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Colonne obbligatorie della competition
ID_COL = 'SeriesInstanceUID'
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

# Carica il modello UNA volta
model = Unet3D().to(DEVICE)
model.eval()

# Funzione di inferenza per singola serie DICOM
def predict(series_path: str) -> pl.DataFrame:
    series_id = os.path.basename(series_path)

    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(series_path)
    reader.SetFileNames(dicom_files)
    image = reader.Execute()

    # Gestione 4D
    if image.GetDimension() == 4:
        size = list(image.GetSize())
        size[3] = 0
        index = [0, 0, 0, 0]
        image = sitk.Extract(image, size, index)
        image = convert_to_float32(image)

    # Resample + Normalizzazione
    image_resampled = resample_image(image, target_size=(224,224))
    arr_img = sitk_to_uint8(image_resampled, is_mask=False)
    arr_img = arr_img.astype(np.float32) / 255.0

    # Converti in tensore (1,1,D,H,W)
    tensor = torch.tensor(arr_img[None, None]).to(DEVICE)

    # Crea tensore di shape (1, 1, 863, 224, 224)
    tensor = torch.zeros((1, 1, 176, 224, 224), dtype=torch.float32).to(DEVICE)

    left_out = tensor.shape[2]
    while left_out > 48:
        left_out -= 44

    pad_total = (16 - (left_out % 16)) % 16  # 0 se già multiplo di 16
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    
    # Padding in formato (D, H, W): (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
    # Qui padding per asse 2 => terzo asse da destra => si passa (padW, padW, padH, padH, padD, padD)
    tensor = F.pad(tensor, (0, 0, 0, 0, pad_left, pad_right))

    global_pred = [0]*14

    # Inferenza
    with torch.no_grad():
        start = 0
        while start < tensor.shape[2]-4:
            out = model(tensor[:,:,start:start+48,:,:])["class"].sigmoid().cpu().numpy().flatten()
            global_pred = [max(global_pred[i], out[i]) for i in range(len(out))]
            start += 44

    out = global_pred

    # Controllo dimensione
    assert len(out) == 14, f"Expected 14 outputs, got {len(out)}"

    # Crea DataFrame
    predictions = [out[0]] + list(out[1:])  # [aneurysm present] + 13 arteries
    df = pl.DataFrame(
        data=[[series_id] + predictions],
        schema=[ID_COL, *LABEL_COLS],
        orient='row'
    )

    # Pulizia necessaria (richiesta da Kaggle)
    shutil.rmtree('/kaggle/shared', ignore_errors=True)

    return df.drop(ID_COL)

predict("rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647")