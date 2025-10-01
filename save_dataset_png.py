import os
import SimpleITK as sitk
import glob
from tqdm import tqdm
import numpy as np
import imageio
import pandas as pd
import ast


# ===== 5. Funzione per plottare le tre immagini =====
import matplotlib.pyplot as plt

def plot_three_images(series_dir, seg_dir, seg_cowseg_dir, uid, i):
    """
    Plotta le tre immagini (serie, segmentazione, cowseg) per un dato uid e indice i.
    """
    img_paths = [
        os.path.join(series_dir, uid, os.listdir(os.path.join(series_dir, uid))[i]),
        os.path.join(seg_dir, uid, os.listdir(os.path.join(seg_dir, uid))[i]),
        os.path.join(seg_cowseg_dir, uid, os.listdir(os.path.join(seg_cowseg_dir, uid))[i])
    ]
    titles = ["Serie", "Segmentazione", "Cowseg"]
    imgs = [imageio.imread(p) for p in img_paths]
    imgs[2]*=18
    if np.max(imgs[2])<1:
        return
    plt.figure(figsize=(12,4))
    for idx, (img, title) in enumerate(zip(imgs, titles)):
        plt.subplot(1,3,idx+1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# ===== 0. Configurazioni =====
base_dir = "/kaggle/input/rsna-intracranial-aneurysm-detection"
output_base = "output"

seg_dir = os.path.join(output_base, "segmentations")
seg_cowseg_dir = os.path.join(output_base, "segmentations_cowseg")
series_dir = os.path.join(output_base, "series")

uid = os.listdir(seg_dir)[0]

for i in range(500):
    #continue
    plot_three_images(series_dir, seg_dir, seg_cowseg_dir, uid, i)

# Cartelle di destinazione
os.makedirs(os.path.join(output_base, "series"), exist_ok=True)
os.makedirs(os.path.join(output_base, "segmentations"), exist_ok=True)
os.makedirs(os.path.join(output_base, "segmentations_cowseg"), exist_ok=True)

# Lista delle serie DICOM
series_dirs = glob.glob(os.path.join(base_dir, "series/*"))

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

def sitk_to_uint8(img, is_mask=False):
    arr = sitk.GetArrayFromImage(img) # Z,H,W
    if is_mask:
        return arr.astype(np.uint8)
    else:
        arr = arr.astype(np.float32)
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)
        return (arr * 255).astype(np.uint8)

# ===== 2. Carica localizers =====
localizers_path = os.path.join(base_dir, "train_localizers.csv")
localizers = pd.read_csv(localizers_path)

new_records = []

# ===== 3. Loop sulle serie =====
for series_path in series_dirs[:40]:
    series_id = os.path.basename(series_path)
    print(f"Processing series: {series_id}")

    # ----- Controlla segmentazioni -----
    seg_path = os.path.join(base_dir, "segmentations", f"{series_id}.nii")
    seg_cow_path = os.path.join(base_dir, "segmentations", f"{series_id}_cowseg.nii")

    seg = sitk.ReadImage(seg_path) if os.path.exists(seg_path) else None
    seg_cow = sitk.ReadImage(seg_cow_path) if os.path.exists(seg_cow_path) else None

    if not seg:
        continue

    # ----- Leggi i DICOM -----
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(series_path)
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
    save_slices(arr_img, os.path.join(output_base, "series", series_id), series_id)

    if seg:
        seg_resampled = resample_mask(seg, image_resampled)
        arr_seg = sitk_to_uint8(seg_resampled, is_mask=False)
        save_slices(arr_seg, os.path.join(output_base, "segmentations", series_id), series_id)

    if seg_cow:
        seg_cow_resampled = resample_mask(seg_cow, image_resampled)
        arr_seg_cow = sitk_to_uint8(seg_cow_resampled, is_mask=True)
        save_slices(arr_seg_cow, os.path.join(output_base, "segmentations_cowseg", series_id), series_id)

    # ----- Aggiorna localizers per questa serie -----
    df_series = localizers[localizers["SeriesInstanceUID"] == series_id]
    
    # lista ordinata dei DICOM della serie
    dicom_list = reader.GetGDCMSeriesFileNames(series_path)
    
    # mappa SOPInstanceUID -> indice slice globale
    sop_uid_to_index = {}
    offset = 0  # contatore cumulativo slice viste finora
    
    for f in dicom_list:
        img = sitk.ReadImage(f)
        if img.HasMetaDataKey("0008|0018"):
            sop_uid = img.GetMetaData("0008|0018")
    
            # quante slice ha questo file
            size = img.GetSize()  # (X, Y, Z)
            n_slices = size[2] if img.GetDimension() == 3 else 1
    
            # assegna un indice globale a tutte le slice di questo file
            for z in range(n_slices):
                sop_uid_to_index[(sop_uid, z)] = offset + z
    
            offset += n_slices  # aggiorna il contatore

    for _, row in df_series.iterrows():
        coords = ast.literal_eval(row["coordinates"])
        x_old, y_old = coords["x"], coords["y"]
        sop_uid = row["SOPInstanceUID"]
    
        # scala XY alle nuove dimensioni
        new_x = x_old * (224 / original_size[0])
        new_y = y_old * (224 / original_size[1])
    
        f_old = coords.get("f", None)
        if f_old is not None:
            # caso normale: f specificato
            key = (sop_uid, int(f_old))
            if key in sop_uid_to_index:
                f_global = sop_uid_to_index[key]
            else:
                print(f"[WARNING] Slice f={f_old} per SOP={sop_uid} non trovata in {series_id}, uso centro volume")
                f_global = new_size[2] // 2
        else:
            # caso senza f: DICOM deve avere 1 sola slice
            # cerco se esistono entry multiple per lo stesso sop_uid
            matching_keys = [k for k in sop_uid_to_index.keys() if k[0] == sop_uid]
            if len(matching_keys) == 1:
                f_global = sop_uid_to_index[matching_keys[0]]
            else:
                print(f"[ERROR] SOP={sop_uid} in serie {series_id} ha {len(matching_keys)} slice ma manca f!")
                continue  # salta questo record
    
        new_coords = {"x": float(new_x), "y": float(new_y), "f": int(f_global)}
    
        new_records.append({
            "SeriesInstanceUID": row["SeriesInstanceUID"],
            "coordinates": str(new_coords),
            "location": row["location"]
        })

# ===== 4. Salva nuovo CSV =====
out_csv = os.path.join(output_base, "train_localizers_resampled.csv")
pd.DataFrame(new_records).to_csv(out_csv, index=False)
print("Saved new localizers to:", out_csv)


    for _, row in df_series.iterrows():
        coords = ast.literal_eval(row["coordinates"])
        x_old, y_old = coords["x"], coords["y"]
        sop_uid = row["SOPInstanceUID"]
    
        # scala XY alle nuove dimensioni
        new_x = x_old * (224 / original_size[0])
        new_y = y_old * (224 / original_size[1])
    
        f_old = coords.get("f", None)
        if f_old is not None:
            # caso normale: f specificato
            key = (sop_uid, int(f_old))
            if key in sop_uid_to_index:
                f_global = sop_uid_to_index[key]
            else:
                print(f"[WARNING] Slice f={f_old} per SOP={sop_uid} non trovata in {series_id}, uso centro volume")
                f_global = new_size[2] // 2
        else:
            # caso senza f: DICOM deve avere 1 sola slice
            # cerco se esistono entry multiple per lo stesso sop_uid
            matching_keys = [k for k in sop_uid_to_index.keys() if k[0] == sop_uid]
            if len(matching_keys) == 1:
                f_global = sop_uid_to_index[matching_keys[0]]
            else:
                print(f"[ERROR] SOP={sop_uid} in serie {series_id} ha {len(matching_keys)} slice ma manca f!")
                continue  # salta questo record
    
        new_coords = {"x": float(new_x), "y": float(new_y), "f": int(f_global)}
    
        new_records.append({
            "SeriesInstanceUID": row["SeriesInstanceUID"],
            "coordinates": str(new_coords),
            "location": row["location"]
        })


print("All series processed!")

# ===== 4. Salva nuovo CSV =====
out_csv = os.path.join(output_base, "train_localizers_resampled.csv")
pd.DataFrame(new_records).to_csv(out_csv, index=False)
print("Saved new localizers to:", out_csv)