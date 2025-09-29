import os
import SimpleITK as sitk
import glob
from tqdm import tqdm
import numpy as np

# ===== 0. Configurazioni =====
base_dir = "rsna-intracranial-aneurysm-detection"
output_base = "output"

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
    original_size = img.GetSize()
    original_spacing = img.GetSpacing()

    # Nuovo spacing XY per raggiungere target_size
    new_spacing = [
        original_spacing[0] * (original_size[0] / target_size[0]),
        original_spacing[1] * (original_size[1] / target_size[1]),
        original_spacing[2]  # mantieni Z
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([target_size[0], target_size[1], original_size[2]])
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)  # linear per immagini
    return resampler.Execute(img)

def resample_mask(mask_img, reference_img):
    """
    Resample mask using nearest neighbor to match reference_img
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return resampler.Execute(mask_img)

def convert_to_float32(img):
    """
    Converte Safe SimpleITK.Image in float32 usando numpy
    """
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    img_float = sitk.GetImageFromArray(arr)
    img_float.SetSpacing(img.GetSpacing())
    img_float.SetOrigin(img.GetOrigin())
    img_float.SetDirection(img.GetDirection())
    return img_float

# ===== 2. Loop sulle serie =====
for series_path in series_dirs:
    series_id = os.path.basename(series_path)
    print(f"Processing series: {series_id}")

    # ----- Leggi i DICOM -----
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(series_path)
    reader.SetFileNames(dicom_files)
    image = reader.Execute()

    # ----- Converti in float32 in modo sicuro -----
    #image = convert_to_float32(image)

    # ----- Controlla segmentazioni -----
    seg_path = os.path.join(base_dir, "segmentations", f"{series_id}.nii")
    seg_cow_path = os.path.join(base_dir, "segmentations", f"{series_id}_cowseg.nii")

    seg = sitk.ReadImage(seg_path) if os.path.exists(seg_path) else None
    seg_cow = sitk.ReadImage(seg_cow_path) if os.path.exists(seg_cow_path) else None

    # ----- Resample immagine -----
    if image.GetDimension() == 4:
        size = list(image.GetSize())
        size[3] = 0  # tieni solo un frame
        index = [0, 0, 0, 0]
        image = sitk.Extract(image, size, index)
        image = convert_to_float32(image)
    image_resampled = resample_image(image)
    sitk.WriteImage(image_resampled, os.path.join(output_base, "series", f"{series_id}.nii"))

    # ----- Resample mask -----
    if seg:
        seg_resampled = resample_mask(seg, image_resampled)
        sitk.WriteImage(seg_resampled, os.path.join(output_base, "segmentations", f"{series_id}.nii"))

    if seg_cow:
        seg_cow_resampled = resample_mask(seg_cow, image_resampled)
        sitk.WriteImage(seg_cow_resampled, os.path.join(output_base, "segmentations_cowseg", f"{series_id}_cowseg.nii"))

    print(f"Series {series_id} processed.\n")

print("All series processed!")