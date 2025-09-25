import pandas as pd
import ast
import os
import pydicom

# =========================
# CONFIG
# =========================
BASE_PATH = "rsna-intracranial-aneurysm-detection"
SERIES_DIR = os.path.join(BASE_PATH, "series")
OUTPUT_CSV = "train_localizers_resampled.csv"
IMG_SIZE = 224

# =========================
# Funzione per calcolare nuove coordinate
# =========================
def resample_coordinates(dcm_path, coordinates):
    """Calcola le nuove coordinate proporzionate al resize."""
    try:
        dcm = pydicom.dcmread(dcm_path)
        original_height, original_width = dcm.pixel_array.shape
        scale_x = IMG_SIZE / original_width
        scale_y = IMG_SIZE / original_height

        new_x = coordinates['x'] * scale_x
        new_y = coordinates['y'] * scale_y
        return {'x': new_x, 'y': new_y}
    except Exception as e:
        print(f"Errore durante il calcolo delle coordinate per {dcm_path}: {e}")
        return None

# =========================
# Lettura del file CSV
# =========================
localizers_df = pd.read_csv("train_localizers.csv")

# =========================
# Iterazione e calcolo nuove coordinate
# =========================
resampled_data = []

for _, row in localizers_df.iterrows():
    series_uid = row['SeriesInstanceUID']
    sop_uid = row['SOPInstanceUID']
    coordinates = ast.literal_eval(row['coordinates'])  # Converte la stringa in dizionario

    # Percorso al file DICOM
    dcm_path = os.path.join(SERIES_DIR, series_uid, f"{sop_uid}.dcm")
    if not os.path.exists(dcm_path):
        print(f"File DICOM non trovato: {dcm_path}")
        continue

    # Calcolo delle nuove coordinate
    new_coordinates = resample_coordinates(dcm_path, coordinates)
    if new_coordinates:
        resampled_data.append({
            'SeriesInstanceUID': series_uid,
            'SOPInstanceUID': sop_uid,
            'coordinates': new_coordinates,
            'location': row['location']
        })

# =========================
# Salvataggio nuovo CSV
# =========================
resampled_df = pd.DataFrame(resampled_data)
resampled_df.to_csv(OUTPUT_CSV, index=False)
print(f"Nuovo file CSV salvato in: {OUTPUT_CSV}")