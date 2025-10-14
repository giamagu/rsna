import os
import shutil
import numpy as np
import pandas as pd
import polars as pl
import SimpleITK as sitk
import torch

# Importa la tua rete e funzioni di utilità
from unet3d import Unet3D            # <-- percorso corretto del tuo modello
from utils_preprocessing import resample_image, convert_to_float32, sitk_to_uint8  # <-- se li hai in un file

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
ckpt_path = "/kaggle/input/your-model/checkpoint_epoch_15.pth"  # aggiorna se serve
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
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

    # Inferenza
    with torch.no_grad():
        out = model(tensor)["class"].sigmoid().cpu().numpy().flatten()

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

# ---- Esegui server per test locale / valutazione Kaggle ----
import kaggle_evaluation.rsna_inference_server as rsna

inference_server = rsna.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    display(pl.read_parquet('/kaggle/working/submission.parquet'))
