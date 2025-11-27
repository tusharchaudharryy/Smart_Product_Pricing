import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import os
import pandas as pd
import multiprocessing
from tqdm import tqdm
from pathlib import Path
from functools import partial
import urllib.request
import re

def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return np.mean(ratio) * 100

def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true)**2))

def r_squared(y_true, y_pred):

    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

def _download_single_image(image_link_tuple, savefolder):
    
    image_link, filename_base = image_link_tuple
    if isinstance(image_link, str) and isinstance(filename_base, str):
        image_save_path = os.path.join(savefolder, f"{filename_base}.jpg")
        if not os.path.exists(image_save_path):
            try:
                urllib.request.urlretrieve(image_link, image_save_path)
            except Exception as ex:
                print(f'Warning: Download failed - {image_link}\n{ex}', file=sys.stderr)
    return

def download_images_from_df(df, download_folder, num_workers=16):
    if 'image_link' not in df.columns or 'sample_id' not in df.columns:
        print("Error: 'image_link' and 'sample_id' columns required.", file=sys.stderr)
        return

    os.makedirs(download_folder, exist_ok=True)
    image_link_tuples = list(zip(df['image_link'].astype(str), df['sample_id'].astype(str)))

    print(f"Starting download of {len(image_link_tuples)} images to {download_folder}...")
    download_func = partial(_download_single_image, savefolder=download_folder)

    with multiprocessing.Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(download_func, image_link_tuples), total=len(image_link_tuples), desc="Downloading Images"))

    print("Image download process complete.")

def extract_ipq(text_series):
    if not isinstance(text_series, pd.Series):
        raise TypeError("Input must be a pandas Series.")
    ipq = text_series.str.extract(r'Item Pack Quantity:\s*(\d+)', expand=False)
    ipq = pd.to_numeric(ipq, errors='coerce').fillna(1.0)
    return ipq.values.reshape(-1, 1) 

    