import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.utils import download_images_from_df
import src.config as config

print(" Downloading Images ")

try:
    train_df = pd.read_csv(config.PROJECT_TRAIN_CSV)
    test_df = pd.read_csv(config.PROJECT_TEST_CSV)
except FileNotFoundError as e:
    print(f"Error loading CSV file: {e}")
    print("Ensure '00_split_data.py' has been run.")
    exit()

all_data_df = pd.concat([
    train_df[['sample_id', 'image_link']],
    test_df[['sample_id', 'image_link']],
], ignore_index=True).drop_duplicates(subset=['sample_id'])

download_images_from_df(all_data_df, config.IMAGE_DIR)

print(" Image Download Complete ")
