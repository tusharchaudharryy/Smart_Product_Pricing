import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.model_selection import train_test_split
import src.config as config

print(" Splitting Data ")

ORIGINAL_TRAIN_PATH = os.path.join(config.DATA_DIR, 'train.csv')

if not os.path.exists(ORIGINAL_TRAIN_PATH):
    print(f"Error: Original train file not found at {ORIGINAL_TRAIN_PATH}")
else:
    original_train_df = pd.read_csv(ORIGINAL_TRAIN_PATH)
    print(f"Loaded original training data: {len(original_train_df)} rows")

    project_train_df, project_test_df = train_test_split(
        original_train_df,
        test_size=0.2,
        random_state=config.RANDOM_STATE
    )
    print(f"Splitting data: {len(project_train_df)} train rows, {len(project_test_df)} test rows")

    project_train_df.to_csv(config.PROJECT_TRAIN_CSV, index=False)
    project_test_df.to_csv(config.PROJECT_TEST_CSV, index=False)

    print(f"New training set saved to: {config.PROJECT_TRAIN_CSV}")
    print(f"New test set saved to: {config.PROJECT_TEST_CSV}")
    print(" Data Splitting Complete ")