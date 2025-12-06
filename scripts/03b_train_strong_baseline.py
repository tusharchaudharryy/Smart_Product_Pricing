import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
import joblib
import gc
from scipy.sparse import hstack, csr_matrix

from src.utils import smape
import src.config as config

FEATURE_DIR = config.FEATURE_DIR
MODEL_DIR = config.MODEL_DIR
TRAIN_CSV = config.PROJECT_TRAIN_CSV

MODEL_ARCH = list(config.HYBRID_MODELS_FOR_FEATURES.keys())[0] 
N_SPLITS = config.HYBRID_N_SPLITS
PARAMS = config.HYBRID_LGBM_PARAMS
RANDOM_STATE = config.RANDOM_STATE

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"--- Training Strong Baseline K-Fold LGBM Model (Text Only) ---")
print(f"Using features from: {MODEL_ARCH}")


try:
    train_ipq = np.load(os.path.join(FEATURE_DIR, 'train_ipq.npy'))
    train_text_features = np.load(os.path.join(FEATURE_DIR, f'train_text_features_{MODEL_ARCH}.npy'))
    

    if train_ipq.ndim == 1:
        train_ipq = train_ipq.reshape(-1, 1)
        
    X = np.hstack([train_text_features, train_ipq]) 

    train_df = pd.read_csv(TRAIN_CSV)
    y = np.log1p(train_df['price'])
    y_true_orig = train_df['price'].values
except FileNotFoundError as e:
    print(f"Error loading features or train CSV: {e}")
    print("Ensure '05_create_deep_features.py' was run successfully.")
    exit()

del train_text_features, train_ipq, train_df
gc.collect()

print(f"Full training feature matrix shape: {X.shape}")


kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
oof_preds = np.zeros(len(X))

print(f"\nStarting training with {N_SPLITS}-Fold Cross-Validation...")

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='mae',
              callbacks=[lgb.early_stopping(100, verbose=100)])

    val_preds_log = model.predict(X_val)
    oof_preds[val_index] = np.expm1(val_preds_log)

    model_filename = os.path.join(MODEL_DIR, f'strong_baseline_lgbm_fold_{fold+1}.joblib')
    joblib.dump(model, model_filename)
    print(f"Fold {fold+1} model saved to {model_filename}")

    del X_train, X_val, y_train, y_val, model
    gc.collect()


oof_preds = np.clip(oof_preds, 0, None)
final_oof_smape = smape(y_true_orig, oof_preds)
print(f"\n Cross-Validation Finished ")
print(f"Final OOF (Out-of-Fold) SMAPE for Strong Baseline: {final_oof_smape:.4f}")
print("-----------------------------------")