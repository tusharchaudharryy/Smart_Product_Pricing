import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from scipy.sparse import load_npz
import joblib
from src.utils import smape, mae, rmse, r_squared
import src.config as config

os.makedirs(config.MODEL_DIR, exist_ok=True)

print("Training Baseline LightGBM Model")

try:
    X = load_npz(config.BASELINE_FEATURES_TRAIN)
    train_df = pd.read_csv(config.PROJECT_TRAIN_CSV)
    y = np.log1p(train_df['price'])
    y_true_orig = train_df['price'].values
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit()

print(f"Loaded features shape: {X.shape}")

X_train, X_val, y_train, y_val, _, y_val_true_orig = train_test_split(
    X, y, y_true_orig, test_size=0.15, random_state=config.RANDOM_STATE
)

params = {
    'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 2000,
    'learning_rate': 0.05, 'num_leaves': 31, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 1, 'verbose': -1,
    'n_jobs': -1, 'seed': config.RANDOM_STATE,
}

print("Training baseline model")
model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          eval_metric='mae',
          callbacks=[lgb.early_stopping(100, verbose=True)])

val_preds_log = model.predict(X_val)
val_preds = np.expm1(val_preds_log)

val_smape = smape(y_val_true_orig, val_preds)
val_mae = mae(y_val_true_orig, val_preds)
val_rmse = rmse(y_val_true_orig, val_preds)
val_r2 = r_squared(y_val_true_orig, val_preds)

print(f"\nBaseline Model Validation Metrics")
print(f"  SMAPE: {val_smape:.4f}")
print(f"  MAE:   {val_mae:.2f}")
print(f"  RMSE:  {val_rmse:.2f}")
print(f"  R²:    {val_r2:.4f}")

print("\nTraining final baseline model on all training data")
final_model = lgb.LGBMRegressor(**params)
final_model.fit(X, y)

joblib.dump(final_model, config.BASELINE_LGBM_MODEL_PATH)
print(f"Baseline model saved to: {config.BASELINE_LGBM_MODEL_PATH}")
print("Baseline Model Training Complete ")
