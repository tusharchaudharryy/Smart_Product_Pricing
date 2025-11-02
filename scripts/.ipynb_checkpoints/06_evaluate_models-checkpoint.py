import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import joblib
from scipy.sparse import load_npz, hstack
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.dataset import MultimodalDataset, get_tokenizer, image_transform
from src.model import MultimodalPricePredictor
from src.utils import smape, mae, rmse, r_squared
import src.config as config

print("--- Evaluating All Models on Project Test Set ---")
os.makedirs(config.FEATURE_DIR, exist_ok=True) 

try:
    test_df = pd.read_csv(config.PROJECT_TEST_CSV)
    y_test_true = test_df['price'].values
    test_ids = test_df['sample_id']
except FileNotFoundError:
    print(f"Error: Project test CSV not found at {config.PROJECT_TEST_CSV}")
    exit()

results = {}
print("\n--- Evaluating Baseline LGBM Model ---")
try:
    baseline_model = joblib.load(config.BASELINE_LGBM_MODEL_PATH)
    X_test_baseline = load_npz(config.BASELINE_FEATURES_TEST)
    print(f"Loaded baseline features shape: {X_test_baseline.shape}")

    baseline_preds_log = baseline_model.predict(X_test_baseline)
    baseline_preds = np.expm1(baseline_preds_log)
    baseline_preds = np.clip(baseline_preds, 0, None)

    results['Baseline LGBM (TFIDF+IPQ)'] = {
        'SMAPE': smape(y_test_true, baseline_preds),
        'MAE': mae(y_test_true, baseline_preds),
        'RMSE': rmse(y_test_true, baseline_preds),
        'R2': r_squared(y_test_true, baseline_preds),
    }
    print("Baseline LGBM evaluation complete.")

except FileNotFoundError:
    print("Baseline model or features not found. Skipping baseline evaluation.")
except Exception as e:
    print(f"Error during baseline evaluation: {e}")

print("\n--- Evaluating Main Multimodal Model (PyTorch) ---")
try:
    tokenizer = get_tokenizer()
    test_dataset = MultimodalDataset(config.PROJECT_TEST_CSV, tokenizer, image_transform, is_test_set=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE * 2, shuffle=False, num_workers=config.NUM_WORKERS)

    main_model_arch = 'efficientnet_b0' 
    main_model_path = config.HYBRID_MODELS_FOR_FEATURES.get(main_model_arch) 
    
    if not main_model_path or not os.path.exists(main_model_path):
        main_model_path = config.MODEL_SAVE_PATH 
        if not os.path.exists(main_model_path):
             raise FileNotFoundError(f"Good 'efficientnet_b0' model not found at {main_model_path} or in HYBRID_MODELS_FOR_FEATURES")
    main_model = MultimodalPricePredictor(image_model_name=main_model_arch).to(config.DEVICE)
    main_model.load_state_dict(torch.load(main_model_path, map_location=config.DEVICE))
    main_model.eval()
    print(f"Loaded main multimodal model: {main_model_path}")

    main_preds_log = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting with Main Model"):
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            image = batch['image'].to(config.DEVICE)
            ipq = batch['ipq'].to(config.DEVICE)
            outputs = main_model(input_ids, attention_mask, image, ipq)
            main_preds_log.extend(outputs.cpu().numpy())

    main_preds = np.expm1(np.array(main_preds_log))
    main_preds = np.clip(main_preds, 0, None)

    results['Main Multimodal (EfficientNet-B0)'] = { 
        'SMAPE': smape(y_test_true, main_preds),
        'MAE': mae(y_test_true, main_preds),
        'RMSE': rmse(y_test_true, main_preds),
        'R2': r_squared(y_test_true, main_preds),
    }
    print("Main multimodal model evaluation complete.")

except FileNotFoundError:
    print(f"Main multimodal model not found. Skipping evaluation.")
except Exception as e:
    print(f"Error during main model evaluation: {e}")

print("\n--- Evaluating Hybrid K-Fold LGBM Model ---")
try:
    print("Loading deep test features...")
    test_ipq = np.load(os.path.join(config.FEATURE_DIR, 'test_ipq.npy'))
    models_for_features = list(config.HYBRID_MODELS_FOR_FEATURES.keys())
    test_deep_features = [np.load(os.path.join(config.FEATURE_DIR, f'test_deep_features_{m}.npy'))
                           for m in models_for_features]
    if test_ipq.ndim == 1:
        test_ipq = test_ipq.reshape(-1, 1)
    
    X_test_hybrid = hstack(test_deep_features + [test_ipq]).tocsr()
    print(f"Loaded hybrid features shape: {X_test_hybrid.shape} (using {models_for_features})")

    hybrid_preds_sum = np.zeros(X_test_hybrid.shape[0])
    
    models_loaded = 0
    for fold in range(config.HYBRID_N_SPLITS):
        model_filename = os.path.join(config.MODEL_DIR, f'hybrid_lgbm_model_fold_{fold+1}.joblib')
        if os.path.exists(model_filename):
            model = joblib.load(model_filename)
            hybrid_preds_log_fold = model.predict(X_test_hybrid)
            hybrid_preds_sum += np.expm1(hybrid_preds_log_fold)
            models_loaded += 1
            print(f"Loaded and predicted with hybrid fold {fold+1} model.")
        else:
            print(f"Warning: Hybrid model file not found for fold {fold+1}. Skipping.")

    if models_loaded > 0:
        hybrid_preds = hybrid_preds_sum / models_loaded
        hybrid_preds = np.clip(hybrid_preds, 0, None)

        results['Hybrid LGBM (K-Fold, EffNet-B0)'] = { 
            'SMAPE': smape(y_test_true, hybrid_preds),
            'MAE': mae(y_test_true, hybrid_preds),
            'RMSE': rmse(y_test_true, hybrid_preds),
            'R2': r_squared(y_test_true, hybrid_preds),
        }
        print("Hybrid K-Fold LGBM evaluation complete.")
    else:
         print("No hybrid K-Fold models loaded. Skipping evaluation.")

except FileNotFoundError:
    print("Deep features or hybrid models not found. Skipping hybrid evaluation.")
    print("Ensure '04_create_deep_features.py' and '05b_train_hybrid_model.py' were run.")
except Exception as e:
    print(f"Error during hybrid model evaluation: {e}")

print("\n\nFinal Evaluation Results ")
results_df = pd.DataFrame(results).T 
print(results_df.to_string())
print("----------------------------------")

results_df.to_csv(os.path.join('reports', 'final_evaluation_metrics.csv'))
print("\nResults saved to reports/final_evaluation_metrics.csv")
