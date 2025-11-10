# #!/usr/bin/env python3
# # scripts/06_evaluate_models.py

# import os
# import sys
# import numpy as np
# import pandas as pd
# import joblib
# import torch
# from tqdm import tqdm
# from scipy.sparse import load_npz, hstack, csr_matrix
# from scipy.sparse import issparse

# # make sure src is importable
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from src.dataset import MultimodalDataset, get_tokenizer, image_transform
# from src.model import MultimodalPricePredictor
# from src.utils import smape, mae, rmse, r_squared
# import src.config as config

# def safe_squeeze(arr):
#     """Return a 1-D numpy array from model outputs (handles (N,1), (N,), (N,k))."""
#     a = np.array(arr)
#     if a.ndim == 2 and a.shape[1] == 1:
#         return a.reshape(-1)
#     if a.ndim == 1:
#         return a
#     # if multidimensional (N,k) where k>1, we try to reduce to 1D by taking first column
#     if a.ndim == 2:
#         return a[:, 0]
#     return a.ravel()

# def ensure_dir(p):
#     os.makedirs(p, exist_ok=True)

# def evaluate_baseline(y_true, results):
#     print("\n--- Evaluating Baseline LGBM Model ---")
#     try:
#         baseline_model = joblib.load(config.BASELINE_LGBM_MODEL_PATH)
#         X_test_baseline = load_npz(config.BASELINE_FEATURES_TEST)
#         print(f"Loaded baseline features shape: {X_test_baseline.shape}")

#         baseline_preds_log = baseline_model.predict(X_test_baseline)
#         baseline_preds = np.expm1(baseline_preds_log)
#         baseline_preds = np.clip(baseline_preds, 0, None)

#         results['Baseline LGBM (TFIDF+IPQ)'] = {
#             'SMAPE': smape(y_true, baseline_preds),
#             'MAE': mae(y_true, baseline_preds),
#             'RMSE': rmse(y_true, baseline_preds),
#             'R2': r_squared(y_true, baseline_preds),
#         }
#         print("Baseline LGBM evaluation complete.")
#     except FileNotFoundError:
#         print("Baseline model or features not found. Skipping baseline evaluation.")
#     except Exception as e:
#         print(f"Error during baseline evaluation: {e}")

# def evaluate_main_multimodal(y_true, results):
#     print("\n--- Evaluating Main Multimodal Model (PyTorch) ---")
#     try:
#         tokenizer = get_tokenizer()
#         test_dataset = MultimodalDataset(config.PROJECT_TEST_CSV, tokenizer, image_transform, is_test_set=False)
#         test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE * 2,
#                                                   shuffle=False, num_workers=config.NUM_WORKERS)

#         main_model_arch = 'efficientnet_b0'
#         main_model_path = config.HYBRID_MODELS_FOR_FEATURES.get(main_model_arch)
#         if not main_model_path or not os.path.exists(main_model_path):
#             main_model_path = config.MODEL_SAVE_PATH
#             if not os.path.exists(main_model_path):
#                 raise FileNotFoundError(f"Main model not found at {main_model_path}")

#         main_model = MultimodalPricePredictor(image_model_name=main_model_arch).to(config.DEVICE)
#         main_model.load_state_dict(torch.load(main_model_path, map_location=config.DEVICE))
#         main_model.eval()
#         print(f"Loaded main multimodal model: {main_model_path}")

#         main_preds_log = []
#         with torch.no_grad():
#             for batch in tqdm(test_loader, desc="Predicting with Main Model"):
#                 input_ids = batch['input_ids'].to(config.DEVICE)
#                 attention_mask = batch['attention_mask'].to(config.DEVICE)
#                 image = batch['image'].to(config.DEVICE)
#                 ipq = batch['ipq'].to(config.DEVICE)
#                 outputs = main_model(input_ids, attention_mask, image, ipq)
#                 preds_np = safe_squeeze(outputs.cpu().numpy())
#                 main_preds_log.extend(preds_np)

#         main_preds = np.expm1(np.array(main_preds_log))
#         main_preds = np.clip(main_preds, 0, None)

#         results['Main Multimodal (EfficientNet-B0)'] = {
#             'SMAPE': smape(y_true, main_preds),
#             'MAE': mae(y_true, main_preds),
#             'RMSE': rmse(y_true, main_preds),
#             'R2': r_squared(y_true, main_preds),
#         }
#         print("Main multimodal model evaluation complete.")
#     except FileNotFoundError:
#         print("Main multimodal model not found. Skipping evaluation.")
#     except Exception as e:
#         print(f"Error during main model evaluation: {e}")

# def evaluate_hybrid(y_true, results):
#     print("\n--- Evaluating Hybrid K-Fold LGBM Model ---")
#     try:
#         # Load IPQ and deep features for listed models
#         print("Loading deep test features...")
#         feature_files = []
#         models_for_features = list(config.HYBRID_MODELS_FOR_FEATURES.keys())
#         if len(models_for_features) == 0:
#             raise FileNotFoundError("No models listed in HYBRID_MODELS_FOR_FEATURES. Nothing to load for hybrid.")

#         test_deep_features = []
#         n_rows = None
#         for m in models_for_features:
#             path = os.path.join(config.FEATURE_DIR, f'test_deep_features_{m}.npy')
#             if not os.path.exists(path):
#                 raise FileNotFoundError(f"Deep features not found: {path}")
#             arr = np.load(path)
#             print(f"Raw loaded {m}: shape={arr.shape} ndim={arr.ndim}")

#             # Convert 1-D arrays into column vectors
#             if arr.ndim == 1:
#                 arr = arr.reshape(-1, 1)
#                 print(f"Reshaped {m} to {arr.shape} (converted 1D -> column)")

#             if arr.ndim != 2:
#                 raise ValueError(f"Unexpected ndim for {path}: {arr.ndim}")

#             if n_rows is None:
#                 n_rows = arr.shape[0]
#             elif arr.shape[0] != n_rows:
#                 raise ValueError(f"Row mismatch for {path}. Expected {n_rows}, got {arr.shape[0]}")

#             test_deep_features.append(arr)
#             feature_files.append(path)

#         # load ipq
#         ipq_path = os.path.join(config.FEATURE_DIR, 'test_ipq.npy')
#         if not os.path.exists(ipq_path):
#             raise FileNotFoundError(f"IPQ features not found: {ipq_path}")
#         test_ipq = np.load(ipq_path)
#         print(f"Raw loaded test_ipq: shape={test_ipq.shape} ndim={test_ipq.ndim}")
#         if test_ipq.ndim == 1:
#             test_ipq = test_ipq.reshape(-1, 1)
#             print(f"Reshaped test_ipq to {test_ipq.shape}")

#         if test_ipq.shape[0] != n_rows:
#             raise ValueError(f"Row mismatch for test_ipq. Expected {n_rows}, got {test_ipq.shape[0]}")

#         # convert all to sparse before hstack
#         sparse_parts = []
#         for arr, fpath in zip(test_deep_features, feature_files):
#             sparse_parts.append(csr_matrix(arr))
#         sparse_parts.append(csr_matrix(test_ipq))

#         X_test_hybrid = hstack(sparse_parts).tocsr()
#         print(f"Loaded hybrid features shape: {X_test_hybrid.shape} (using {models_for_features})")

#         # predict by averaging K folds if present
#         hybrid_preds_sum = np.zeros(X_test_hybrid.shape[0], dtype=float)
#         models_loaded = 0
#         for fold in range(config.HYBRID_N_SPLITS):
#             model_filename = os.path.join(config.MODEL_DIR, f'hybrid_lgbm_model_fold_{fold+1}.joblib')
#             if os.path.exists(model_filename):
#                 model = joblib.load(model_filename)
#                 preds_log = model.predict(X_test_hybrid)
#                 preds = np.expm1(preds_log)
#                 preds = np.clip(preds, 0, None)
#                 hybrid_preds_sum += preds
#                 models_loaded += 1
#                 print(f"Loaded and predicted with hybrid fold {fold+1} model.")
#             else:
#                 print(f"Warning: Hybrid model file not found for fold {fold+1}. Skipping.")

#         if models_loaded > 0:
#             hybrid_preds = hybrid_preds_sum / models_loaded
#             results['Hybrid LGBM (K-Fold, EffNet-B0)'] = {
#                 'SMAPE': smape(y_true, hybrid_preds),
#                 'MAE': mae(y_true, hybrid_preds),
#                 'RMSE': rmse(y_true, hybrid_preds),
#                 'R2': r_squared(y_true, hybrid_preds),
#             }
#             print("Hybrid K-Fold LGBM evaluation complete.")
#         else:
#             print("No hybrid K-Fold models loaded. Skipping hybrid evaluation.")
#     except FileNotFoundError as e:
#         print(f"Deep features or hybrid models not found. Skipping hybrid evaluation. ({e})")
#         print("Ensure '04_create_deep_features.py' and '05b_train_hybrid_model.py' were run.")
#     except Exception as e:
#         print(f"Error during hybrid model evaluation: {e}")

# def main():
#     print("--- Evaluating All Models on Project Test Set ---")
#     ensure_dir(config.FEATURE_DIR)
#     ensure_dir('reports')

#     # load test csv
#     try:
#         test_df = pd.read_csv(config.PROJECT_TEST_CSV)
#         y_test_true = test_df['price'].values
#         test_ids = test_df.get('sample_id', pd.Series(np.arange(len(test_df))))
#     except FileNotFoundError:
#         print(f"Error: Project test CSV not found at {config.PROJECT_TEST_CSV}")
#         sys.exit(1)

#     results = {}
#     evaluate_baseline(y_test_true, results)
#     evaluate_main_multimodal(y_test_true, results)
#     evaluate_hybrid(y_test_true, results)

#     # Final results
#     print("\n\nFinal Evaluation Results ")
#     if len(results) == 0:
#         print("No results to display.")
#         return

#     results_df = pd.DataFrame(results).T
#     # format numeric columns
#     results_df = results_df[['SMAPE', 'MAE', 'RMSE', 'R2']]
#     print(results_df.to_string())
#     print("----------------------------------")

#     out_path = os.path.join('reports', 'final_evaluation_metrics.csv')
#     results_df.to_csv(out_path, float_format='%.6f')
#     print(f"\nResults saved to {out_path}")

# if __name__ == '__main__':
#     main()

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import joblib
from scipy.sparse import load_npz, hstack, csr_matrix
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
predictions_df = pd.DataFrame({'sample_id': test_ids, 'actual_price': y_test_true})

print("\n--- Evaluating Model 1: Baseline LGBM (TFIDF+IPQ) ---")
try:
    baseline_model = joblib.load(config.BASELINE_LGBM_MODEL_PATH)
    X_test_baseline = load_npz(config.BASELINE_FEATURES_TEST)
    print(f"Loaded baseline features shape: {X_test_baseline.shape}")

    baseline_preds_log = baseline_model.predict(X_test_baseline)
    baseline_preds = np.expm1(baseline_preds_log)
    baseline_preds = np.clip(baseline_preds, 0, None)
    predictions_df['pred_baseline_tfidf'] = baseline_preds

    results['1. Baseline (TF-IDF)'] = {
        'SMAPE': smape(y_test_true, baseline_preds),
        'MAE': mae(y_test_true, baseline_preds),
        'RMSE': rmse(y_test_true, baseline_preds),
        'R2': r_squared(y_test_true, baseline_preds),
    }
    print("Baseline (TF-IDF) evaluation complete.")
except Exception as e:
    print(f"Could not evaluate Baseline (TF-IDF) model: {e}")

print("\n--- Evaluating Model 2: Main Multimodal (PyTorch) ---")
try:
    tokenizer = get_tokenizer()
    test_dataset = MultimodalDataset(config.PROJECT_TEST_CSV, tokenizer, image_transform, is_test_set=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE * 2, shuffle=False, num_workers=config.NUM_WORKERS)
    
    main_model_arch = config.IMAGE_MODEL_NAME
    main_model_path = config.MODEL_SAVE_PATH
    
    if not os.path.exists(main_model_path):
        main_model_path = config.HYBRID_MODELS_FOR_FEATURES.get(main_model_arch)
        if not os.path.exists(main_model_path):
            raise FileNotFoundError(f"Main PyTorch model not found at {config.MODEL_SAVE_PATH} or {config.HYBRID_MODELS_FOR_FEATURES.get(main_model_arch)}")

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
    predictions_df['pred_main_pytorch'] = main_preds

    results[f'2. Main Multimodal (PyTorch {main_model_arch})'] = {
        'SMAPE': smape(y_test_true, main_preds),
        'MAE': mae(y_test_true, main_preds),
        'RMSE': rmse(y_test_true, main_preds),
        'R2': r_squared(y_test_true, main_preds),
    }
    print("Main multimodal model evaluation complete.")
except Exception as e:
    print(f"Could not evaluate Main Multimodal model: {e}")

print("\n--- Evaluating Model 3: Strong Baseline (K-Fold Text-Only) ---")
try:
    print("Loading deep text features...")
    test_ipq = np.load(os.path.join(config.FEATURE_DIR, 'test_ipq.npy'))
    if test_ipq.ndim == 1:
        test_ipq = test_ipq.reshape(-1, 1)
    
    model_arch = list(config.HYBRID_MODELS_FOR_FEATURES.keys())[0]
    test_text_features = np.load(os.path.join(config.FEATURE_DIR, f'test_text_features_{model_arch}.npy'))
    
    X_test_strong_baseline = np.hstack([test_text_features, test_ipq])
    print(f"Loaded strong baseline features shape: {X_test_strong_baseline.shape}")

    strong_baseline_preds_sum = np.zeros(X_test_strong_baseline.shape[0])
    models_loaded = 0
    for fold in range(config.HYBRID_N_SPLITS):
        model_filename = os.path.join(config.MODEL_DIR, f'strong_baseline_lgbm_fold_{fold+1}.joblib')
        if os.path.exists(model_filename):
            model = joblib.load(model_filename)
            preds_log_fold = model.predict(X_test_strong_baseline)
            strong_baseline_preds_sum += np.expm1(preds_log_fold)
            models_loaded += 1
            print(f"Loaded and predicted with strong baseline fold {fold+1} model.")
        else:
            print(f"Warning: Strong baseline model file not found for fold {fold+1}. Skipping.")

    if models_loaded == config.HYBRID_N_SPLITS:
        strong_baseline_preds = strong_baseline_preds_sum / models_loaded
        strong_baseline_preds = np.clip(strong_baseline_preds, 0, None)
        predictions_df['pred_strong_baseline'] = strong_baseline_preds

        results['3. Strong Baseline (K-Fold Text)'] = {
            'SMAPE': smape(y_test_true, strong_baseline_preds),
            'MAE': mae(y_test_true, strong_baseline_preds),
            'RMSE': rmse(y_test_true, strong_baseline_preds),
            'R2': r_squared(y_test_true, strong_baseline_preds),
        }
        print("Strong Baseline (K-Fold Text) evaluation complete.")
    else:
        print(f"Could not load all {config.HYBRID_N_SPLITS} strong baseline models. Skipping evaluation.")
except Exception as e:
    print(f"Could not evaluate Strong Baseline model: {e}")

print("\n--- Evaluating Model 4: FINAL Hybrid (K-Fold Multimodal) ---")
try:
    print("Loading all deep features...")
    test_ipq = np.load(os.path.join(config.FEATURE_DIR, 'test_ipq.npy'))
    if test_ipq.ndim == 1:
        test_ipq = test_ipq.reshape(-1, 1)

    test_features_list = []
    models_for_features = list(config.HYBRID_MODELS_FOR_FEATURES.keys())
    for model_arch in models_for_features:
        test_text_features = np.load(os.path.join(config.FEATURE_DIR, f'test_text_features_{model_arch}.npy'))
        test_image_features = np.load(os.path.join(config.FEATURE_DIR, f'test_image_features_{model_arch}.npy'))
        test_features_list.append(test_text_features)
        test_features_list.append(test_image_features)
    
    test_features_list.append(test_ipq)
    X_test_final_hybrid = np.hstack(test_features_list)
    print(f"Loaded final hybrid features shape: {X_test_final_hybrid.shape} (using {models_for_features})")

    final_hybrid_preds_sum = np.zeros(X_test_final_hybrid.shape[0])
    models_loaded = 0
    for fold in range(config.HYBRID_N_SPLITS):
        model_filename = os.path.join(config.MODEL_DIR, f'final_hybrid_lgbm_fold_{fold+1}.joblib')
        if os.path.exists(model_filename):
            model = joblib.load(model_filename)
            preds_log_fold = model.predict(X_test_final_hybrid)
            final_hybrid_preds_sum += np.expm1(preds_log_fold)
            models_loaded += 1
            print(f"Loaded and predicted with final hybrid fold {fold+1} model.")
        else:
            print(f"Warning: Final hybrid model file not found for fold {fold+1}. Skipping.")

    if models_loaded == config.HYBRID_N_SPLITS:
        final_hybrid_preds = final_hybrid_preds_sum / models_loaded
        final_hybrid_preds = np.clip(final_hybrid_preds, 0, None)
        predictions_df['pred_final_hybrid'] = final_hybrid_preds

        results[f'4. Final Hybrid (K-Fold Multimodal)'] = {
            'SMAPE': smape(y_test_true, final_hybrid_preds),
            'MAE': mae(y_test_true, final_hybrid_preds),
            'RMSE': rmse(y_test_true, final_hybrid_preds),
            'R2': r_squared(y_test_true, final_hybrid_preds),
        }
        print("Final Hybrid (K-Fold Multimodal) evaluation complete.")
    else:
        print(f"Could not load all {config.HYBRID_N_SPLITS} final hybrid models. Skipping evaluation.")
except Exception as e:
    print(f"Could not evaluate Final Hybrid model: {e}")

print("\n\nFinal Evaluation Results")
results_df = pd.DataFrame(results).T.sort_values(by='SMAPE')
print(results_df.to_string())
print("----------------------------------")

results_df.to_csv(os.path.join('reports', 'final_evaluation_metrics.csv'))
predictions_df.to_csv(os.path.join('reports', 'final_predictions.csv'), index=False)
print("\nResults saved to reports/final_evaluation_metrics.csv")
print("Predictions saved to reports/final_predictions.csv")
