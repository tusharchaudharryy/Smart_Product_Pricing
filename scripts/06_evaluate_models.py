import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch
from tqdm import tqdm
from scipy.sparse import load_npz, hstack, csr_matrix
from scipy.sparse import issparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import MultimodalDataset, get_tokenizer, image_transform
from src.model import MultimodalPricePredictor
from src.utils import smape, mae, rmse, r_squared
import src.config as config

def safe_squeeze(arr):
    a = np.array(arr)
    if a.ndim == 2 and a.shape[1] == 1:
        return a.reshape(-1)
    if a.ndim == 1:
        return a
    if a.ndim == 2:
        return a[:, 0]
    return a.ravel()

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def evaluate_baseline(y_true, results):
    print("\nEvaluating Baseline LGBM Model ")
    try:
        baseline_model = joblib.load(config.BASELINE_LGBM_MODEL_PATH)
        X_test_baseline = load_npz(config.BASELINE_FEATURES_TEST)
        print(f"Loaded baseline features shape: {X_test_baseline.shape}")

        baseline_preds_log = baseline_model.predict(X_test_baseline)
        baseline_preds = np.expm1(baseline_preds_log)
        baseline_preds = np.clip(baseline_preds, 0, None)

        results['Baseline LGBM (TFIDF+IPQ)'] = {
            'SMAPE': smape(y_true, baseline_preds),
            'MAE': mae(y_true, baseline_preds),
            'RMSE': rmse(y_true, baseline_preds),
            'R2': r_squared(y_true, baseline_preds),
        }
        print("Baseline LGBM evaluation complete.")
    except FileNotFoundError:
        print("Baseline model or features not found. Skipping baseline evaluation.")
    except Exception as e:
        print(f"Error during baseline evaluation: {e}")

def evaluate_main_multimodal(y_true, results):
    print("\nEvaluating Main Multimodal Model (PyTorch)")
    try:
        tokenizer = get_tokenizer()
        test_dataset = MultimodalDataset(config.PROJECT_TEST_CSV, tokenizer, image_transform, is_test_set=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE * 2,
                                                  shuffle=False, num_workers=config.NUM_WORKERS)

        main_model_arch = 'efficientnet_b0'
        main_model_path = config.HYBRID_MODELS_FOR_FEATURES.get(main_model_arch)
        if not main_model_path or not os.path.exists(main_model_path):
            main_model_path = config.MODEL_SAVE_PATH
            if not os.path.exists(main_model_path):
                raise FileNotFoundError(f"Main model not found at {main_model_path}")

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
                preds_np = safe_squeeze(outputs.cpu().numpy())
                main_preds_log.extend(preds_np)

        main_preds = np.expm1(np.array(main_preds_log))
        main_preds = np.clip(main_preds, 0, None)

        results['Main Multimodal (EfficientNet-B0)'] = {
            'SMAPE': smape(y_true, main_preds),
            'MAE': mae(y_true, main_preds),
            'RMSE': rmse(y_true, main_preds),
            'R2': r_squared(y_true, main_preds),
        }
        print("Main multimodal model evaluation complete.")
    except FileNotFoundError:
        print("Main multimodal model not found. Skipping evaluation.")
    except Exception as e:
        print(f"Error during main model evaluation: {e}")

def evaluate_hybrid(y_true, results):
    print("\nEvaluating Hybrid K-Fold LGBM Model")
    try:
        print("Loading deep test features...")
        feature_files = []
        models_for_features = list(config.HYBRID_MODELS_FOR_FEATURES.keys())
        if len(models_for_features) == 0:
            raise FileNotFoundError("No models listed in HYBRID_MODELS_FOR_FEATURES. Nothing to load for hybrid.")

        test_deep_features = []
        n_rows = None
        for m in models_for_features:
            path = os.path.join(config.FEATURE_DIR, f'test_deep_features_{m}.npy')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Deep features not found: {path}")
            arr = np.load(path)
            print(f"Raw loaded {m}: shape={arr.shape} ndim={arr.ndim}")

            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
                print(f"Reshaped {m} to {arr.shape} (converted 1D -> column)")

            if arr.ndim != 2:
                raise ValueError(f"Unexpected ndim for {path}: {arr.ndim}")

            if n_rows is None:
                n_rows = arr.shape[0]
            elif arr.shape[0] != n_rows:
                raise ValueError(f"Row mismatch for {path}. Expected {n_rows}, got {arr.shape[0]}")

            test_deep_features.append(arr)
            feature_files.append(path)

        ipq_path = os.path.join(config.FEATURE_DIR, 'test_ipq.npy')
        if not os.path.exists(ipq_path):
            raise FileNotFoundError(f"IPQ features not found: {ipq_path}")
        test_ipq = np.load(ipq_path)
        print(f"Raw loaded test_ipq: shape={test_ipq.shape} ndim={test_ipq.ndim}")
        if test_ipq.ndim == 1:
            test_ipq = test_ipq.reshape(-1, 1)
            print(f"Reshaped test_ipq to {test_ipq.shape}")

        if test_ipq.shape[0] != n_rows:
            raise ValueError(f"Row mismatch for test_ipq. Expected {n_rows}, got {test_ipq.shape[0]}")

        sparse_parts = []
        for arr, fpath in zip(test_deep_features, feature_files):
            sparse_parts.append(csr_matrix(arr))
        sparse_parts.append(csr_matrix(test_ipq))

        X_test_hybrid = hstack(sparse_parts).tocsr()
        print(f"Loaded hybrid features shape: {X_test_hybrid.shape} (using {models_for_features})")

        hybrid_preds_sum = np.zeros(X_test_hybrid.shape[0], dtype=float)
        models_loaded = 0
        for fold in range(config.HYBRID_N_SPLITS):
            model_filename = os.path.join(config.MODEL_DIR, f'hybrid_lgbm_model_fold_{fold+1}.joblib')
            if os.path.exists(model_filename):
                model = joblib.load(model_filename)
                preds_log = model.predict(X_test_hybrid)
                preds = np.expm1(preds_log)
                preds = np.clip(preds, 0, None)
                hybrid_preds_sum += preds
                models_loaded += 1
                print(f"Loaded and predicted with hybrid fold {fold+1} model.")
            else:
                print(f"Warning: Hybrid model file not found for fold {fold+1}. Skipping.")

        if models_loaded > 0:
            hybrid_preds = hybrid_preds_sum / models_loaded
            results['Hybrid LGBM (K-Fold, EffNet-B0)'] = {
                'SMAPE': smape(y_true, hybrid_preds),
                'MAE': mae(y_true, hybrid_preds),
                'RMSE': rmse(y_true, hybrid_preds),
                'R2': r_squared(y_true, hybrid_preds),
            }
            print("Hybrid K-Fold LGBM evaluation complete.")
        else:
            print("No hybrid K-Fold models loaded. Skipping hybrid evaluation.")
    except FileNotFoundError as e:
        print(f"Deep features or hybrid models not found. Skipping hybrid evaluation. ({e})")
        print("Ensure '04_create_deep_features.py' and '05b_train_hybrid_model.py' were run.")
    except Exception as e:
        print(f"Error during hybrid model evaluation: {e}")

def main():
    print("--- Evaluating All Models on Project Test Set ---")
    ensure_dir(config.FEATURE_DIR)
    ensure_dir('reports')

    try:
        test_df = pd.read_csv(config.PROJECT_TEST_CSV)
        y_test_true = test_df['price'].values
        test_ids = test_df.get('sample_id', pd.Series(np.arange(len(test_df))))
    except FileNotFoundError:
        print(f"Error: Project test CSV not found at {config.PROJECT_TEST_CSV}")
        sys.exit(1)

    results = {}
    evaluate_baseline(y_test_true, results)
    evaluate_main_multimodal(y_test_true, results)
    evaluate_hybrid(y_test_true, results)

    print("\n\nFinal Evaluation Results ")
    if len(results) == 0:
        print("No results to display.")
        return

    results_df = pd.DataFrame(results).T
    results_df = results_df[['SMAPE', 'MAE', 'RMSE', 'R2']]
    print(results_df.to_string())

    out_path = os.path.join('reports', 'final_evaluation_metrics.csv')
    results_df.to_csv(out_path, float_format='%.6f')
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    main()
