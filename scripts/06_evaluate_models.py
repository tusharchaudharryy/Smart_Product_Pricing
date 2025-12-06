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

print("Evaluating All Models on Project Test Set")
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

print("\nEvaluating Model 1: Baseline LGBM (TFIDF+IPQ)")
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

print("\nEvaluating Model 2: Main Multimodal (PyTorch)")
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

print("\nEvaluating Model 3: Strong Baseline (K-Fold Text-Only)")
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

print("\nEvaluating Model 4: FINAL Hybrid (K-Fold Multimodal)")
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
