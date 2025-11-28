import torch
import os

DATA_DIR = './data/'
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
FEATURE_DIR = './features/'
MODEL_DIR = './models/'

PROJECT_TRAIN_CSV = './data/project_train.csv'
PROJECT_TEST_CSV = './data/project_test.csv'

TEXT_MODEL_NAME = 'distilbert-base-uncased'
IMAGE_MODEL_NAME = 'efficientnet_b0'
IMAGE_SIZE = 224
MAX_TEXT_LENGTH = 256
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'efficientnet_b0_feature_extractor.pth')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
LEARNING_RATE = 5e-6
EPOCHS = 15
PATIENCE = 3

BASELINE_LGBM_MODEL_PATH = os.path.join(MODEL_DIR, 'baseline_lgbm_model.joblib')
BASELINE_FEATURES_TRAIN = os.path.join(FEATURE_DIR, 'train_baseline_features.npz')
BASELINE_FEATURES_TEST = os.path.join(FEATURE_DIR, 'test_baseline_features.npz')

HYBRID_LGBM_PARAMS = {
    'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 3500,
    'learning_rate': 0.015, 'num_leaves': 200, 'max_depth': 10,
    'min_child_samples': 50, 'feature_fraction': 0.7, 'bagging_fraction': 0.5,
    'bagging_freq': 3, 'lambda_l1': 1e-05, 'lambda_l2': 0.1,
    'verbose': -1, 'n_jobs': -1, 'seed': 42, 'boosting_type': 'gbdt',
}
HYBRID_N_SPLITS = 5

HYBRID_MODELS_FOR_FEATURES = {
    'efficientnet_b0': os.path.join(MODEL_DIR, 'efficientnet_b0_feature_extractor.pth'),
}


RANDOM_STATE = 42
NUM_WORKERS = 4


