import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, save_npz
from src.utils import extract_ipq
import src.config as config

os.makedirs(config.FEATURE_DIR, exist_ok=True)

print(" Creating Baseline Features (TF-IDF + IPQ) ")

try:
    train_df = pd.read_csv(config.PROJECT_TRAIN_CSV)
    test_df = pd.read_csv(config.PROJECT_TEST_CSV)
except FileNotFoundError as e:
    print(f"Error loading CSV: {e}")
    exit()

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
full_df = pd.concat([train_df.drop(columns=['price']), test_df], axis=0, ignore_index=True)

print("Calculating TF-IDF features")
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, stop_words='english')
text_features = tfidf.fit_transform(full_df['catalog_content'].fillna(''))

print("Extracting IPQ features")
ipq_features = extract_ipq(full_df['catalog_content'])

print("Combining features")
X_full = hstack([text_features, ipq_features]).tocsr()

X_train_baseline = X_full[:len(train_df)]
X_test_baseline = X_full[len(train_df):]

print(f"Train feature shape: {X_train_baseline.shape}")
print(f"Test feature shape: {X_test_baseline.shape}")

save_npz(config.BASELINE_FEATURES_TRAIN, X_train_baseline)
save_npz(config.BASELINE_FEATURES_TEST, X_test_baseline)

print(f"Baseline training features saved to: {config.BASELINE_FEATURES_TRAIN}")
print(f"Baseline test features saved to: {config.BASELINE_FEATURES_TEST}")
print(" Baseline Feature Creation Complete ")

