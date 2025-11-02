import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import time

from src.dataset import MultimodalDataset, get_tokenizer, image_transform
from src.model import MultimodalPricePredictor
from src.utils import smape, mae, rmse, r_squared
import src.config as config

print(f" Training Main Multimodal Model ")
print(f"Using device: {config.DEVICE}")
os.makedirs(config.MODEL_DIR, exist_ok=True)

try:
    full_train_df = pd.read_csv(config.PROJECT_TRAIN_CSV)
except FileNotFoundError:
    print(f"Error: Training CSV not found at {config.PROJECT_TRAIN_CSV}")
    exit()

train_indices, val_indices = train_test_split(
    range(len(full_train_df)), test_size=0.15, random_state=config.RANDOM_STATE
)
print(f"Data split: {len(train_indices)} training samples, {len(val_indices)} validation samples")

tokenizer = get_tokenizer()
full_dataset = MultimodalDataset(config.PROJECT_TRAIN_CSV, tokenizer, image_transform, is_test_set=False)

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE * 2, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

model = MultimodalPricePredictor().to(config.DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

best_val_smape = float('inf')
epochs_no_improve = 0
training_start_time = time.time()

for epoch in range(config.EPOCHS):
    epoch_start_time = time.time()
    model.train()
    total_train_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]", leave=False)
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        image = batch['image'].to(config.DEVICE)
        ipq = batch['ipq'].to(config.DEVICE)
        targets = batch['target'].to(config.DEVICE)
        outputs = model(input_ids, attention_mask, image, ipq)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item()
        progress_bar.set_postfix(loss=total_train_loss / (progress_bar.n + 1e-6))
    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    val_preds_log, val_targets_log, val_prices_true = [], [], []
    progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Val]", leave=False)
    with torch.no_grad():
        for batch in progress_bar_val:
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            image = batch['image'].to(config.DEVICE)
            ipq = batch['ipq'].to(config.DEVICE)
            targets = batch['target'].to(config.DEVICE)
            outputs = model(input_ids, attention_mask, image, ipq)
            val_preds_log.extend(outputs.cpu().numpy())
            val_targets_log.extend(targets.cpu().numpy())
            val_prices_true.extend(np.expm1(targets.cpu().numpy()))

    val_preds_orig = np.expm1(np.array(val_preds_log))
    val_prices_true = np.array(val_prices_true)

    val_smape = smape(val_prices_true, val_preds_orig)
    val_mae = mae(val_prices_true, val_preds_orig)
    val_rmse = rmse(val_prices_true, val_preds_orig)
    val_r2 = r_squared(val_prices_true, val_preds_orig)
    avg_val_loss = criterion(torch.tensor(val_preds_log), torch.tensor(val_targets_log)).item()

    epoch_duration = time.time() - epoch_start_time
    print(f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val SMAPE: {val_smape:.4f} | Val MAE: {val_mae:.2f} | Val RMSE: {val_rmse:.2f} | Val R2: {val_r2:.4f} | Time: {epoch_duration:.2f}s")

    if val_smape < best_val_smape:
        best_val_smape = val_smape
        epochs_no_improve = 0
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        print(f"   Val SMAPE improved to {best_val_smape:.4f}. Model saved to {config.MODEL_SAVE_PATH}")
    else:
        epochs_no_improve += 1
        print(f"  Val SMAPE did not improve. Counter: {epochs_no_improve}/{config.PATIENCE}")

    if epochs_no_improve >= config.PATIENCE:
        print(f"\nEarly stopping triggered after {epoch+1} epochs.")
        break

total_training_time = time.time() - training_start_time
print(f"\n Training Complete ")
print(f"Best Validation SMAPE achieved: {best_val_smape:.4f}")
print(f"Model saved to: {config.MODEL_SAVE_PATH}")
print(f"Total Training Time: {total_training_time:.2f}s")