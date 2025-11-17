import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import Dataset
from PIL import Image
import re
import pandas as pd
import numpy as np

from transformers import DistilBertTokenizer
from torchvision import transforms
import src.config as config

class MultimodalDataset(Dataset):
    def __init__(self, csv_path, tokenizer, image_transform, image_dir=config.IMAGE_DIR, is_test_set=False):
        try:
            self.df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_path}", file=sys.stderr)
            self.df = pd.DataFrame()
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.image_dir = image_dir
        self.max_length = config.MAX_TEXT_LENGTH
        self.is_test_set = is_test_set

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['catalog_content']) if pd.notna(row['catalog_content']) else ''
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_length)
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        ipq = 1.0
        match = re.search(r"Item Pack Quantity:\s*(\d+)", text)
        if match:
            try:
                ipq = float(match.group(1))
            except ValueError:
                ipq = 1.0

        image_filename = f"{row['sample_id']}.jpg"
        image_path = os.path.join(self.image_dir, image_filename)
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE), (255, 255, 255))
        except Exception as e:
            print(f"Warning: Error loading image {image_path}. Using placeholder. Error: {e}", file=sys.stderr)
            image = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE), (255, 255, 255))
        image = self.image_transform(image)

        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image': image,
            'ipq': torch.tensor(ipq, dtype=torch.float32),
            'sample_id': str(row['sample_id'])
        }

        if not self.is_test_set and 'price' in row and pd.notna(row['price']):
            item['target'] = torch.tensor(np.log1p(row['price']), dtype=torch.float32)
        elif not self.is_test_set:
            item['target'] = torch.tensor(0.0, dtype=torch.float32)

        return item


image_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_tokenizer():
    return DistilBertTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
