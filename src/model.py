import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import timm
from transformers import DistilBertModel
import src.config as config

class MultimodalPricePredictor(nn.Module):
    def __init__(self,
                 text_model_name=config.TEXT_MODEL_NAME,
                 image_model_name=config.IMAGE_MODEL_NAME,
                 pretrained=True):
        super().__init__()
        self.text_model = DistilBertModel.from_pretrained(text_model_name)
        self.image_model = timm.create_model(image_model_name, pretrained=pretrained, num_classes=0)
        text_features_dim = self.text_model.config.dim
        image_features_dim = self.image_model.num_features
        combined_features_dim = text_features_dim + image_features_dim + 1

        self.regressor = nn.Sequential(
            nn.BatchNorm1d(combined_features_dim),
            nn.Linear(combined_features_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4), 
            nn.Linear(512, 1)
        )

    def forward(self, input_ids, attention_mask, image, ipq):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_output.last_hidden_state[:, 0, :]
        image_features = self.image_model(image)
        combined_features = torch.cat([text_features, image_features, ipq.unsqueeze(1)], dim=1)
        log_price_prediction = self.regressor(combined_features)
        return log_price_prediction.squeeze(-1)