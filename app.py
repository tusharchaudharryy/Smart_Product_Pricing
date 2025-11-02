import sys
import os
import io
import re
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

import torch
import torch.nn as nn
from PIL import Image
from transformers import DistilBertTokenizer, DistilBertModel
from torchvision import transforms
import timm

import src.config as config

app = FastAPI(title="Smart Product Price Predictor", description="A multimodal AI application to predict product prices.")
model_cache = {}

class MultimodalPricePredictor(nn.Module):
    def __init__(self,
                 text_model_name=config.TEXT_MODEL_NAME,
                 image_model_name='efficientnet_b0',
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
        combined_features = torch.cat([text_features, image_features, ipq], dim=1)
        log_price_prediction = self.regressor(combined_features)
        return log_price_prediction.squeeze(-1)

@app.on_event("startup")
def load_model():
    print("--- Loading model and preprocessors... ---")
    APP_MODEL_ARCH = 'efficientnet_b0'
    model_path = config.HYBRID_MODELS_FOR_FEATURES.get(APP_MODEL_ARCH)
    if not model_path:
        print(f"Error: 'HYBRID_MODELS_FOR_FEATURES['{APP_MODEL_ARCH}']' not set in config.py")
        model_cache["model"] = None
        return
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        model_cache["model"] = None
        return
    model_cache["tokenizer"] = DistilBertTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    model_cache["image_transform"] = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = MultimodalPricePredictor(
        image_model_name=APP_MODEL_ARCH,
        text_model_name=config.TEXT_MODEL_NAME
    )
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()
    model_cache["model"] = model
    print(f"--- Model '{model_path}' ({APP_MODEL_ARCH}) and preprocessors loaded successfully. ---")

def preprocess(image_bytes: bytes, text: str):
    tokenizer = model_cache["tokenizer"]
    image_transform = model_cache["image_transform"]
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = image_transform(image)
    except Exception as e:
        print(f"Error processing image: {e}")
        image = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE), (255, 255, 255))
        image_tensor = image_transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    text = str(text) if pd.notna(text) else ''
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=config.MAX_TEXT_LENGTH
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    ipq = 1.0
    match = re.search(r"Item Pack Quantity:\s*(\d+)", text)
    if match:
        try:
            ipq = float(match.group(1))
        except ValueError:
            ipq = 1.0
    ipq_tensor = torch.tensor([ipq], dtype=torch.float32).unsqueeze(0)
    input_ids = input_ids.to(config.DEVICE)
    attention_mask = attention_mask.to(config.DEVICE)
    image_tensor = image_tensor.to(config.DEVICE)
    ipq_tensor = ipq_tensor.to(config.DEVICE)
    return input_ids, attention_mask, image_tensor, ipq_tensor

def postprocess(log_price_tensor: torch.Tensor) -> str:
    price = np.expm1(log_price_tensor.detach().cpu().numpy())[0]
    price = max(0, price)
    return f"${price:,.2f}"

# HTML_CONTENT = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Smart Price Predictor</title>
#     <script src="https://cdn.tailwindcss.com"></script>
#     <style>
#         body { font-family: 'Inter', sans-serif; }
#         .spinner { border: 4px solid rgba(0, 0, 0, 0.1); width: 36px; height: 36px; border-radius: 50%; border-left-color: #09f; animation: spin 1s ease infinite; }
#         @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
#     </style>
# </head>
# <body class="bg-gray-100 min-h-screen flex items-center justify-center">
#     <div class="bg-white rounded-2xl shadow-2xl p-8 md:p-12 w-full max-w-2xl">
#         <h1 class="text-4xl font-bold text-gray-800 mb-2">Smart Price Predictor</h1>
#         <p class="text-gray-600 mb-8">Upload an image and enter a description to predict the price.</p>
#         <form id="predict-form" class="space-y-6">
#             <div>
#                 <label for="text-input" class="block text-sm font-medium text-gray-700 mb-2">Product Description</label>
#                 <textarea id="text-input" name="text" rows="4" class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition" placeholder="e.g., 'Brand new stainless steel watch, Item Pack Quantity: 1'"></textarea>
#             </div>
#             <div>
#                 <label for="image-input" class="block text-sm font-medium text-gray-700 mb-2">Product Image</label>
#                 <input id="image-input" name="image" type="file" accept="image/*" class="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 transition cursor-pointer"/>
#                 <img id="image-preview" src="" alt="Image preview" class="mt-4 rounded-lg shadow-sm hidden w-32 h-32 object-cover"/>
#             </div>
#             <button type="submit" class="w-full bg-blue-600 text-white font-bold py-3 px-6 rounded-lg shadow-lg hover:bg-blue-700 transition transform hover:-translate-y-0.5 flex items-center justify-center space-x-2">
#                 <span id="button-text">Predict Price</span>
#                 <div id="loading-spinner" class="spinner hidden"></div>
#             </button>
#         </form>
#         <div id="result-container" class="mt-8 p-6 bg-gray-50 rounded-lg text-center hidden">
#             <p class="text-lg text-gray-600">Predicted Price:</p>
#             <p id="result-price" class="text-5xl font-bold text-blue-600">$0.00</p>
#         </div>
#     </div>
#     <script>
#         const form = document.getElementById('predict-form');
#         const textInput = document.getElementById('text-input');
#         const imageInput = document.getElementById('image-input');
#         const imagePreview = document.getElementById('image-preview');
#         const resultContainer = document.getElementById('result-container');
#         const resultPrice = document.getElementById('result-price');
#         const buttonText = document.getElementById('button-text');
#         const loadingSpinner = document.getElementById('loading-spinner');
#         imageInput.addEventListener('change', () => {
#             const file = imageInput.files[0];
#             if (file) {
#                 const reader = new FileReader();
#                 reader.onload = (e) => {
#                     imagePreview.src = e.target.result;
#                     imagePreview.classList.remove('hidden');
#                 }
#                 reader.readAsDataURL(file);
#             } else {
#                 imagePreview.classList.add('hidden');
#             }
#         });
#         form.addEventListener('submit', async (e) => {
#             e.preventDefault();
#             if (!imageInput.files[0] || !textInput.value) {
#                 alert('Please provide both an image and a description.');
#                 return;
#             }
#             buttonText.classList.add('hidden');
#             loadingSpinner.classList.remove('hidden');
#             resultContainer.classList.add('hidden');
#             const formData = new FormData();
#             formData.append('text', textInput.value);
#             formData.append('image', imageInput.files[0]);
#             try {
#                 const response = await fetch('/predict', { method: 'POST', body: formData });
#                 if (!response.ok) {
#                     const err = await response.json();
#                     throw new Error(`HTTP error! status: ${response.status}, message: ${err.detail || 'Unknown error'}`);
#                 }
#                 const data = await response.json();
#                 resultPrice.textContent = data.predicted_price;
#                 resultContainer.classList.remove('hidden');
#             } catch (error) {
#                 console.error('Error predicting price:', error);
#                 alert('An error occurred while predicting the price. Please check the console.');
#             } finally {
#                 buttonText.classList.remove('hidden');
#                 loadingSpinner.classList.add('hidden');
#             }
#         });
#     </script>
# </body>
# </html>
# """

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Smart Price Predictor</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body {
      font-family: 'Inter', sans-serif;
      background: linear-gradient(135deg, #667eea, #764ba2);
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #222;
    }
    .glass {
      background: rgba(255, 255, 255, 0.15);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 255, 255, 0.2);
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    }
    .spinner {
      border: 4px solid rgba(0, 0, 0, 0.1);
      width: 28px;
      height: 28px;
      border-radius: 50%;
      border-left-color: #fff;
      animation: spin 1s ease infinite;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    .fade-in {
      animation: fadeIn 0.8s ease-in-out forwards;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <div class="glass rounded-3xl p-10 md:p-14 w-full max-w-2xl text-center text-white">
    <h1 class="text-4xl font-extrabold mb-3 tracking-tight drop-shadow"> Smart Price Predictor</h1>
    <p class="text-lg text-gray-200 mb-8">Upload your product image and description to predict its price.</p>
    
    <form id="predict-form" class="space-y-6">
      <div>
        <label for="text-input" class="block text-sm font-semibold text-gray-100 mb-2 text-left">Product Description</label>
        <textarea id="text-input" name="text" rows="4"
          class="w-full p-4 rounded-xl text-gray-800 focus:ring-4 focus:ring-blue-400 outline-none transition border border-transparent shadow-sm"
          placeholder="e.g., 'Stainless steel wristwatch, Item Pack Quantity: 1'"></textarea>
      </div>
      
      <div>
        <label for="image-input" class="block text-sm font-semibold text-gray-100 mb-2 text-left">Product Image</label>
        <input id="image-input" name="image" type="file" accept="image/*"
          class="w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4
          file:rounded-lg file:border-0 file:text-sm file:font-semibold
          file:bg-white/20 file:text-white hover:file:bg-white/30 cursor-pointer transition" />
        <img id="image-preview" src="" alt="Image preview"
          class="mt-4 rounded-xl shadow-md hidden w-32 h-32 object-cover mx-auto hover:scale-105 transition-transform duration-300"/>
      </div>

      <button type="submit"
        class="w-full bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-bold py-3 px-6 rounded-xl shadow-lg transition transform hover:-translate-y-0.5 flex items-center justify-center space-x-2">
        <span id="button-text"> Predict Price</span>
        <div id="loading-spinner" class="spinner hidden"></div>
      </button>
    </form>

    <div id="result-container" class="mt-10 p-6 bg-white/10 rounded-2xl hidden fade-in">
      <p class="text-md text-gray-200">Predicted Price:</p>
      <p id="result-price" class="text-5xl font-extrabold text-white mt-2">$0.00</p>
    </div>
  </div>

  <script>
    const form = document.getElementById('predict-form');
    const textInput = document.getElementById('text-input');
    const imageInput = document.getElementById('image-input');
    const imagePreview = document.getElementById('image-preview');
    const resultContainer = document.getElementById('result-container');
    const resultPrice = document.getElementById('result-price');
    const buttonText = document.getElementById('button-text');
    const loadingSpinner = document.getElementById('loading-spinner');

    imageInput.addEventListener('change', () => {
      const file = imageInput.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          imagePreview.src = e.target.result;
          imagePreview.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
      } else {
        imagePreview.classList.add('hidden');
      }
    });

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (!imageInput.files[0] || !textInput.value) {
        alert('Please provide both an image and a description.');
        return;
      }
      buttonText.classList.add('hidden');
      loadingSpinner.classList.remove('hidden');
      resultContainer.classList.add('hidden');

      const formData = new FormData();
      formData.append('text', textInput.value);
      formData.append('image', imageInput.files[0]);

      try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        if (!response.ok) {
          const err = await response.json();
          throw new Error(`HTTP error! status: ${response.status}, message: ${err.detail || 'Unknown error'}`);
        }
        const data = await response.json();
        resultPrice.textContent = data.predicted_price;
        resultContainer.classList.remove('hidden');
        resultContainer.classList.add('fade-in');
      } catch (error) {
        console.error('Error predicting price:', error);
        alert('An error occurred while predicting the price. Please check the console.');
      } finally {
        buttonText.classList.remove('hidden');
        loadingSpinner.classList.add('hidden');
      }
    });
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_homepage(request: Request):
    return HTMLResponse(content=HTML_CONTENT, status_code=200)

@app.post("/predict")
async def predict_price(
    text: str = Form(...),
    image: UploadFile = File(...)
):
    if not model_cache.get("model"):
        return JSONResponse(
            status_code=503,
            content={"error": "Model is not loaded. Please check server logs."}
        )
    try:
        image_bytes = await image.read()
        input_ids, attention_mask, image_tensor, ipq_tensor = preprocess(image_bytes, text)
        with torch.no_grad():
            log_price_output = model_cache["model"](
                input_ids=input_ids,
                attention_mask=attention_mask,
                image=image_tensor,
                ipq=ipq_tensor
            )
        formatted_price = postprocess(log_price_output)
        return JSONResponse(content={"predicted_price": formatted_price})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "An internal server error occurred.", "detail": str(e)}
        )

if __name__ == "__main__":
    print("--- Starting FastAPI Server ---")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

# http://127.0.0.1:8000