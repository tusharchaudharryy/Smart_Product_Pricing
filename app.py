import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
import torch
import torch.nn as nn
import timm
from transformers import DistilBertModel, DistilBertTokenizer
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import re
import uvicorn
import joblib 
import src.config as config 

TEXT_MODEL_NAME = config.TEXT_MODEL_NAME
IMAGE_MODEL_NAME = 'efficientnet_b0' 
IMAGE_SIZE = config.IMAGE_SIZE
MAX_TEXT_LENGTH = config.MAX_TEXT_LENGTH
DEVICE = config.DEVICE

PYTORCH_MODEL_PATH = config.HYBRID_MODELS_FOR_FEATURES['efficientnet_b0']

K_FOLD_MODELS = [
    os.path.join(config.MODEL_DIR, f'hybrid_lgbm_model_fold_{i+1}.joblib')
    for i in range(config.HYBRID_N_SPLITS)
]

class MultimodalPricePredictor(nn.Module):
    def __init__(self,
                 text_model_name=TEXT_MODEL_NAME,
                 image_model_name=IMAGE_MODEL_NAME,
                 pretrained=False):
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

    def extract_features(self, input_ids, attention_mask, image):
        """
        A new method to only extract the deep features needed by LightGBM.
        """
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_output.last_hidden_state[:, 0, :]
        image_features = self.image_model(image)
        combined_features = torch.cat([text_features, image_features], dim=1)
        return combined_features

app_state = {
    "tokenizer": None,
    "pytorch_feature_extractor": None,
    "kfold_lgbm_models": []
}

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0) 

def preprocess_text(text: str, tokenizer):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=MAX_TEXT_LENGTH
    )
    return inputs['input_ids'], inputs['attention_mask']

def extract_ipq_feature(text: str) -> np.ndarray:
    ipq = 1.0
    match = re.search(r"Item Pack Quantity:\s*(\d+)", text)
    if match:
        try: ipq = float(match.group(1))
        except ValueError: ipq = 1.0
    return np.array([[ipq]]) 

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Loading all models for Hybrid K-Fold App... ---")
    try:
        app_state['tokenizer'] = DistilBertTokenizer.from_pretrained(TEXT_MODEL_NAME)
        
        if not os.path.exists(PYTORCH_MODEL_PATH):
            raise FileNotFoundError(f"PyTorch feature extractor not found at {PYTORCH_MODEL_PATH}")
        
        pytorch_model = MultimodalPricePredictor(image_model_name=IMAGE_MODEL_NAME)
        pytorch_model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=DEVICE))
        pytorch_model.to(DEVICE)
        pytorch_model.eval()
        app_state['pytorch_feature_extractor'] = pytorch_model
        print(f"Loaded PyTorch feature extractor: {PYTORCH_MODEL_PATH}")

        lgbm_models = []
        for model_path in K_FOLD_MODELS:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"K-Fold model not found at {model_path}")
            model = joblib.load(model_path)
            lgbm_models.append(model)
            print(f"Loaded K-Fold model: {model_path}")
        
        app_state['kfold_lgbm_models'] = lgbm_models
        
        print(f"--- All {len(lgbm_models) + 1} models and preprocessors loaded successfully. ---")
    
    except Exception as e:
        print(f"FATAL ERROR during model loading: {e}", file=sys.stderr)
        raise e 
        
    yield 
    
    print("--- Shutting down and cleaning up... ---")
    app_state.clear()

app = FastAPI(lifespan=lifespan)

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Price Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { 
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .loader {
            border: 4px solid #f3f3f3; border-top: 4px solid #4f46e5;
            border-radius: 50%; width: 40px; height: 40px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        #image-preview {
            max-height: 200px;
            width: auto;
            margin: 0 auto;
            border-radius: 0.5rem;
        }
        .dragover {
            border-color: #4f46e5;
            background-color: #f4f4f5;
        }
    </style>
</head>
<body class="flex items-center justify-center min-h-screen p-4">
    <div class="bg-white p-8 rounded-2xl shadow-xl w-full max-w-md">
        
        <div class="text-center mb-8">
            <svg class="mx-auto h-12 w-12 text-indigo-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" d="M12 6v12m-3-2.818.879.879A3 3 0 1 0 12 15h0a3 3 0 0 0 .879-5.901L12 12m0 0L8.12 8.121A3 3 0 0 0 12.001 12h0a3 3 0 0 0-3.879-2.879L12 12Zm0 0a3 3 0 0 0 3.879 2.879l-3.879-3.879A3 3 0 0 0 12 12h0a3 3 0 0 0 3.879-2.879L12 12Z" />
            </svg>
            <h1 class="text-3xl font-bold text-gray-800 mt-2">Smart Price Predictor</h1>
            <p class="text-sm text-gray-500">Using Hybrid K-Fold LGBM Model</p>
        </div>
        
        <form id="price-form" class="space-y-6">
            <div>
                <label for="text_input" class="block text-sm font-medium text-gray-700">Product Description</label>
                <textarea id="text_input" name="text_input" rows="4" class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm transition duration-150" placeholder="e.g., Men's Regular Fit Polo Shirt... Item Pack Quantity: 1" required></textarea>
            </div>
            
            <div>
                <label for="image_input" class="block text-sm font-medium text-gray-700">Product Image</label>
                <div id="image-drop-area" class="mt-1 flex flex-col items-center justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md transition duration-150">
                    <div id="upload-prompt" class="space-y-1 text-center">
                        <svg class="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
                            <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path>
                        </svg>
                        <div class="flex text-sm text-gray-600">
                            <label for="image_file" class="relative cursor-pointer bg-white rounded-md font-medium text-indigo-600 hover:text-indigo-500 focus-within:outline-none">
                                <span>Upload a file</span>
                                <input id="image_file" name="image_file" type="file" class="sr-only" accept="image/jpeg, image/png">
                            </label>
                            <p class="pl-1">or drag and drop</p>
                        </div>
                        <p class="text-xs text-gray-500" id="file-name">PNG, JPG up to 10MB</p>
                    </div>
                    <img id="image-preview" src="" alt="Image preview" class="hidden mt-4"/>
                </div>
            </div>
            
            <button type="submit" id="predict-button" class="w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition duration-150 transform hover:scale-105">
                Predict Price
            </button>
        </form>
        
        <!-- Result/Loading Area -->
        <div id="result-area" class="mt-6 text-center" style="display: none;">
            <div id="loader" class="loader mx-auto"></div>
            <div id="prediction" class="p-6 bg-green-50 border border-green-200 rounded-lg" style="display: none;">
                <p class="text-md font-medium text-green-800">Predicted Price:</p>
                <p id="price-value" class="text-4xl font-bold text-green-700 mt-2"></p>
            </div>
            <div id="error-msg" class="p-4 bg-red-100 border border-red-300 text-red-700 rounded-lg" style="display: none;"></div>
        </div>
    </div>
    
    <script>
        const form = document.getElementById('price-form');
        const button = document.getElementById('predict-button');
        const resultArea = document.getElementById('result-area');
        const loader = document.getElementById('loader');
        const prediction = document.getElementById('prediction');
        const priceValue = document.getElementById('price-value');
        const errorMsg = document.getElementById('error-msg');
        const imageInput = document.getElementById('image_file');
        const dropArea = document.getElementById('image-drop-area');
        const fileNameDisplay = document.getElementById('file-name');
        const uploadPrompt = document.getElementById('upload-prompt');
        const imagePreview = document.getElementById('image-preview');
        let file;

        imageInput.addEventListener('change', (e) => handleFile(e.target.files[0]));
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
        });
        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
        });
        dropArea.addEventListener('drop', (e) => handleFile(e.dataTransfer.files[0]), false);

        function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

        function handleFile(selectedFile) {
            if (selectedFile && selectedFile.type.startsWith('image/')) {
                file = selectedFile;
                fileNameDisplay.textContent = file.name;
                fileNameDisplay.classList.add('font-medium', 'text-gray-900');
                
                // Show image preview
                const reader = new FileReader();
                reader.onload = (e) => {
                    imagePreview.src = e.target.result;
                    imagePreview.classList.remove('hidden');
                    uploadPrompt.classList.add('hidden'); // Hide the upload prompt
                };
                reader.readAsDataURL(file);
            } else {
                file = null;
                fileNameDisplay.textContent = 'PNG, JPG up to 10MB';
                fileNameDisplay.classList.remove('font-medium', 'text-gray-900');
                imagePreview.classList.add('hidden');
                uploadPrompt.classList.remove('hidden');
                showError('Invalid file type. Please upload a JPG or PNG image.');
            }
        }

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const text = document.getElementById('text_input').value;
            if (!file || !text) { showError('Please provide both a description and an image.'); return; }
            showLoading();
            
            const formData = new FormData();
            formData.append('text', text);
            formData.append('image', file);
            
            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();
                if (response.ok) { showResult(data.predicted_price); } else { showError(data.detail || 'An unknown error occurred.'); }
            } catch (err) { console.error('Fetch error:', err); showError('Failed to connect to the server. Is it running?'); }
        });

        function showLoading() {
            resultArea.style.display = 'block';
            loader.style.display = 'block';
            prediction.style.display = 'none';
            errorMsg.style.display = 'none';
            button.disabled = true;
            button.classList.add('opacity-50', 'cursor-not-allowed');
        }
        function showResult(price) {
            loader.style.display = 'none';
            errorMsg.style.display = 'none';
            prediction.style.display = 'block';
            priceValue.textContent = `$${price.toFixed(2)}`;
            button.disabled = false;
            button.classList.remove('opacity-50', 'cursor-not-allowed');
        }
        function showError(message) {
            loader.style.display = 'none';
            prediction.style.display = 'none';
            resultArea.style.display = 'block';
            errorMsg.style.display = 'block';
            errorMsg.textContent = `Error: ${message}`;
            button.disabled = false;
            button.classList.remove('opacity-50', 'cursor-not-allowed');
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_root():
    """Serves the main HTML page."""
    return HTML_CONTENT

@app.post("/predict")
async def predict(text: str = Form(...), image: UploadFile = File(...)):
    """Handles the prediction request."""
    if 'pytorch_feature_extractor' not in app_state or not app_state['kfold_lgbm_models']:
        raise HTTPException(status_code=503, detail="Models are not loaded or are still loading. Please try again.")

    try:
        image_bytes = await image.read()
        
        tokenizer = app_state['tokenizer']
        input_ids, attention_mask = preprocess_text(text, tokenizer)
        image_tensor = preprocess_image(image_bytes)
        ipq_feature = extract_ipq_feature(text) 
        
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        image_tensor = image_tensor.to(DEVICE)

        pytorch_model = app_state['pytorch_feature_extractor']
        with torch.no_grad():
            deep_features_tensor = pytorch_model.extract_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image=image_tensor
            )
        deep_features = deep_features_tensor.cpu().numpy() 
        
        final_feature_vector = np.concatenate([deep_features, ipq_feature], axis=1)

        kfold_models = app_state['kfold_lgbm_models']
        all_preds_log = []
        for lgbm_model in kfold_models:
            pred_log = lgbm_model.predict(final_feature_vector)[0]
            all_preds_log.append(pred_log)
        
        avg_log_price = np.mean(all_preds_log)
        
        predicted_price = np.expm1(avg_log_price)
        
        return JSONResponse(content={"predicted_price": float(predicted_price)})

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    print("Starting FastAPI server with uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

# http://127.0.0.1:8000