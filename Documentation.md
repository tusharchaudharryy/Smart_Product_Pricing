# Multimodal Product Price Prediction

## 1. Project Overview (Week 1)

### 1.1. Problem Definition

This project tackles the "Smart Product Price Prediction" challenge. The primary goal is to develop a machine learning model that can accurately predict the price of an e-commerce product by leveraging multimodal data.

**Input:** A product's textual description (`catalog_content`) and its associated product image (`image_link`).  
**Output:** A predicted price (float).

**Hypothesis:** A model that can process both text and images simultaneously (a multimodal model) will outperform models that rely on only one data type (unimodal), as product value is communicated through both its description and its visual presentation.

### 1.2. Dataset

The project uses the **ML Challenge 2025** dataset, which contains 75,000 labeled product samples.

To ensure a robust and fair evaluation, the original `train.csv` (75k samples with prices) was split into two new sets using an 80/20 ratio (`random_state=42`):

- **project_train.csv:** (60,000 samples) Used for all model training and validation.  
- **project_test.csv:** (15,000 samples) A held-out test set, used only at the very end to generate the final performance metrics for all models.

### 1.3. Evaluation Metrics

To provide a comprehensive analysis of model performance, we used four distinct metrics:

- **SMAPE (Primary Metric):** Symmetric Mean Absolute Percentage Error. This was our main metric as it measures the relative error, which is crucial for a target (price) that spans many orders of magnitude.
- **MAE (Mean Absolute Error):** Measures the average absolute dollar amount our predictions are off by.
- **RMSE (Root Mean Squared Error):** Similar to MAE, but heavily penalizes large errors.
- **R-squared (R²):** Measures the proportion of the price variance that the model can explain.

---

## 2. Methodology & Model Architectures

We implemented three distinct models to systematically evaluate different approaches.

### 2.1. Model 1: Baseline (Text-Only LGBM) - Week 3

This model serves as our non-trivial baseline to determine the predictive power of text features alone.

**Features:**

- **TF-IDF:** A 50,000-feature sparse matrix generated from the catalog_content (using 1- and 2-ngrams).
- **IPQ:** The "Item Pack Quantity" (a number) was extracted from the text using regular expressions.

**Algorithm:** A LightGBM (LGBM) regressor. This is a fast and powerful gradient boosting model that works well on sparse data.

---

### 2.2. Model 2: Main Multimodal Model (PyTorch) - Week 2

This is our main end-to-end deep learning model, built to learn from text, images, and the IPQ feature simultaneously.

**Architecture (MultimodalPricePredictor):**

- **Text Branch:** A pre-trained `distilbert-base-uncased` transformer to read the text and extract a 768-dim feature vector.
- **Image Branch:** A pre-trained `efficientnet_b0` (a powerful CNN) to see the product image and extract a 1280-dim feature vector.
- **IPQ Feature:** The 1-dim IPQ feature.
- **Fusion:** These three inputs are concatenated into a single 2049-dim vector.
- **MLP Head:** This vector is passed through an MLP (Linear -> ReLU -> Dropout -> Linear) to regress the final log-price prediction.

**Training:** The model was trained to minimize `MSELoss` (on the log-price) using an AdamW optimizer and early stopping (`patience=3`) based on the validation SMAPE score. Hyperparameters like the learning rate (`5e-6`) were selected manually based on initial experiments to ensure stability.

---

### 2.3. Model 3: Hybrid K-Fold LGBM (Our Best Model)

This advanced model combines the strengths of the previous two approaches in a two-stage process.

**Stage 1: Deep Feature Extraction**  
We used our trained PyTorch model (Model 2) as a "feature extractor." We fed all 75,000 products through it and saved the 2048-dim internal feature vectors (text + image) to disk.

**Stage 2: K-Fold Gradient Boosting**  
Features: The final training data consisted of the 2048 deep features combined with the 1-dim IPQ feature (total: 2049 features).

**Algorithm:** We trained a new LightGBM model on these rich, deep features.

**Hyperparameter Tuning (Optuna):** To find the best possible version of this model, we used the Optuna framework to perform an automated 10-trial hyperparameter search. This search optimized key parameters like `learning_rate`, `num_leaves`, `n_estimators`, and `max_depth` to minimize the validation SMAPE. This step was critical for achieving our best score.

**K-Fold Cross-Validation:** To maximize robustness, we used the best hyperparameters from Optuna and trained the final model using **5-Fold Cross-Validation**. The final prediction for any product is the average prediction from all 5 models.

---

## 3. Final Results & Analysis (Week 4)

All three models were evaluated on the held-out `project_test.csv` (15,000 samples). The results, achieved after the hyperparameter tuning phase, clearly show the superiority of the hybrid approach.

| Model | SMAPE (↓) | MAE (↓) | RMSE (↓) | R² (↑) |
|--------|------------|-----------|-----------|-----------|
| **Hybrid (K-Fold + Deep Features)** | **49.68%** | **11.09** | **32.38** | **0.3053** |
| Baseline (Text-Only LGBM) | 51.68% | 11.71 | 34.25 | 0.2229 |
| Main Model (End-to-End PyTorch) | 52.76% | 11.95 | 32.91 | 0.2824 |

### 3.1. Analysis

**Hybrid Model is the Winner:**  
The Hybrid K-Fold LGBM model, optimized with Optuna, was the best performer across all four metrics. This validates our hypothesis that combining deep multimodal features with a powerful, well-tuned gradient boosting strategy is the optimal approach for this problem.

**Image Features Matter:**  
The Hybrid model (SMAPE 49.68%) significantly outperformed the Baseline text-only model (SMAPE 51.68%). This 2-point improvement confirms that the visual information from the product images provides essential predictive value that text alone misses.

**Baseline vs. End-to-End:**  
Our simple text-only Baseline (51.68%) surprisingly outperformed the complex end-to-end PyTorch model (52.76%) on the primary SMAPE metric. This suggests that while the PyTorch model was good at learning (as shown by its strong RMSE/R² scores), it was difficult to train perfectly, and the simpler LGBM was more effective at modeling the sparse TF-IDF data directly.

---

### 3.2. Error Analysis & Future Work

The Hybrid model's errors were highest on very cheap (< $5) or very expensive luxury/collectible items, where price is driven more by brand value than physical features.

**Future work to break the 40% SMAPE barrier would focus on:**

- **Categorical Feature Engineering:** Extracting "Brand Name" and "Product Category" from the text to feed directly into the LGBM model.
- **Model Stacking:** Using the 5 K-Fold models' predictions as inputs to a final "meta-model" to learn the optimal blend.

---

## 4. FastAPI Application

As a final deliverable, we built an interactive web application using **FastAPI**.

**Function:** The app loads our trained `efficientnet_b0` PyTorch model.  
**Interface:** It serves a simple HTML frontend that allows any user to:

- Upload a product image.
- Type a product description (including IPQ).
- Click "Predict Price".

**Result:** The application sends the data to a `/predict` endpoint, processes the inputs with the model, and displays the final predicted price to the user in real-time.  
This successfully demonstrates how the model can be deployed for practical, real-world use.

