# Yelp Star Rating Prediction with Deep Learning

A comprehensive guide to replicate the EC349 assignment project in Python with Apple GPU acceleration. This project builds a neural network to predict user star ratings on the Yelp Academic dataset using features engineered from business, user, and review data, with transformer-based sentiment analysis.

## Table of Contents

- [Project Overview](#project-overview)
- [Prerequisites & Environment Setup](#prerequisites--environment-setup)
- [Data Pipeline](#data-pipeline)
- [Feature Engineering](#feature-engineering)
- [Sentiment Analysis](#sentiment-analysis)
- [Feature Selection](#feature-selection)
- [Model Architecture](#model-architecture)
- [Training & Evaluation](#training--evaluation)
- [Key Challenges & Solutions](#key-challenges--solutions)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

This project aims to predict Yelp review star ratings (1-5) using a combination of:

- **User features**: average stars, review count, elite status history, time on platform
- **Business features**: average stars, review count
- **Review features**: sentiment score from DistilBERT transformer model

The final model is a sequential neural network with batch normalization and dropout layers, trained with early stopping to prevent overfitting.

### Key Statistics
- **Target variable**: Star ratings (1-5)
- **Input features**: 5 (after RFE)
- **Network architecture**: Dense(256) → BN → Dense(128) → Dense(1)
- **Training samples**: ~130,000 (stratified sampling)
- **Test MSE**: 0.8115

### Data Distribution Challenge
The original dataset has severe class imbalance (~60% of reviews are 4-5 stars). The project addresses this through **stratified sampling**, creating equal representation of each star rating. This improves model learning across the full rating spectrum at the cost of not being perfectly representative of the population.

---

## Prerequisites & Environment Setup

### System Requirements
- **macOS** with Apple Silicon (M1/M2/M3) or Intel
- **RAM**: 16GB minimum (32GB recommended for full dataset processing)
- **Disk space**: ~15GB for Yelp Academic dataset + outputs
- **Python**: 3.10+

### Installing with Apple GPU Support

#### Option 1: Using Conda (Recommended)

```bash
# Create a conda environment with PyTorch Apple GPU support
conda create -n yelp-py python=3.10

conda activate yelp-py

# Install PyTorch with Metal acceleration (Apple GPU)
# Check https://pytorch.org for latest command
conda install pytorch::pytorch torchvision torchaudio -c pytorch

# Install core dependencies
pip install tensorflow-macos tensorflow-metal  # For Keras/TensorFlow
# OR use PyTorch Lightning instead (preferred for Apple GPU)
pip install pytorch-lightning
```

#### Option 2: Using pip with venv

```bash
python3 -m venv yelp-env
source yelp-env/bin/activate

pip install --upgrade pip

# PyTorch with Metal acceleration
pip install torch torchvision torchaudio

# For transformer models
pip install transformers[torch]

# Data processing
pip install pandas numpy scikit-learn

# Neural networks
pip install pytorch-lightning

# Sentiment analysis
pip install torch-transformers
```

### Core Dependencies

```
# Core ML/Data
torch>=2.0          # PyTorch (Metal acceleration enabled by default on Apple Silicon)
transformers>=4.30  # Hugging Face transformers (DistilBERT)
scikit-learn>=1.3   # Preprocessing, RFE
pandas>=1.5         # Data manipulation
numpy>=1.23         # Numerical computing
pytorch-lightning   # Training framework (handles GPU/CPU transparently)

# Data I/O
pyarrow>=12.0       # Parquet support
duckdb>=0.8         # SQL queries on data

# Utilities
python-dotenv       # Configuration
tqdm                # Progress bars
```

### Verify Apple GPU Support

```python
import torch
print(torch.backends.mps.is_available())  # Should print: True
print(torch.backends.mps.is_built())      # Should print: True

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))  # For TensorFlow/Keras
```

---

## Data Pipeline

### 1. Obtaining the Data

The project uses pre-converted CSV files from the `data/` directory. These CSV files are ready to use and have been converted from the original Yelp Academic Dataset JSON format.

Required files:
- `data/yelp_business_data.csv`
- `data/yelp_review.csv`
- `data/yelp_user.csv`
- `data/yelp_checkin_data.csv` (optional)
- `data/yelp_tip_data.csv` (optional)

### 2. Data Loading & Preprocessing

The project uses CSV files that are ready to load. Load them directly using pandas:

```python
import pandas as pd
from datetime import datetime

# Load datasets from CSV files
business_df = pd.read_csv('data/yelp_business_data.csv')
review_df = pd.read_csv('data/yelp_review.csv')
user_df = pd.read_csv('data/yelp_user.csv')

print(f"Business records: {len(business_df)}")
print(f"Review records: {len(review_df)}")
print(f"User records: {len(user_df)}")
```

### 3. Data Merging & Cleaning

```python
# Rename columns for clarity
user_df = user_df.rename(columns={
    'useful': 'total_useful',
    'funny': 'total_funny',
    'cool': 'total_cool',
    'review_count': 'user_review_count',
    'name': 'user_name',
    'average_stars': 'user_average_stars'
})

business_df = business_df.rename(columns={
    'stars': 'business_average_stars',
    'review_count': 'business_review_count',
    'name': 'business_name'
})

# Convert date columns
review_df['date'] = pd.to_datetime(review_df['date'])
user_df['yelping_since'] = pd.to_datetime(user_df['yelping_since'])

# Inner join: review (core) → user → business
final_data = review_df.merge(user_df, on='user_id', how='inner')
final_data = final_data.merge(business_df, on='business_id', how='inner')

# Remove rows with missing critical values
final_data = final_data.dropna(subset=['stars', 'text', 'business_average_stars', 
                                        'user_average_stars', 'user_review_count'])

print(f"Final merged dataset: {len(final_data)} rows")
final_data.to_csv('merged_data.csv', index=False)
```

---

## Feature Engineering

### Engineered Features

#### 1. Time on Platform (Weeks)
```python
final_data['time_yelping'] = (final_data['date'] - final_data['yelping_since']).dt.total_seconds() / (7 * 24 * 3600)
```

#### 2. Elite Status Features

Parse the elite years from the pipe-separated elite field:

```python
def parse_elite_years(elite_str):
    """Convert elite string to list of years"""
    if pd.isna(elite_str) or elite_str == '':
        return []
    return [int(y) for y in str(elite_str).split(',') if y.strip()]

def count_elite_statuses(elite_str, review_year):
    """Count total elite statuses up to review year"""
    elite_years = parse_elite_years(elite_str)
    return sum(1 for year in elite_years if year <= review_year)

def check_elite_status(elite_str, review_year):
    """Check if user had elite status in review year or prior"""
    elite_years = parse_elite_years(elite_str)
    return int(review_year in elite_years or (review_year - 1) in elite_years)

# Extract review year
final_data['date_year'] = final_data['date'].dt.year

# Apply elite status features
final_data['total_elite_statuses'] = final_data.apply(
    lambda row: count_elite_statuses(row['elite'], row['date_year']), axis=1
)
final_data['elite_status'] = final_data.apply(
    lambda row: check_elite_status(row['elite'], row['date_year']), axis=1
)
```

#### 3. Cumulative User Statistics (Optional)
```python
# Sort by user and date for cumulative calculations
final_data = final_data.sort_values(['user_id', 'date']).reset_index(drop=True)

final_data['cumulative_stars'] = final_data.groupby('user_id')['stars'].cumsum()
```

### Handling Missing Values

```python
# Check for missing values in engineered features
print(final_data[['time_yelping', 'total_elite_statuses', 'elite_status']].isnull().sum())

# For any remaining nulls, use forward fill or median imputation
final_data['time_yelping'].fillna(final_data['time_yelping'].median(), inplace=True)
final_data['total_elite_statuses'].fillna(0, inplace=True)
final_data['elite_status'].fillna(0, inplace=True)

# Save intermediate result
final_data.to_csv('featured_data.csv', index=False)
```

---

## Sentiment Analysis

### Transformer-Based Sentiment Analysis with DistilBERT

The original project uses the R `text` package with a Python backend. In Python, use Hugging Face `transformers` directly with Metal GPU acceleration.

### Key Challenge Addressed
The original DistilBERT model has a 512-token maximum sequence length. The R implementation used crude truncation. For a more robust approach in Python:

**Option 1: Smart Truncation (Keep Semantically Important Parts)**

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
MAX_TOKENS = 512

def smart_truncate_text(text, max_tokens=MAX_TOKENS):
    """
    Truncate text intelligently by keeping the middle portion 
    (often contains review body) rather than just the first N chars
    """
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= max_tokens:
        return text
    
    # Keep first 256 and last 256 tokens for important context
    truncated_tokens = tokens[:256] + tokens[-256:]
    return tokenizer.convert_tokens_to_string(truncated_tokens)

final_data['text_truncated'] = final_data['text'].apply(smart_truncate_text)
```

**Option 2: Batch Sentiment Analysis with GPU**

```python
import torch
from transformers import pipeline
from tqdm import tqdm

# Initialize pipeline with Metal GPU support
device = "mps" if torch.backends.mps.is_available() else "cpu"
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if device == "cuda" else -1,  # -1 for CPU, 0+ for GPU
    batch_size=32
)

# Process reviews in batches
batch_size = 64
sentiments = []

for i in tqdm(range(0, len(final_data), batch_size), desc="Sentiment Analysis"):
    batch_texts = final_data['text_truncated'].iloc[i:i+batch_size].tolist()
    
    results = sentiment_pipeline(batch_texts)
    sentiments.extend(results)

# Extract sentiment scores and labels
final_data['sentiment_label'] = [s['label'] for s in sentiments]
final_data['sentiment_score_raw'] = [s['score'] for s in sentiments]

# Convert to signed score: negative = -score, positive = +score
final_data['normalized_sentiment_score'] = final_data.apply(
    lambda row: -row['sentiment_score_raw'] if row['sentiment_label'] == 'NEGATIVE' else row['sentiment_score_raw'],
    axis=1
)

# Save sentiment results
final_data.to_csv('sentiment_data.csv', index=False)
```

### Estimated Processing Time
- ~130,000 reviews with Apple GPU: **2-4 hours**
- Without GPU: **12-24 hours**
- Use batching and `tqdm` to monitor progress

---

## Feature Selection

### Recursive Feature Elimination (RFE)

The project uses RFE with Random Forest to identify optimal feature subsets.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
import pandas as pd

# Load sentiment data
data = pd.read_csv('sentiment_data.csv')

# Define candidate features for RFE
candidate_features = [
    'business_average_stars',
    'user_average_stars',
    'user_review_count',
    'total_elite_statuses',
    'time_yelping',
    'elite_status',
    'normalized_sentiment_score'
]

# Remove rows with missing values in these columns
data_clean = data.dropna(subset=candidate_features + ['stars'])

X = data_clean[candidate_features]
y = data_clean['stars']

print(f"Cleaned data shape: {X.shape}")

# RFE with Random Forest (5-fold CV for speed)
estimator = RandomForestRegressor(n_estimators=100, random_state=1, n_jobs=-1)

rfe = RFE(estimator=estimator, n_features_to_select=None, step=1)
rfe.fit(X, y)

# Get rankings
feature_ranking = pd.DataFrame({
    'Feature': candidate_features,
    'Ranking': rfe.ranking_,
    'Selected': rfe.support_
})
feature_ranking = feature_ranking.sort_values('Ranking')
print("\nRFE Results:")
print(feature_ranking)

# Extract optimal features
optimal_features = feature_ranking[feature_ranking['Selected']]['Feature'].tolist()
print(f"\nOptimal features selected: {optimal_features}")

# Prepare final dataset
final_model_data = data_clean[optimal_features + ['stars']].copy()
final_model_data.to_csv('final_model_data.csv', index=False)

# Save feature list for model training
import json
with open('optimal_features.json', 'w') as f:
    json.dump(optimal_features, f)
```

### Expected Output
Based on the original project:
```
Optimal features: ['user_average_stars', 'business_average_stars', 
                   'normalized_sentiment_score', 'user_review_count', 
                   'business_review_count']
```

---

## Model Architecture

### Sequential Neural Network

The model uses a classical feedforward architecture with regularization:

```
Input (5 features)
    ↓
Dense(256, ReLU)
    ↓
Batch Normalization
    ↓
Dropout(0.5)
    ↓
Dense(128, ReLU)
    ↓
Dropout(0.5)
    ↓
Dense(1, Linear)  ← Regression output
    ↓
Output (predicted star rating)
```

### Implementation in PyTorch

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau

class YelpRatingPredictor(pl.LightningModule):
    def __init__(self, input_size=5, learning_rate=0.0001):
        super().__init__()
        self.save_hyperparameters()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(128, 1)
        )
        
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x):
        return self.network(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x).squeeze()
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x).squeeze()
        loss = self.loss_fn(y_pred, y)
        mae = torch.mean(torch.abs(y_pred - y))
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mae', mae)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x).squeeze()
        loss = self.loss_fn(y_pred, y)
        mae = torch.mean(torch.abs(y_pred - y))
        self.log('test_loss', loss)
        self.log('test_mae', mae)
        return {'loss': loss, 'mae': mae}
    
    def configure_optimizers(self):
        optimizer = RMSprop(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
```

### Alternative: TensorFlow/Keras Implementation

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(5,)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    
    layers.Dense(1)
])

model.compile(
    loss='mse',
    optimizer=keras.optimizers.RMSprop(learning_rate=0.0001),
    metrics=['mae']
)

# Enable Apple GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs available: {len(gpus)}")
```

---

## Training & Evaluation

### 1. Data Stratification & Splitting

Address class imbalance through stratified sampling:

```python
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

data = pd.read_csv('final_model_data.csv')

# Stratified downsampling to balance classes
print("Star distribution before stratification:")
print(data['stars'].value_counts().sort_index())

# Downsample to match the least frequent class
target_size = 130_000 // 5  # 26,000 per class for equal representation
balanced_data = data.groupby('stars', group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), target_size), random_state=1)
)

print("\nStar distribution after stratification:")
print(balanced_data['stars'].value_counts().sort_index())

# Train-test split (80-20)
with open('optimal_features.json', 'r') as f:
    optimal_features = json.load(f)

X = balanced_data[optimal_features].values
y = balanced_data['stars'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
```

### 2. Feature Normalization

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to tensors for PyTorch
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test)
```

### 3. Training with PyTorch Lightning

```python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

# Create dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64)

# Initialize model
model = YelpRatingPredictor(input_size=len(optimal_features), learning_rate=0.0001)

# Configure trainer with Metal GPU
trainer = Trainer(
    max_epochs=40,
    accelerator='cpu',  # 'mps' for Apple GPU, 'cuda' for NVIDIA, 'cpu' for CPU
    devices=1,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, mode='min'),
        ModelCheckpoint(save_top_k=1, monitor='val_loss')
    ],
    enable_progress_bar=True,
    log_every_n_steps=10
)

# Train
trainer.fit(model, train_loader, val_loader)

# Test
trainer.test(model, val_loader)
```

### 4. Evaluation Metrics

```python
# Get predictions
model.eval()
with torch.no_grad():
    y_pred_train = model(X_train_tensor).squeeze().numpy()
    y_pred_test = model(X_test_tensor).squeeze().numpy()

# Calculate metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Train MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test R²: {test_r2:.4f}")
```

---

## Key Challenges & Solutions

### 1. **Text Truncation for DistilBERT**

**Challenge**: Review texts exceed the 512-token limit; crude truncation loses semantic information.

**Solution Options**:
- **Smart truncation**: Keep first + last portions to preserve context
- **Hierarchical approach**: Split long texts into chunks, average sentiment scores
- **Model optimization**: Use a model with longer context window (e.g., `longformer`, `big-bird`)

```python
# Example: Chunk-based averaging
def chunk_sentiment_analysis(text, max_tokens=512, chunk_overlap=50):
    """Split text into overlapping chunks and average sentiment"""
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= max_tokens:
        return sentiment_pipeline(text)[0]
    
    chunks = []
    for i in range(0, len(tokens), max_tokens - chunk_overlap):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
    
    sentiments = sentiment_pipeline(chunks, batch_size=32)
    avg_score = np.mean([s['score'] for s in sentiments])
    label = 'POSITIVE' if avg_score > 0.5 else 'NEGATIVE'
    
    return {'label': label, 'score': avg_score}
```

### 2. **Class Imbalance**

**Challenge**: 60%+ of reviews are 4-5 stars; model biased toward high ratings.

**Solutions Implemented**:
- ✅ **Stratified sampling** (used in original): Equal samples per class
- Alternative: **Weighted loss function**

```python
# Weighted loss to handle class imbalance
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', 
                                      classes=np.unique(y_train),
                                      y=y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
```

### 3. **Overfitting (Per Original Project)**

The learning curve in the original R project shows overfitting: validation loss flat, training loss declining.

**Prevention Strategies**:
```python
# Already implemented:
# - Batch normalization
# - Dropout (50% and 50%)
# - Early stopping (patience=5)

# Additional recommendations:
# - L1/L2 regularization
regularized_model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(5,),
                 kernel_regularizer=keras.regularizers.l2(0.001)),
    # ... rest of model
])

# - Data augmentation (for tabular data, use mixup/cutmix)
# - Ensemble methods (train multiple models, average predictions)
```

### 4. **GPU Memory Optimization for Apple Silicon**

```python
# Clear GPU cache if needed
import torch
torch.mps.empty_cache()

# Reduce batch size if OOM errors occur
batch_size = 32  # Instead of 64

# Use mixed precision training
from pytorch_lightning.plugins import MixedPrecisionPlugin
trainer = Trainer(
    plugins=[MixedPrecisionPlugin(precision='16-mixed')],
    accelerator='mps'
)
```

### 5. **Reproducibility Across Runs**

```python
import random
import numpy as np
import torch

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

set_seed(1)
```

---

## Project Structure

Recommended Python project structure:

```
yelp-prediction/
├── data/
│   ├── yelp_business_data.csv
│   ├── yelp_review.csv
│   ├── yelp_user.csv
│   ├── yelp_checkin_data.csv
│   ├── yelp_tip_data.csv
│   ├── processed/
│   │   ├── merged_data.csv
│   │   ├── featured_data.csv
│   │   ├── sentiment_data.csv
│   │   └── final_model_data.csv
│   └── splits/
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       └── y_test.npy
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration constants
│   ├── data_loading.py        # Load CSV functions
│   ├── preprocessing.py       # Data cleaning & merging
│   ├── features.py            # Feature engineering functions
│   ├── sentiment.py           # Sentiment analysis pipeline
│   ├── feature_selection.py   # RFE and feature ranking
│   ├── model.py               # Model architecture definitions
│   └── train.py               # Training loop
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_sentiment_analysis.ipynb
│   ├── 05_feature_selection.ipynb
│   └── 06_model_training.ipynb
├── models/
│   └── best_model.pt          # Saved trained model
├── outputs/
│   ├── learning_curves.png
│   ├── predictions.csv
│   └── metrics.json
├── requirements.txt
├── setup.py
└── README.md                  # This file
```

---

## Troubleshooting

### Issue: `torch.mps.is_available()` returns False

**Solution**: Ensure PyTorch is installed correctly for Apple Silicon:
```bash
# Reinstall PyTorch
conda uninstall torch torchvision torchaudio
conda install pytorch::pytorch torchvision torchaudio -c pytorch
```

### Issue: Sentiment Analysis Too Slow

**Solutions**:
1. Reduce batch size: `sentiment_pipeline(..., batch_size=16)`
2. Use CPU with multithreading: `tokenizer_parallelism=True`
3. Sample data for testing: `final_data.sample(frac=0.1)`
4. Use smaller model: `distilbert-base-uncased` (faster than full BERT)

### Issue: Memory Errors During Training

**Solutions**:
```python
# Reduce batch size
batch_size = 32  # from 64

# Enable gradient checkpointing (PyTorch)
model = YelpRatingPredictor(...)
model.network.gradient_checkpointing_enable()

# Use mixed precision
trainer = Trainer(precision='16-mixed', accelerator='mps')

# Process data in smaller chunks
train_loader = DataLoader(..., num_workers=0)  # No multiprocessing
```

### Issue: DistilBERT Model Download Fails

**Solution**: Pre-download the model:
```python
from transformers import AutoModel, AutoTokenizer

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Models cached in ~/.cache/huggingface/
```

### Issue: Stratified Sampling Not Working

**Solution**: Ensure stars are categorical:
```python
# Check for non-integer star values
print(data['stars'].unique())
print(data['stars'].dtype)

# Convert if needed
data['stars'] = data['stars'].astype(int)
```

### Issue: Different Results from R Version

**Possible Causes**:
1. **Random seed**: Ensure `set_seed(1)` is called
2. **Stratification**: Different sampling methods → different data splits
3. **Normalization**: MinMaxScaler vs Keras normalize may differ slightly
4. **Model initialization**: Neural network weights randomly initialized

**Validation**: Train on same data split:
```python
# Export R data
data.to_csv('validation_data.csv', index=False)
# Load in Python and compare processing steps
```

---

## Performance Expectations

### Estimated Runtime (on M1/M2 with 16GB RAM)

| Step | Time | Notes |
|------|------|-------|
| Load & merge data | 1-2 min | CSV loading is faster than JSONL |
| Feature engineering | 1-2 min | Vectorized operations |
| Sentiment analysis | 2-4 hrs | Bottleneck; depends on GPU |
| Feature selection (RFE) | 20-30 min | RF with CV |
| Model training | 5-15 min | 40 epochs with early stopping |
| **Total** | **3-5 hours** | Full pipeline |

### Expected Metrics

Based on the original project:
- **Test MSE**: ~0.81
- **Test MAE**: ~0.65
- **Train/Val gap**: Indicates overfitting (addressable with modifications)

---

## References & Further Reading

1. **Yelp Academic Dataset**: https://www.yelp.com/dataset
2. **DistilBERT Paper**: Sanh et al., 2019. "DistilBERT, a distilled version of BERT"
3. **Hugging Face Documentation**: https://huggingface.co/docs/transformers
4. **PyTorch Lightning**: https://pytorch-lightning.readthedocs.io/
5. **Apple Metal Optimization**: https://developer.apple.com/metal/
6. **Scikit-learn RFE**: https://scikit-learn.org/stable/modules/feature_selection.html

---

## License & Attribution

This is a replication guide for the EC349 assignment project. Original project: u2008071, 2023.

For questions about Apple GPU implementation, refer to:
- https://github.com/pytorch/pytorch/issues (MPS-related issues)
- https://developer.apple.com/metal/ (Metal framework documentation)
