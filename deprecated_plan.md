# Yelp Star Rating Prediction Project Plan

## 1. Executive Summary

This project aims to replicate the EC349 assignment's Yelp star rating prediction model in Python, leveraging Apple GPU acceleration for improved performance. The original R implementation will be ported to Python using PyTorch and Hugging Face transformers, with Metal GPU support for Apple Silicon devices. The project builds a neural network to predict user star ratings (1-5) using engineered features from business, user, and review data, incorporating DistilBERT-based sentiment analysis.

Key components include data preprocessing, feature engineering, sentiment analysis, feature selection via Recursive Feature Elimination (RFE), and neural network training with regularization techniques. Expected outcomes include a trained model achieving Test MSE of ~0.8115 and MAE of ~0.65, matching or exceeding the original R implementation's performance.

## 2. Project Scope and Objectives

The implementation will achieve the following:

- **Data Processing**: Load and merge Yelp Academic Dataset CSV files (business, user, review data)
- **Feature Engineering**: Create time-based features (yelping tenure), elite status indicators, and business/user statistics
- **Sentiment Analysis**: Apply DistilBERT transformer to extract sentiment scores from review text with smart truncation for 512-token limit
- **Feature Selection**: Use RFE with Random Forest to identify optimal feature subset (expected: 5 features)
- **Model Training**: Implement sequential neural network (Dense 256 → BN → Dropout → Dense 128 → Dense 1) with early stopping
- **Evaluation**: Achieve comparable performance metrics to original R version (MSE ~0.81, MAE ~0.65)

The project addresses class imbalance through stratified sampling and implements GPU acceleration for sentiment analysis and training phases.

## 3. Prerequisites and Environment Setup

### System Requirements
- **Hardware**: macOS with Apple Silicon (M1/M2/M3) or Intel; 16GB RAM minimum (32GB recommended); 15GB disk space
- **Software**: Python 3.10+, conda or pip package managers

### Environment Setup Steps
1. Create conda environment: `conda create -n yelp-py python=3.10`
2. Install PyTorch with Metal acceleration: `conda install pytorch::pytorch torchvision torchaudio -c pytorch`
3. Install core dependencies:
   - PyTorch Lightning for training framework
   - Transformers for DistilBERT sentiment analysis
   - Scikit-learn for preprocessing and RFE
   - Pandas/NumPy for data manipulation
4. Verify GPU support: `torch.backends.mps.is_available()` should return True

### Validation Criteria
- PyTorch MPS backend available and enabled
- All dependencies importable without errors
- Test script runs sentiment analysis on sample text in <30 seconds

## 4. Implementation Phases

### Phase 1: Data Pipeline (1-2 hours)
**Objectives**: Load, clean, and merge Yelp dataset CSV files
**Required Inputs**: CSV files (yelp_business_data.csv, yelp_review.csv, yelp_user.csv)
**Expected Outputs**: merged_data.csv with cleaned, joined dataset

**Implementation Steps**:
1. Load CSV files using pandas
2. Rename columns for consistency (e.g., 'useful' → 'total_useful')
3. Convert date columns to datetime format
4. Perform inner joins: review → user → business
5. Remove rows with missing critical values
6. Save merged dataset

**Validation Criteria**: Dataset shape ~130k+ rows, no missing values in key columns
**Estimated Time**: 30 minutes
**Resources**: 4GB RAM

### Phase 2: Feature Engineering (30-45 minutes)
**Objectives**: Create engineered features from raw data
**Required Inputs**: merged_data.csv
**Expected Outputs**: featured_data.csv

**Implementation Steps**:
1. Calculate time on platform: `(date - yelping_since).dt.total_seconds() / (7*24*3600)`
2. Parse elite status: extract years from pipe-separated string, count total elite statuses
3. Create elite status indicators: check if user had elite status in review year
4. Handle missing values: median imputation for time_yelping, 0 for elite features
5. Save featured dataset

**Validation Criteria**: New columns created without NaN values, time_yelping > 0, elite_status binary
**Estimated Time**: 30 minutes
**Resources**: 8GB RAM

### Phase 3: Sentiment Analysis (2-4 hours)
**Objectives**: Extract sentiment scores from review text using DistilBERT
**Required Inputs**: featured_data.csv
**Expected Outputs**: sentiment_data.csv

**Implementation Steps**:
1. Implement smart truncation: keep first 256 + last 256 tokens for context preservation
2. Initialize Hugging Face sentiment pipeline with MPS device
3. Batch process reviews (batch_size=64) with tqdm progress monitoring
4. Convert sentiment to signed score: negative = -score, positive = +score
5. Save sentiment-enriched dataset

**Validation Criteria**: sentiment_score_raw between 0-1, normalized_sentiment_score between -1 and 1, no processing errors
**Estimated Time**: 3 hours (GPU bottleneck)
**Resources**: 16GB RAM, Apple GPU

### Phase 4: Feature Selection (20-30 minutes)
**Objectives**: Identify optimal feature subset using RFE
**Required Inputs**: sentiment_data.csv
**Expected Outputs**: final_model_data.csv, optimal_features.json

**Implementation Steps**:
1. Define candidate features list (7 features)
2. Remove rows with missing values in candidates + target
3. Initialize RandomForestRegressor with n_estimators=100
4. Run RFE with step=1 to select optimal subset
5. Extract and save selected features (expected: 5 features)
6. Prepare final dataset for modeling

**Validation Criteria**: RFE selects exactly 5 features including sentiment_score, no missing values in final dataset
**Estimated Time**: 25 minutes
**Resources**: 8GB RAM

### Phase 5: Model Training & Evaluation (15-30 minutes)
**Objectives**: Train neural network and evaluate performance
**Required Inputs**: final_model_data.csv, optimal_features.json
**Expected Outputs**: trained model checkpoint, evaluation metrics

**Implementation Steps**:
1. Implement stratified sampling: downsample to 26k samples per class (130k total)
2. Split data 80/20 train/test with stratification
3. Normalize features using MinMaxScaler
4. Define PyTorch Lightning model: Dense(256) → BN → Dropout(0.5) → Dense(128) → Dropout(0.5) → Dense(1)
5. Configure training: RMSprop optimizer, early stopping (patience=5), max 40 epochs
6. Train with MPS accelerator, evaluate MSE/MAE/R² metrics

**Validation Criteria**: Test MSE ≤ 0.85, MAE ≤ 0.70, training completes without GPU memory errors
**Estimated Time**: 20 minutes
**Resources**: 8GB RAM, Apple GPU

## 5. Project Structure

```
yelp-prediction/
├── data/
│   ├── yelp_business_data.csv
│   ├── yelp_review.csv
│   ├── yelp_user.csv
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
│   ├── config.py              # Configuration constants
│   ├── data_loading.py        # Load CSV functions
│   ├── preprocessing.py       # Data cleaning & merging
│   ├── features.py            # Feature engineering functions
│   ├── sentiment.py           # Sentiment analysis pipeline
│   ├── feature_selection.py   # RFE and feature ranking
│   ├── model.py               # Model architecture definitions
│   └── train.py               # Training loop
├── models/
│   └── best_model.pt          # Saved trained model
├── outputs/
│   ├── learning_curves.png
│   ├── predictions.csv
│   └── metrics.json
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_sentiment_analysis.ipynb
│   ├── 05_feature_selection.ipynb
│   └── 06_model_training.ipynb
├── requirements.txt
├── setup.py
└── README.md
```

## 6. Risk Assessment and Mitigation

### Memory Issues During Sentiment Analysis
**Risk**: GPU memory exhaustion with large batches or long reviews
**Mitigation**: Reduce batch_size to 32, implement gradient checkpointing, use CPU fallback

### GPU Compatibility Problems
**Risk**: MPS backend not available or unstable on certain macOS versions
**Mitigation**: Verify torch.backends.mps.is_available(), implement CPU fallback, update PyTorch to latest version

### Class Imbalance Impact
**Risk**: Model biased toward high ratings despite stratification
**Mitigation**: Validate stratification effectiveness, consider weighted loss functions, monitor per-class metrics

### Overfitting
**Risk**: Training loss decreases while validation loss plateaus
**Mitigation**: Implement dropout (50%), batch normalization, early stopping, L1/L2 regularization

### Text Truncation Information Loss
**Risk**: DistilBERT 512-token limit loses semantic information
**Mitigation**: Use smart truncation (first + last portions), consider chunk-based averaging, evaluate alternative models

### Reproducibility Issues
**Risk**: Different results due to random seeds or data splits
**Mitigation**: Set seeds (random=1, numpy=1, torch=1), use fixed train_test_split random_state=1

## 7. Testing and Validation Strategy

### Unit Testing Approach
- **Data Loading**: Verify CSV files load correctly, check column names and dtypes
- **Feature Engineering**: Test elite parsing functions, validate time calculations
- **Sentiment Analysis**: Compare sample outputs against expected sentiment labels
- **Feature Selection**: Verify RFE selects expected 5 features
- **Model Training**: Check tensor shapes, loss convergence, GPU utilization

### Integration Testing
- **End-to-End Pipeline**: Run complete pipeline on 10% sample data, verify intermediate file creation
- **Performance Validation**: Compare metrics against original R implementation (±5% tolerance)
- **GPU Acceleration**: Monitor that sentiment analysis uses MPS backend, training completes in expected time

### Validation Against Original R Implementation
- **Data Processing**: Compare merged dataset shapes and summary statistics
- **Feature Engineering**: Verify elite status counts and time calculations match
- **Sentiment Scores**: Sample 100 reviews, compare sentiment scores (±0.1 tolerance)
- **Model Performance**: Achieve Test MSE ≤ 0.85, MAE ≤ 0.70, R² ≥ 0.15

## 8. Timeline and Milestones

### Phase 1-2: Data Preparation (Week 1)
- Day 1: Environment setup and data loading
- Day 2: Feature engineering completion
- **Milestone**: featured_data.csv created with all engineered features

### Phase 3: Sentiment Analysis (Week 1-2)
- Day 3-4: Sentiment pipeline implementation and testing
- Day 5: Full dataset sentiment analysis (GPU bottleneck)
- **Milestone**: sentiment_data.csv with sentiment scores for all reviews

### Phase 4-5: Modeling (Week 2)
- Day 6: Feature selection and model preparation
- Day 7: Training and evaluation
- **Milestone**: Trained model achieving target metrics

### Total Timeline: 1-2 weeks
- **Optimistic**: 5 days with fast GPU
- **Realistic**: 7-10 days including troubleshooting
- **Pessimistic**: 14 days with hardware/software issues

## 9. Dependencies and Integration Points

### Sequential Dependencies
1. **Data Pipeline** → **Feature Engineering**: Merged data required for feature calculations
2. **Feature Engineering** → **Sentiment Analysis**: Text column needed for transformer processing
3. **Sentiment Analysis** → **Feature Selection**: Sentiment scores required for RFE
4. **Feature Selection** → **Model Training**: Optimal features needed for model input

### Parallel Processing Opportunities
- Data loading can run independently
- Feature engineering (non-sentiment) can be parallelized
- Model architecture definition independent of data processing

### External Dependencies
- **Hugging Face Hub**: DistilBERT model download (pre-cache recommended)
- **PyTorch MPS**: Apple GPU acceleration (fallback to CPU)
- **Yelp Dataset**: Pre-converted CSV files must be available

### Integration Points
- **Data Flow**: CSV → pandas DataFrame → numpy arrays → PyTorch tensors
- **Model Integration**: PyTorch Lightning handles GPU/CPU abstraction
- **Configuration**: Centralized config.py for hyperparameters and file paths

## 10. Success Criteria

### Technical Success Metrics
- **Data Processing**: Successfully merge 130k+ reviews without data loss
- **Feature Engineering**: Create all required features with <1% missing values
- **Sentiment Analysis**: Process full dataset with GPU acceleration in <4 hours
- **Feature Selection**: RFE selects 5 optimal features including sentiment_score
- **Model Training**: Complete training without GPU memory errors

### Performance Success Metrics
- **Test MSE**: ≤ 0.85 (target: ~0.81)
- **Test MAE**: ≤ 0.70 (target: ~0.65)
- **Test R²**: ≥ 0.10 (target: ~0.15)
- **Training Time**: <30 minutes for 40 epochs
- **GPU Utilization**: Sentiment analysis uses MPS backend

### Validation Success Criteria
- **Reproducibility**: Same random seed produces identical results
- **Compatibility**: Code runs on Apple Silicon M1/M2/M3 and Intel Macs
- **Maintainability**: Modular code structure with clear separation of concerns
- **Documentation**: Complete README with setup and usage instructions

### Overall Project Success
Project is complete when the Python implementation matches or exceeds the original R version's performance, runs efficiently on Apple hardware, and provides a reproducible, well-documented codebase for future extensions.