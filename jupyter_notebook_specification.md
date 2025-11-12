# Jupyter Notebook Specification: Interactive Yelp Rating Prediction Pipeline

## Overview

This specification outlines the requirements for creating an interactive Jupyter notebook version of the automated machine learning pipeline for Yelp star rating prediction. The notebook should transform the linear script execution into an educational, interactive experience that allows users to:

- Understand each pipeline stage step-by-step
- Visualize data and results at each stage
- Modify parameters and observe their impact
- Learn about machine learning concepts through hands-on exploration
- Debug and troubleshoot the pipeline interactively

## Notebook Structure and Sections

### 1. Introduction and Setup

**Purpose**: Introduce the project, explain the dataset, and set up the environment.

**Required Content**:
- Project overview and objectives
- Dataset description (Yelp Academic Dataset)
- Pipeline stages overview with expected outputs
- Environment setup (imports, GPU detection, reproducibility)
- Configuration display

**Interactive Elements**:
- Environment verification cells
- GPU availability check with fallback options
- Configuration parameter display

**Code Snippets**:
```python
# Environment setup
import sys
import os
import logging
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

# Import all modules
from src.preprocessing import preprocess_pipeline
from src.features import feature_engineering_pipeline
from src.sentiment import sentiment_analysis_pipeline
from src.feature_selection import feature_selection_pipeline
from src.train import training_pipeline
from src import config
import src.utils as utils

# GPU detection
device_info = utils.verify_gpu_support()
print(f"GPU Support: {device_info}")

# Set seed for reproducibility
utils.set_seed(config.SEED)
print(f"Random seed set to: {config.SEED}")
```

### 2. Data Loading and Preprocessing

**Purpose**: Load raw data and perform initial preprocessing steps.

**Required Content**:
- File path verification
- Data loading with progress bars
- Initial data exploration (head, shape, dtypes)
- Column renaming explanation
- Date conversion demonstration
- Data merging visualization
- Missing value analysis

**Interactive Elements**:
- Data preview widgets
- Shape and memory usage display
- Missing value heatmaps
- Distribution plots for key variables

**Code Snippets**:
```python
# Load and display data info
print("Loading Yelp datasets...")
print(f"Business data: {config.BUSINESS_DATA_PATH}")
print(f"Review data: {config.REVIEW_DATA_PATH}")
print(f"User data: {config.USER_DATA_PATH}")

# Run preprocessing pipeline
merged_df = preprocess_pipeline()

# Display results
print(f"Merged dataset shape: {merged_df.shape}")
print(f"Memory usage: {merged_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
display(merged_df.head())
```

### 3. Feature Engineering

**Purpose**: Create derived features from raw data.

**Required Content**:
- Time-based feature creation (time_yelping)
- Elite status parsing and feature engineering
- Missing value imputation strategies
- Feature correlation analysis
- Distribution visualizations

**Interactive Elements**:
- Before/after feature engineering comparisons
- Correlation heatmaps
- Feature importance previews
- Interactive parameter tuning for imputation

**Code Snippets**:
```python
# Feature engineering pipeline
featured_df = feature_engineering_pipeline(merged_df)

# Display new features
new_features = [col for col in featured_df.columns if col not in merged_df.columns]
print(f"Added {len(new_features)} new features: {new_features}")

# Show feature statistics
display(featured_df[new_features].describe())
```

### 4. Sentiment Analysis

**Purpose**: Extract sentiment scores from review text using transformers.

**Required Content**:
- Text preprocessing (smart truncation)
- Model initialization and device configuration
- Batch processing with progress tracking
- Sentiment score normalization
- Text length analysis and truncation visualization

**Interactive Elements**:
- Sample text processing demonstration
- Sentiment distribution histograms
- Text length vs sentiment correlation
- Interactive batch size adjustment
- Model selection options (if multiple available)

**Code Snippets**:
```python
# Sentiment analysis pipeline
sentiment_df = sentiment_analysis_pipeline(featured_df)

# Display sentiment statistics
sentiment_cols = ['sentiment_label', 'sentiment_score_raw', 'normalized_sentiment_score']
display(sentiment_df[sentiment_cols].describe())

# Visualize sentiment distribution
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sentiment_df['normalized_sentiment_score'].hist(bins=50, alpha=0.7)
plt.title('Distribution of Normalized Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()
```

### 5. Feature Selection

**Purpose**: Identify optimal feature subset using statistical methods.

**Required Content**:
- Feature correlation analysis
- Best subset selection implementation
- Feature ranking and importance scores
- Selected features validation
- Performance comparison (before/after selection)

**Interactive Elements**:
- Feature correlation matrix
- Selection algorithm parameter tuning
- Feature importance bar charts
- Cross-validation performance plots

**Code Snippets**:
```python
# Feature selection pipeline
final_df, optimal_features = feature_selection_pipeline(sentiment_df)

print(f"Selected {len(optimal_features)} optimal features:")
for i, feature in enumerate(optimal_features, 1):
    print(f"{i}. {feature}")

# Display final dataset info
print(f"Final dataset shape: {final_df.shape}")
display(final_df.head())
```

### 6. Model Training and Evaluation

**Purpose**: Train neural network and evaluate performance.

**Required Content**:
- Data stratification and splitting
- Model architecture visualization
- Training progress with metrics
- Hyperparameter configuration
- Performance evaluation (MSE, MAE, R²)
- Learning curves and validation plots

**Interactive Elements**:
- Training parameter sliders (learning rate, batch size, epochs)
- Real-time loss plotting
- Model prediction vs actual scatter plots
- Hyperparameter impact analysis
- Cross-validation results

**Code Snippets**:
```python
# Model training pipeline
training_results = training_pipeline()

# Display training metrics
metrics = training_results['metrics']
print("Training Results:")
print(f"MSE: {metrics['mse']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
print(f"R²: {metrics['r2']:.4f}")

# Load and display model info
model_path = training_results['model_path']
scaler_path = training_results['scaler_path']
print(f"Model saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")
```

### 7. Inference and Predictions

**Purpose**: Demonstrate model usage for new predictions.

**Required Content**:
- Model loading and preparation
- Example input creation
- Prediction generation
- Results interpretation
- Batch prediction capabilities

**Interactive Elements**:
- Custom input forms for prediction
- Prediction confidence intervals
- Error analysis visualizations
- Batch prediction upload interface

**Code Snippets**:
```python
# Load model and scaler
import torch
import pickle

model = YelpRatingPredictor(input_size=len(optimal_features))
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Create example prediction
example_input = {
    'user_average_stars': 4.2,
    'business_average_stars': 4.0,
    'time_yelping': 52.3,
    'elite_status': 1,
    'normalized_sentiment_score': 0.8
}

# Filter to optimal features and predict
filtered_input = {k: v for k, v in example_input.items() if k in optimal_features}
input_df = pd.DataFrame([filtered_input])
scaled_input = scaler.transform(input_df.values)
input_tensor = torch.FloatTensor(scaled_input)

with torch.no_grad():
    prediction = model(input_tensor).item()

print(f"Predicted rating: {prediction:.2f}")
print(f"Actual range: 1-5 stars")
```

### 8. Analysis and Insights

**Purpose**: Provide deeper analysis of results and model behavior.

**Required Content**:
- Feature importance analysis
- Error analysis by rating category
- Model limitations and assumptions
- Performance comparison with baselines
- Future improvement suggestions

**Interactive Elements**:
- Feature contribution analysis
- Prediction error distribution
- Confidence interval calculations
- What-if scenario analysis

## Technical Specifications

### Cell Organization
- **Markdown cells**: Clear explanations, mathematical formulas, results interpretation
- **Code cells**: Modular, well-commented, error-handled
- **Output cells**: Rich displays, plots, tables, interactive widgets

### Error Handling
- Try-except blocks around all major operations
- Informative error messages with troubleshooting guidance
- Graceful fallbacks for missing dependencies

### Performance Considerations
- Memory usage monitoring
- Progress bars for long-running operations
- Caching options for intermediate results
- GPU memory management

### Educational Elements
- **Learning objectives** at the start of each section
- **Key concepts** explanations with examples
- **Exercises** for users to modify and experiment
- **Discussion questions** to encourage critical thinking

### Visualization Requirements
- Matplotlib and seaborn for static plots
- Plotly for interactive visualizations
- Pandas styling for dataframes
- Custom plotting functions for consistency

### Parameter Customization
- Configurable batch sizes, learning rates, epochs
- Feature selection algorithm options
- Model architecture variations
- Cross-validation fold selection

## Dependencies and Environment

### Required Packages
- All packages from requirements.txt
- Additional notebook-specific packages:
  - jupyter-widgets
  - ipywidgets
  - plotly
  - matplotlib
  - seaborn

### Environment Setup
- Conda environment with GPU support
- Jupyter notebook server configuration
- Extension installations (widgets, plotting)

## Testing and Validation

### Notebook Testing
- Run all cells successfully
- Verify outputs match expected results
- Test interactive elements functionality
- Validate visualizations render correctly

### Performance Validation
- Memory usage stays within limits
- GPU utilization is efficient
- Processing times are reasonable
- Results match script outputs

## Documentation and User Guide

### Inline Documentation
- Comprehensive docstrings
- Code comments explaining complex logic
- Markdown explanations of concepts

### User Guidance
- Setup instructions
- Troubleshooting guide
- Parameter explanation
- Expected runtime information

This specification ensures the Jupyter notebook provides an engaging, educational experience while maintaining the full functionality of the automated pipeline.