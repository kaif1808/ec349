import pandas as pd
import json
import torch
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Tuple, Dict, Any
from src import config
from src.model import YelpRatingPredictor
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def stratify_and_split(df: pd.DataFrame, target_size: int = 130000) -> pd.DataFrame:
    """
    Stratify the dataset by 'stars' and downsample to equal samples per class.

    Args:
        df: Input DataFrame containing 'stars' column
        target_size: Total target size for the stratified dataset

    Returns:
        Stratified DataFrame with equal samples per class
    """
    # Group by 'stars' and downsample each group
    samples_per_class = target_size // 5  # 5 classes (1-5 stars)

    stratified_df = df.groupby('stars', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), samples_per_class), random_state=1)
    ).reset_index(drop=True)

    return stratified_df


def prepare_train_test_data(df: pd.DataFrame, features: List[str], test_size: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, MinMaxScaler]:
    """
    Prepare train/test data with stratification, normalization, and PyTorch tensor conversion.

    Args:
        df: Input DataFrame
        features: List of feature column names
        test_size: Fraction of data to use for testing

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler)
    """
    # Prepare features and target
    X = df[features]
    y = df['stars']

    # Split into train/test with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=1
    )

    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values)
    y_test_tensor = torch.FloatTensor(y_test.values)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler
def create_dataloaders(X_train, y_train, X_val, y_val, batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and validation datasets.

    Args:
        X_train: Training features tensor
        y_train: Training labels tensor
        X_val: Validation features tensor
        y_val: Validation labels tensor
        batch_size: Batch size for DataLoaders

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, max_epochs: int = 40) -> pl.Trainer:
    """
    Train the model using PyTorch Lightning.

    Args:
        model: PyTorch Lightning model to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        max_epochs: Maximum number of epochs

    Returns:
        Trained PyTorch Lightning Trainer
    """
    # Detect device
    if torch.backends.mps.is_available():
        accelerator = 'mps'
    else:
        accelerator = 'cpu'

    # Configure callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Configure trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=max_epochs,
        callbacks=[early_stopping]
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), os.path.join('models', 'best_model.pt'))

    return trainer


def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    """
    Evaluate the model on test data and return metrics.

    Args:
        model: Trained PyTorch model
        X_test: Test features tensor
        y_test: Test labels tensor

    Returns:
        Dictionary with MSE, MAE, and R² metrics
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        mse = mean_squared_error(y_test.cpu().numpy(), predictions.cpu().numpy())
        mae = mean_absolute_error(y_test.cpu().numpy(), predictions.cpu().numpy())
        r2 = r2_score(y_test.cpu().numpy(), predictions.cpu().numpy())
    return {'mse': mse, 'mae': mae, 'r2': r2}


def training_pipeline() -> Dict[str, Any]:
    """
    Complete training pipeline: load data, train model, evaluate, and save artifacts.

    Returns:
        Dictionary with results including metrics and file paths
    """
    # Load final model data and optimal features
    df = pd.read_csv('data/processed/final_model_data.csv')
    with open('data/processed/optimal_features.json', 'r') as f:
        features = json.load(f)

    # Stratify data
    df_stratified = stratify_and_split(df)

    # Prepare train/test splits
    X_train, X_test, y_train, y_test, scaler = prepare_train_test_data(df_stratified, features)

    # Split train into train/val for training
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=1
    )

    # Create DataLoaders
    train_loader, val_loader = create_dataloaders(
        X_train_split, y_train_split, X_val, y_val, batch_size=config.BATCH_SIZE
    )

    # Initialize model
    input_size = len(features)
    model = YelpRatingPredictor(input_size=input_size, learning_rate=config.LEARNING_RATE)

    # Train model
    trainer = train_model(model, train_loader, val_loader, max_epochs=config.MAX_EPOCHS)

    # Evaluate on test set
    metrics = evaluate_model(model, X_test, y_test)

    # Save scaler
    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Save metrics
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/metrics.json', 'w') as f:
        json.dump(metrics, f)

    # Return results
    return {
        'metrics': metrics,
        'model_path': 'models/best_model.pt',
        'scaler_path': 'models/scaler.pkl',
        'metrics_path': 'outputs/metrics.json'
    }