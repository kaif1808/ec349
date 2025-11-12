import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any


class YelpRatingPredictor(pl.LightningModule):
    """
    PyTorch Lightning module for predicting Yelp ratings using a neural network.

    This model consists of a feedforward neural network with dropout and batch normalization
    layers, designed to predict star ratings based on input features.

    Attributes:
        network: Sequential neural network layers
        criterion: Mean squared error loss function
    """

    def __init__(self, input_size: int = 5, learning_rate: float = 0.0001) -> None:
        super().__init__()
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
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor

        Returns:
            Output tensor predictions
        """
        return self.network(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Training step for one batch.

        Args:
            batch: Tuple of (features, targets)
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Validation step for one batch.

        Args:
            batch: Tuple of (features, targets)
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        mae = torch.mean(torch.abs(preds - y))
        self.log('val_loss', loss)
        self.log('val_mae', mae)
        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step for one batch.

        Args:
            batch: Tuple of (features, targets)
            batch_idx: Batch index

        Returns:
            Dictionary with test loss and MAE
        """
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        mae = torch.mean(torch.abs(preds - y))
        return {'test_loss': loss, 'test_mae': mae}

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizer and learning rate scheduler.

        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=3
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}