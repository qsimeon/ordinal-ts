"""
Self-Supervised Time Series Ordinality Prediction

This script implements a self-supervised learning task for time series understanding.
The network learns to predict the correct order (ordinality) of scrambled time series segments.

The approach:
1. Take a time series and divide it into bins/segments
2. Scramble the order of these bins
3. Train a neural network to predict the original position (ordinal rank) of each bin
4. Use cross-entropy loss and optional ranking-based metrics

This helps the network learn meaningful temporal representations without labels.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Configuration for the self-supervised ordinality prediction task."""
    num_bins: int = 5  # Number of segments to divide time series into
    hidden_dim: int = 128
    num_layers: int = 2
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 50
    sequence_length: int = 100  # Length of each time series
    num_samples: int = 1000  # Number of synthetic time series to generate
    dropout: float = 0.2
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Data Generation
# ============================================================================

def generate_synthetic_time_series(num_samples: int, length: int, 
                                   noise_level: float = 0.1) -> np.ndarray:
    """
    Generate synthetic time series data with various patterns.
    
    Args:
        num_samples: Number of time series to generate
        length: Length of each time series
        noise_level: Amount of Gaussian noise to add
        
    Returns:
        Array of shape (num_samples, length)
    """
    time_series = []
    
    for i in range(num_samples):
        t = np.linspace(0, 4 * np.pi, length)
        
        # Mix different patterns
        pattern_type = i % 4
        
        if pattern_type == 0:
            # Sine wave with trend
            signal = np.sin(t) + 0.1 * t
        elif pattern_type == 1:
            # Multiple frequencies
            signal = np.sin(t) + 0.5 * np.sin(3 * t)
        elif pattern_type == 2:
            # Exponential growth/decay
            signal = np.exp(-t / 10) * np.sin(t)
        else:
            # Polynomial trend with oscillation
            signal = 0.01 * t**2 + np.cos(2 * t)
        
        # Add noise
        signal += np.random.normal(0, noise_level, length)
        time_series.append(signal)
    
    return np.array(time_series, dtype=np.float32)


def scramble_time_series(time_series: np.ndarray, num_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Divide time series into bins and scramble them.
    
    Args:
        time_series: Array of shape (length,)
        num_bins: Number of bins to divide the series into
        
    Returns:
        scrambled_series: Scrambled time series of shape (num_bins, bin_length)
        original_positions: Original position indices for each bin
    """
    length = len(time_series)
    bin_length = length // num_bins
    
    # Truncate to make it divisible
    truncated_length = bin_length * num_bins
    time_series = time_series[:truncated_length]
    
    # Divide into bins
    bins = time_series.reshape(num_bins, bin_length)
    
    # Create permutation
    original_positions = np.arange(num_bins)
    scrambled_positions = np.random.permutation(num_bins)
    
    # Scramble bins
    scrambled_bins = bins[scrambled_positions]
    
    # Create inverse mapping (what was the original position of each scrambled bin)
    inverse_mapping = np.argsort(scrambled_positions)
    
    return scrambled_bins, inverse_mapping


# ============================================================================
# Dataset
# ============================================================================

class OrdinalityDataset(Dataset):
    """Dataset for self-supervised ordinality prediction."""
    
    def __init__(self, time_series_data: np.ndarray, num_bins: int):
        """
        Args:
            time_series_data: Array of shape (num_samples, length)
            num_bins: Number of bins to divide each series into
        """
        self.time_series_data = time_series_data
        self.num_bins = num_bins
        self.num_samples = len(time_series_data)
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            scrambled_bins: Tensor of shape (num_bins, bin_length)
            labels: Tensor of shape (num_bins,) with original positions
        """
        time_series = self.time_series_data[idx]
        scrambled_bins, labels = scramble_time_series(time_series, self.num_bins)
        
        return torch.FloatTensor(scrambled_bins), torch.LongTensor(labels)


# ============================================================================
# Model Architecture
# ============================================================================

class OrdinalityPredictor(nn.Module):
    """
    Neural network that predicts the ordinality of scrambled time series bins.
    
    Architecture:
    1. Process each bin with a shared LSTM encoder
    2. Aggregate bin representations
    3. Use attention mechanism to capture relationships between bins
    4. Predict the original position for each bin
    """
    
    def __init__(self, bin_length: int, hidden_dim: int, num_bins: int, 
                 num_layers: int = 2, dropout: float = 0.2):
        """
        Args:
            bin_length: Length of each time series bin
            hidden_dim: Hidden dimension size
            num_bins: Number of bins (also the number of classes)
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(OrdinalityPredictor, self).__init__()
        
        self.bin_length = bin_length
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins
        
        # LSTM encoder for each bin
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Self-attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Classification head for each bin
        self.classifier = nn.Linear(hidden_dim, num_bins)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, num_bins, bin_length)
            
        Returns:
            logits: Tensor of shape (batch_size, num_bins, num_bins)
                   Predictions for the original position of each bin
        """
        batch_size, num_bins, bin_length = x.shape
        
        # Reshape to process all bins together
        x = x.reshape(batch_size * num_bins, bin_length, 1)
        
        # Encode each bin with LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state as bin representation
        bin_representations = lstm_out[:, -1, :]  # (batch_size * num_bins, hidden_dim)
        
        # Reshape back to separate bins
        bin_representations = bin_representations.reshape(batch_size, num_bins, self.hidden_dim)
        
        # Apply self-attention to capture relationships between bins
        attn_out, _ = self.attention(
            bin_representations, 
            bin_representations, 
            bin_representations
        )
        
        # Residual connection and layer norm
        bin_representations = self.layer_norm1(bin_representations + attn_out)
        
        # Feed-forward network with residual connection
        ffn_out = self.ffn(bin_representations)
        bin_representations = self.layer_norm2(bin_representations + ffn_out)
        
        # Apply dropout
        bin_representations = self.dropout(bin_representations)
        
        # Classify each bin's original position
        logits = self.classifier(bin_representations)  # (batch_size, num_bins, num_bins)
        
        return logits


# ============================================================================
# Training and Evaluation
# ============================================================================

class OrdinalityTrainer:
    """Trainer for the ordinality prediction task."""
    
    def __init__(self, model: nn.Module, config: Config):
        """
        Args:
            model: The neural network model
            config: Configuration object
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def compute_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute accuracy of predictions.
        
        Args:
            logits: Tensor of shape (batch_size, num_bins, num_bins)
            labels: Tensor of shape (batch_size, num_bins)
            
        Returns:
            Accuracy as a float
        """
        predictions = torch.argmax(logits, dim=2)  # (batch_size, num_bins)
        correct = (predictions == labels).float()
        accuracy = correct.mean().item()
        return accuracy
    
    def compute_kendall_tau(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute Kendall's Tau correlation (ranking metric).
        
        Args:
            logits: Tensor of shape (batch_size, num_bins, num_bins)
            labels: Tensor of shape (batch_size, num_bins)
            
        Returns:
            Average Kendall's Tau across batch
        """
        predictions = torch.argmax(logits, dim=2).cpu().numpy()
        labels = labels.cpu().numpy()
        
        taus = []
        for pred, label in zip(predictions, labels):
            # Count concordant and discordant pairs
            concordant = 0
            discordant = 0
            
            for i in range(len(pred)):
                for j in range(i + 1, len(pred)):
                    pred_order = pred[i] < pred[j]
                    label_order = label[i] < label[j]
                    
                    if pred_order == label_order:
                        concordant += 1
                    else:
                        discordant += 1
            
            n = len(pred)
            tau = (concordant - discordant) / (n * (n - 1) / 2)
            taus.append(tau)
        
        return np.mean(taus)
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Average loss and accuracy for the epoch
        """
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for batch_idx, (scrambled_bins, labels) in enumerate(dataloader):
            scrambled_bins = scrambled_bins.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits = self.model(scrambled_bins)
            
            # Compute loss
            # Reshape for cross-entropy: (batch_size * num_bins, num_bins)
            batch_size, num_bins, num_classes = logits.shape
            logits_flat = logits.reshape(batch_size * num_bins, num_classes)
            labels_flat = labels.reshape(batch_size * num_bins)
            
            loss = self.criterion(logits_flat, labels_flat)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_accuracy += self.compute_accuracy(logits, labels)
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_accuracy
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """
        Evaluate the model.
        
        Returns:
            Average loss, accuracy, and Kendall's Tau
        """
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_kendall_tau = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for scrambled_bins, labels in dataloader:
                scrambled_bins = scrambled_bins.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(scrambled_bins)
                
                # Compute loss
                batch_size, num_bins, num_classes = logits.shape
                logits_flat = logits.reshape(batch_size * num_bins, num_classes)
                labels_flat = labels.reshape(batch_size * num_bins)
                
                loss = self.criterion(logits_flat, labels_flat)
                
                # Track metrics
                total_loss += loss.item()
                total_accuracy += self.compute_accuracy(logits, labels)
                total_kendall_tau += self.compute_kendall_tau(logits, labels)
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_kendall_tau = total_kendall_tau / num_batches
        
        return avg_loss, avg_accuracy, avg_kendall_tau
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 80)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Evaluate
            val_loss, val_acc, val_tau = self.evaluate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print progress
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Tau: {val_tau:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  → New best model (val_loss: {val_loss:.4f})")
        
        print("-" * 80)
        print("Training completed!")


# ============================================================================
# Visualization
# ============================================================================

def visualize_results(trainer: OrdinalityTrainer, test_loader: DataLoader, 
                     num_examples: int = 3):
    """
    Visualize training results and example predictions.
    
    Args:
        trainer: Trained model trainer
        test_loader: Test data loader
        num_examples: Number of examples to visualize
    """
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    axes[0].plot(trainer.train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(trainer.val_losses, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(trainer.train_accuracies, label='Train Accuracy', linewidth=2)
    axes[1].plot(trainer.val_accuracies, label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("Saved training curves to 'training_curves.png'")
    plt.close()
    
    # Visualize example predictions
    trainer.model.eval()
    
    # Get a batch of test data
    scrambled_bins, labels = next(iter(test_loader))
    scrambled_bins = scrambled_bins.to(trainer.device)
    labels = labels.to(trainer.device)
    
    with torch.no_grad():
        logits = trainer.model(scrambled_bins)
        predictions = torch.argmax(logits, dim=2)
    
    # Move to CPU for plotting
    scrambled_bins = scrambled_bins.cpu().numpy()
    labels = labels.cpu().numpy()
    predictions = predictions.cpu().numpy()
    
    # Plot examples
    fig, axes = plt.subplots(num_examples, 1, figsize=(14, 4 * num_examples))
    if num_examples == 1:
        axes = [axes]
    
    for idx in range(min(num_examples, len(scrambled_bins))):
        ax = axes[idx]
        
        # Reconstruct the time series
        bins = scrambled_bins[idx]
        true_order = labels[idx]
        pred_order = predictions[idx]
        
        # Plot scrambled version
        scrambled_series = bins.flatten()
        t_scrambled = np.arange(len(scrambled_series))
        
        # Reconstruct with true order
        true_reconstructed = bins[np.argsort(true_order)].flatten()
        
        # Reconstruct with predicted order
        pred_reconstructed = bins[np.argsort(pred_order)].flatten()
        
        ax.plot(t_scrambled, scrambled_series, 'gray', alpha=0.5, 
                linewidth=1.5, label='Scrambled')
        ax.plot(t_scrambled, true_reconstructed, 'g-', linewidth=2, 
                label='True Order')
        ax.plot(t_scrambled, pred_reconstructed, 'r--', linewidth=2, 
                label='Predicted Order')
        
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(f'Example {idx + 1}: True Order = {true_order}, '
                    f'Predicted = {pred_order}', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_examples.png', dpi=150, bbox_inches='tight')
    print("Saved prediction examples to 'prediction_examples.png'")
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("Self-Supervised Time Series Ordinality Prediction")
    print("=" * 80)
    print()
    
    # Configuration
    config = Config()
    print("Configuration:")
    print(f"  Number of bins: {config.num_bins}")
    print(f"  Sequence length: {config.sequence_length}")
    print(f"  Hidden dimension: {config.hidden_dim}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Number of epochs: {config.num_epochs}")
    print(f"  Device: {config.device}")
    print()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Generate synthetic data
    print("Generating synthetic time series data...")
    all_time_series = generate_synthetic_time_series(
        num_samples=config.num_samples,
        length=config.sequence_length
    )
    
    # Split into train/val/test
    train_size = int(0.7 * config.num_samples)
    val_size = int(0.15 * config.num_samples)
    
    train_data = all_time_series[:train_size]
    val_data = all_time_series[train_size:train_size + val_size]
    test_data = all_time_series[train_size + val_size:]
    
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Test samples: {len(test_data)}")
    print()
    
    # Create datasets
    train_dataset = OrdinalityDataset(train_data, config.num_bins)
    val_dataset = OrdinalityDataset(val_data, config.num_bins)
    test_dataset = OrdinalityDataset(test_data, config.num_bins)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Calculate bin length
    bin_length = config.sequence_length // config.num_bins
    
    # Create model
    print("Creating model...")
    model = OrdinalityPredictor(
        bin_length=bin_length,
        hidden_dim=config.hidden_dim,
        num_bins=config.num_bins,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    print()
    
    # Create trainer
    trainer = OrdinalityTrainer(model, config)
    
    # Train model
    print("Starting training...")
    print()
    trainer.train(train_loader, val_loader)
    print()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_acc, test_tau = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Kendall's Tau: {test_tau:.4f}")
    print()
    
    # Visualize results
    print("Generating visualizations...")
    visualize_results(trainer, test_loader, num_examples=3)
    print()
    
    # Demonstrate inference on a single example
    print("=" * 80)
    print("Single Example Inference")
    print("=" * 80)
    
    # Get one example
    single_example = test_data[0]
    scrambled_bins, true_labels = scramble_time_series(single_example, config.num_bins)
    
    print(f"Original time series shape: {single_example.shape}")
    print(f"Scrambled bins shape: {scrambled_bins.shape}")
    print(f"True original positions: {true_labels}")
    
    # Predict
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(scrambled_bins).unsqueeze(0).to(config.device)
        logits = model(input_tensor)
        predictions = torch.argmax(logits, dim=2).cpu().numpy()[0]
    
    print(f"Predicted positions: {predictions}")
    print(f"Correct predictions: {np.sum(predictions == true_labels)}/{config.num_bins}")
    print()
    
    print("=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
