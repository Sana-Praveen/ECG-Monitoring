#!/usr/bin/env python3
"""
Training Module for ECG Arrhythmia Classification Models
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import our models
import sys
sys.path.append('src')
from models.ecg_models import get_model, count_parameters

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ECGTrainer:
    """
    Trainer class for ECG Arrhythmia Classification
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the trainer with configuration"""
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        self.model_dir = Path(self.config['paths']['model_save_path'])
        self.checkpoint_dir = Path(self.config['paths']['checkpoint_path'])
        self.logs_dir = Path(self.config['paths']['logs_path'])
        
        for directory in [self.model_dir, self.checkpoint_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.batch_size = self.config['training']['batch_size']
        self.epochs = self.config['training']['epochs']
        self.learning_rate = self.config['training']['learning_rate']
        self.patience = self.config['training']['patience']
        self.validation_split = self.config['training']['validation_split']
        self.test_split = self.config['training']['test_split']
        
        # Class names
        self.class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unclassifiable']
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        logger.info(f"Training on device: {self.device}")
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load processed ECG data"""
        data_path = Path(self.config['data']['processed_data_path'])
        
        # Load numpy arrays
        X = np.load(data_path / 'X_data.npy')
        y = np.load(data_path / 'y_data.npy')
        
        logger.info(f"Data loaded: X shape={X.shape}, y shape={y.shape}")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data loaders for training, validation, and testing"""
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_split, random_state=42, stratify=y
        )
        
        # Second split: separate train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=self.validation_split/(1-self.test_split), 
            random_state=42, stratify=y_temp
        )
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train).unsqueeze(1)  # Add channel dimension
        X_val = torch.FloatTensor(X_val).unsqueeze(1)
        X_test = torch.FloatTensor(X_test).unsqueeze(1)
        
        y_train = torch.LongTensor(y_train)
        y_val = torch.LongTensor(y_val)
        y_test = torch.LongTensor(y_test)
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, 
                               shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, 
                                shuffle=False, num_workers=0)
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
        
        # Store test loader for later evaluation
        self.test_loader = test_loader
        
        return train_loader, val_loader, test_loader
    
    def get_class_weights(self, y: np.ndarray) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset"""
        if self.config['training']['class_weights'] == 'balanced':
            classes = np.unique(y)
            class_weights = compute_class_weight('balanced', classes=classes, y=y)
            return torch.FloatTensor(class_weights).to(self.device)
        else:
            return None
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
        """Train model for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': running_loss / (batch_idx + 1),
                'Acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                      criterion: nn.Module) -> Tuple[float, float]:
        """Validate model for one epoch"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train_model(self, model_name: str) -> nn.Module:
        """Train a specific model"""
        
        # Load and prepare data
        X, y = self.load_data()
        train_loader, val_loader, test_loader = self.prepare_data(X, y)
        
        # Create model
        model = get_model(model_name, self.config)
        model = model.to(self.device)
        
        # Model summary
        logger.info(f"Model: {model_name}")
        logger.info(f"Parameters: {count_parameters(model):,}")
        
        # Loss function and optimizer
        class_weights = self.get_class_weights(y)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info("Starting training...")
        
        for epoch in range(self.epochs):
            # Training phase
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch(model, val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Logging
            logger.info(f'Epoch {epoch+1}/{self.epochs}:')
            logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            logger.info(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'config': self.config
                }, self.checkpoint_dir / f'best_{model_name.lower()}.pth')
                logger.info(f'  New best model saved!')
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                logger.info(f'Early stopping after {epoch+1} epochs')
                break
        
        # Load best model
        checkpoint = torch.load(self.checkpoint_dir / f'best_{model_name.lower()}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Save final model
        torch.save(model.state_dict(), self.model_dir / f'{model_name.lower()}_final.pth')
        
        # Save training history
        with open(self.logs_dir / f'{model_name.lower()}_history.pkl', 'wb') as f:
            pickle.dump(self.history, f)
        
        logger.info(f"Training completed for {model_name}")
        
        return model
    
    def evaluate_model(self, model: nn.Module, model_name: str) -> Dict:
        """Evaluate model performance on test set"""
        model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        class_report = classification_report(all_targets, all_predictions, 
                                           target_names=self.class_names,
                                           output_dict=True)
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        
        # Save results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix
        }
        
        with open(self.logs_dir / f'{model_name.lower()}_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        # Print results
        logger.info(f"\n=== {model_name} Test Results ===")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(all_targets, all_predictions, 
                                        target_names=self.class_names))
        
        return results
    
    def plot_training_history(self, model_name: str):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name} - Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.title(f'{model_name} - Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.logs_dir / f'{model_name.lower()}_training_history.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, conf_matrix: np.ndarray, model_name: str):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(self.logs_dir / f'{model_name.lower()}_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def train_and_evaluate(self, model_name: str) -> Dict:
        """Complete training and evaluation pipeline"""
        # Reset history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Train model
        model = self.train_model(model_name)
        
        # Evaluate model
        results = self.evaluate_model(model, model_name)
        
        # Plot results
        self.plot_training_history(model_name)
        self.plot_confusion_matrix(results['confusion_matrix'], model_name)
        
        return results
    
    def compare_models(self, model_names: List[str]) -> pd.DataFrame:
        """Train and compare multiple models"""
        results = []
        
        for model_name in model_names:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_name}")
            logger.info(f"{'='*50}")
            
            try:
                result = self.train_and_evaluate(model_name)
                results.append({
                    'Model': model_name,
                    'Accuracy': result['accuracy'],
                    'Precision': result['classification_report']['weighted avg']['precision'],
                    'Recall': result['classification_report']['weighted avg']['recall'],
                    'F1-Score': result['classification_report']['weighted avg']['f1-score']
                })
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                results.append({
                    'Model': model_name,
                    'Accuracy': 0.0,
                    'Precision': 0.0,
                    'Recall': 0.0,
                    'F1-Score': 0.0
                })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        # Save comparison
        comparison_df.to_csv(self.logs_dir / 'model_comparison.csv', index=False)
        
        # Print comparison
        logger.info("\n=== Model Comparison ===")
        logger.info(comparison_df.to_string(index=False))
        
        return comparison_df

def main():
    """Main training function"""
    trainer = ECGTrainer()
    
    # Models to train
    models_to_train = ['CNN', 'CNN_LSTM', 'ResNet1D']
    
    # Train and compare models
    comparison_results = trainer.compare_models(models_to_train)
    
    print("\n=== Training Complete ===")
    print("Check the logs directory for detailed results and plots.")
    
if __name__ == "__main__":
    main()
