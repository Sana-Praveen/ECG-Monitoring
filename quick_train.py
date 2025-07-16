#!/usr/bin/env python3
"""
Quick Training Script for ECG Arrhythmia Classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import yaml
from pathlib import Path
import logging
from tqdm import tqdm

# Import our models
import sys
sys.path.append('src')
from models.ecg_models import get_model, count_parameters

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(config):
    """Load processed ECG data"""
    data_path = Path(config['data']['processed_data_path'])
    
    X = np.load(data_path / 'X_data.npy')
    y = np.load(data_path / 'y_data.npy')
    
    logger.info(f"Data loaded: X shape={X.shape}, y shape={y.shape}")
    return X, y

def prepare_data(X, y, config):
    """Prepare data loaders"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train).unsqueeze(1)
    X_val = torch.FloatTensor(X_val).unsqueeze(1)
    X_test = torch.FloatTensor(X_test).unsqueeze(1)
    
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    y_test = torch.LongTensor(y_test)
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), 
                             batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), 
                           batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), 
                            batch_size=batch_size, shuffle=False)
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    return train_loader, val_loader, test_loader, y_train.numpy()

def train_model(model_name="CNN", epochs=10):
    """Train a single model quickly"""
    
    # Load configuration
    config = load_config()
    config['training']['epochs'] = epochs  # Override epochs for quick training
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    # Load and prepare data
    X, y = load_data(config)
    train_loader, val_loader, test_loader, y_train = prepare_data(X, y, config)
    
    # Create model
    model = get_model(model_name, config)
    model = model.to(device)
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Parameters: {count_parameters(model):,}")
    
    # Calculate class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Training loop
    best_val_acc = 0.0
    
    logger.info("Starting training...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            pbar.set_postfix({
                'Loss': train_loss / (batch_idx + 1),
                'Acc': 100. * train_correct / train_total
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        logger.info(f'Epoch {epoch+1}/{epochs}:')\n        logger.info(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')\n        logger.info(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')\n        \n        # Save best model\n        if val_acc > best_val_acc:\n            best_val_acc = val_acc\n            # Create directories if they don't exist\n            Path('models/trained_models').mkdir(parents=True, exist_ok=True)\n            torch.save(model.state_dict(), f'models/trained_models/{model_name.lower()}_quick.pth')\n            logger.info(f'  New best model saved! Val Acc: {val_acc:.2f}%')\n    \n    # Test evaluation\n    logger.info(\"\\nEvaluating on test set...\")\n    model.eval()\n    test_predictions = []\n    test_targets = []\n    \n    with torch.no_grad():\n        for data, target in test_loader:\n            data, target = data.to(device), target.to(device)\n            output = model(data)\n            _, predicted = torch.max(output, 1)\n            \n            test_predictions.extend(predicted.cpu().numpy())\n            test_targets.extend(target.cpu().numpy())\n    \n    # Calculate final metrics\n    test_accuracy = accuracy_score(test_targets, test_predictions)\n    \n    logger.info(f\"\\n=== {model_name} Test Results ===\")\n    logger.info(f\"Test Accuracy: {test_accuracy:.4f}\")\n    \n    class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unclassifiable']\n    logger.info(\"\\nClassification Report:\")\n    logger.info(classification_report(test_targets, test_predictions, \n                                    target_names=class_names))\n    \n    return model, test_accuracy

def main():\n    \"\"\"Main function\"\"\"\n    logger.info(\"=\" * 60)\n    logger.info(\"Quick ECG Arrhythmia Classification Training\")\n    logger.info(\"=\" * 60)\n    \n    # Train CNN model for 10 epochs (quick demonstration)\n    model, accuracy = train_model(\"CNN\", epochs=10)\n    \n    logger.info(f\"\\nTraining completed! Final test accuracy: {accuracy:.4f}\")\n    logger.info(\"Model saved to: models/trained_models/cnn_quick.pth\")\n    \n    # Test inference\n    logger.info(\"\\nTesting inference...\")\n    try:\n        from src.inference.realtime_inference import ECGInferenceEngine\n        inference_engine = ECGInferenceEngine(\"models/trained_models/cnn_quick.pth\")\n        logger.info(\"✅ Inference engine successfully initialized!\")\n        \n        # Test with a sample\n        sample_signal = np.random.randn(1080)  # Random signal for testing\n        prediction = inference_engine.predict_segment(sample_signal)\n        logger.info(f\"Sample prediction: {prediction['class_name']} (confidence: {prediction['confidence']:.2f})\")\n        \n    except Exception as e:\n        logger.error(f\"Error testing inference: {str(e)}\")\n\nif __name__ == \"__main__\":\n    main()
