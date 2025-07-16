#!/usr/bin/env python3
"""
Deep Learning Models for ECG Arrhythmia Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network for ECG classification
    """
    
    def __init__(self, input_length: int = 1080, num_classes: int = 5, 
                 dropout_rate: float = 0.3):
        super(CNN1D, self).__init__()
        
        self.input_length = input_length
        self.num_classes = num_classes
        
        # First convolutional block
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        # Second convolutional block
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        # Third convolutional block
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        
        # Fourth convolutional block
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(2)
        
        # Calculate the size after convolutions
        self.feature_size = self._calculate_conv_output_size()
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.feature_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def _calculate_conv_output_size(self):
        """Calculate the output size after convolutions"""
        size = self.input_length
        size = size // 2  # pool1
        size = size // 2  # pool2
        size = size // 2  # pool3
        size = size // 2  # pool4
        return size * 256
        
    def forward(self, x):
        # Input shape: (batch_size, 1, sequence_length)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

class BiLSTM(nn.Module):
    """
    Bidirectional LSTM for ECG classification
    """
    
    def __init__(self, input_size: int = 1, hidden_size: int = 64, 
                 num_layers: int = 2, num_classes: int = 5, 
                 dropout_rate: float = 0.3):
        super(BiLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True, 
                           dropout=dropout_rate if num_layers > 1 else 0)
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size * 2, 128)  # *2 for bidirectional
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)
        if len(x.shape) == 3 and x.shape[1] == 1:
            x = x.permute(0, 2, 1)  # (batch_size, sequence_length, 1)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(last_output)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

class CNN_LSTM(nn.Module):
    """
    Hybrid CNN-LSTM model for ECG classification
    """
    
    def __init__(self, input_length: int = 1080, num_classes: int = 5,
                 cnn_filters: int = 64, lstm_hidden: int = 64,
                 dropout_rate: float = 0.3):
        super(CNN_LSTM, self).__init__()
        
        # CNN feature extractor
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, cnn_filters, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(cnn_filters)
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate sequence length after CNN
        self.cnn_output_length = input_length // 8  # 3 pooling operations
        
        # LSTM layer
        self.lstm = nn.LSTM(cnn_filters, lstm_hidden, batch_first=True, 
                           bidirectional=True, dropout=dropout_rate)
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(lstm_hidden * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Prepare for LSTM: (batch_size, sequence_length, features)
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(last_output)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

class ResNet1D(nn.Module):
    """
    1D ResNet for ECG classification
    """
    
    def __init__(self, input_length: int = 1080, num_classes: int = 5,
                 dropout_rate: float = 0.3):
        super(ResNet1D, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """Create a residual layer"""
        layers = []
        
        # First block (may have stride > 1)
        layers.append(ResidualBlock1D(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
            
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class ResidualBlock1D(nn.Module):
    """
    1D Residual Block
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class TransformerModel(nn.Module):
    """
    Transformer model for ECG classification
    """
    
    def __init__(self, input_length: int = 1080, num_classes: int = 5,
                 d_model: int = 256, nhead: int = 8, num_layers: int = 6,
                 dropout_rate: float = 0.3):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.input_length = input_length
        
        # Input projection
        self.input_projection = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, input_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Input shape: (batch_size, 1, sequence_length)
        x = x.permute(0, 2, 1)  # (batch_size, sequence_length, 1)
        
        # Project to d_model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    """
    
    def __init__(self, d_model: int, max_length: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)

def get_model(model_name: str, config: Dict) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_name: Name of the model architecture
        config: Configuration dictionary
        
    Returns:
        PyTorch model
    """
    input_length = config['model']['input_shape'][0]
    num_classes = config['model']['num_classes']
    dropout_rate = config['model']['dropout_rate']
    
    if model_name.upper() == 'CNN':
        return CNN1D(input_length, num_classes, dropout_rate)
    elif model_name.upper() == 'LSTM':
        hidden_units = config['model']['lstm_units']
        return BiLSTM(1, hidden_units, 2, num_classes, dropout_rate)
    elif model_name.upper() == 'CNN_LSTM':
        hidden_units = config['model']['hidden_units']
        lstm_units = config['model']['lstm_units']
        return CNN_LSTM(input_length, num_classes, hidden_units, 
                       lstm_units, dropout_rate)
    elif model_name.upper() == 'RESNET1D':
        return ResNet1D(input_length, num_classes, dropout_rate)
    elif model_name.upper() == 'TRANSFORMER':
        return TransformerModel(input_length, num_classes, 
                               dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model: nn.Module, input_shape: Tuple[int, ...]) -> None:
    """Print model summary"""
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Input shape: {input_shape}")
    print("-" * 50)
