#!/usr/bin/env python3
"""
Real-time ECG Inference Module for Arrhythmia Detection
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import logging
from typing import Dict, List, Tuple, Optional
import time
from collections import deque
import threading
from scipy import signal as sp_signal
import pickle
import json

# Import our models
import sys
sys.path.append('src')
from models.ecg_models import get_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ECGInferenceEngine:
    """
    Real-time ECG Inference Engine for Arrhythmia Detection
    """
    
    def __init__(self, model_path: str, config_path: str = "config/config.yaml"):
        """
        Initialize the inference engine
        
        Args:
            model_path: Path to the trained model
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # ECG processing parameters
        self.fs = self.config['data']['sampling_rate']
        self.window_size = self.config['data']['window_size']
        self.buffer_size = self.config['realtime']['buffer_size']
        self.alert_threshold = self.config['realtime']['alert_threshold']
        
        # Signal buffer for real-time processing
        self.signal_buffer = deque(maxlen=self.buffer_size)
        
        # Class names
        self.class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unclassifiable']
        
        # Prediction history
        self.prediction_history = deque(maxlen=100)
        self.alert_history = deque(maxlen=50)
        
        # Processing statistics
        self.stats = {
            'total_predictions': 0,
            'normal_count': 0,
            'abnormal_count': 0,
            'alerts_triggered': 0,
            'processing_time': [],
            'start_time': time.time()
        }
        
        logger.info(f"Inference engine initialized on {self.device}")
        logger.info(f"Model loaded: {Path(model_path).name}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained model"""
        # Determine model type from filename
        model_name = Path(model_path).stem.split('_')[0].upper()
        
        # Create model architecture
        model = get_model(model_name, self.config)
        
        # Load state dict
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        return model
    
    def preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Preprocess ECG signal for inference
        
        Args:
            signal: Raw ECG signal
            
        Returns:
            Preprocessed signal
        """
        # Bandpass filter
        if self.config['preprocessing']['filter_type'] == 'bandpass':
            low_freq = self.config['preprocessing']['low_freq']
            high_freq = self.config['preprocessing']['high_freq']
            
            nyquist = self.fs / 2
            low = low_freq / nyquist
            high = high_freq / nyquist
            
            b, a = sp_signal.butter(4, [low, high], btype='band')
            filtered_signal = sp_signal.filtfilt(b, a, signal)
        else:
            filtered_signal = signal
        
        # Notch filter
        if self.config['preprocessing']['notch_freq']:
            notch_freq = self.config['preprocessing']['notch_freq']
            Q = 30
            
            b, a = sp_signal.iirnotch(notch_freq, Q, self.fs)
            filtered_signal = sp_signal.filtfilt(b, a, filtered_signal)
        
        # Remove baseline
        if self.config['preprocessing']['remove_baseline']:
            b, a = sp_signal.butter(4, 0.5/(self.fs/2), btype='high')
            filtered_signal = sp_signal.filtfilt(b, a, filtered_signal)
        
        # Normalize
        if self.config['preprocessing']['normalize']:
            filtered_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
        
        return filtered_signal
    
    def detect_r_peaks(self, signal: np.ndarray, min_distance: int = 150) -> np.ndarray:
        """
        Simple R-peak detection using local maxima
        
        Args:
            signal: ECG signal
            min_distance: Minimum distance between peaks
            
        Returns:
            R-peak indices
        """
        # Find local maxima
        peaks = []
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                peaks.append(i)
        
        # Filter peaks by minimum distance
        if not peaks:
            return np.array([])
        
        filtered_peaks = [peaks[0]]
        for peak in peaks[1:]:
            if peak - filtered_peaks[-1] >= min_distance:
                filtered_peaks.append(peak)
        
        return np.array(filtered_peaks)
    
    def segment_around_peaks(self, signal: np.ndarray, peaks: np.ndarray) -> List[np.ndarray]:
        """
        Segment ECG signal around R-peaks
        
        Args:
            signal: ECG signal
            peaks: R-peak indices
            
        Returns:
            List of ECG segments
        """
        segments = []
        half_window = self.window_size // 2
        
        for peak in peaks:
            if peak >= half_window and peak < len(signal) - half_window:
                segment = signal[peak - half_window:peak + half_window]
                if len(segment) == self.window_size:
                    segments.append(segment)
        
        return segments
    
    def predict_segment(self, segment: np.ndarray) -> Dict:
        """
        Predict arrhythmia class for a single ECG segment
        
        Args:
            segment: ECG segment
            
        Returns:
            Prediction results
        """
        start_time = time.time()
        
        # Prepare input tensor
        input_tensor = torch.FloatTensor(segment).unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        processing_time = time.time() - start_time
        
        result = {
            'class': predicted_class,
            'class_name': self.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy(),
            'processing_time': processing_time
        }
        
        # Update statistics
        self.stats['total_predictions'] += 1
        self.stats['processing_time'].append(processing_time)
        
        if predicted_class == 0:  # Normal
            self.stats['normal_count'] += 1
        else:
            self.stats['abnormal_count'] += 1
        
        # Store prediction history
        self.prediction_history.append(result)
        
        return result
    
    def check_alert_conditions(self, prediction: Dict) -> bool:
        """
        Check if an alert should be triggered
        
        Args:
            prediction: Prediction result
            
        Returns:
            True if alert should be triggered
        """
        # Alert conditions
        alert_triggered = False
        
        # High confidence abnormal rhythm
        if prediction['class'] != 0 and prediction['confidence'] > self.alert_threshold:
            alert_triggered = True
        
        # Multiple consecutive abnormal predictions
        if len(self.prediction_history) >= 3:
            recent_predictions = list(self.prediction_history)[-3:]
            if all(p['class'] != 0 for p in recent_predictions):
                alert_triggered = True
        
        if alert_triggered:
            self.stats['alerts_triggered'] += 1
            alert_info = {
                'timestamp': time.time(),
                'prediction': prediction,
                'type': 'arrhythmia_detected'
            }
            self.alert_history.append(alert_info)
            logger.warning(f"ALERT: {prediction['class_name']} detected with {prediction['confidence']:.2f} confidence")
        
        return alert_triggered
    
    def process_signal_chunk(self, signal_chunk: np.ndarray) -> List[Dict]:
        """
        Process a chunk of ECG signal
        
        Args:
            signal_chunk: Raw ECG signal chunk
            
        Returns:
            List of prediction results
        """
        # Add to buffer
        self.signal_buffer.extend(signal_chunk)
        
        # Only process if we have enough data
        if len(self.signal_buffer) < self.window_size:
            return []
        
        # Get signal from buffer
        signal_array = np.array(self.signal_buffer)
        
        # Preprocess signal
        processed_signal = self.preprocess_signal(signal_array)
        
        # Detect R-peaks
        r_peaks = self.detect_r_peaks(processed_signal)
        
        # Segment around peaks
        segments = self.segment_around_peaks(processed_signal, r_peaks)
        
        # Predict for each segment
        predictions = []
        for segment in segments:
            prediction = self.predict_segment(segment)
            
            # Check for alerts
            alert_triggered = self.check_alert_conditions(prediction)
            prediction['alert'] = alert_triggered
            
            predictions.append(prediction)
        
        return predictions
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        runtime = time.time() - self.stats['start_time']
        avg_processing_time = np.mean(self.stats['processing_time']) if self.stats['processing_time'] else 0
        
        return {
            'runtime_seconds': runtime,
            'total_predictions': self.stats['total_predictions'],
            'normal_count': self.stats['normal_count'],
            'abnormal_count': self.stats['abnormal_count'],
            'alerts_triggered': self.stats['alerts_triggered'],
            'predictions_per_second': self.stats['total_predictions'] / runtime if runtime > 0 else 0,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'normal_percentage': self.stats['normal_count'] / self.stats['total_predictions'] * 100 if self.stats['total_predictions'] > 0 else 0
        }
    
    def get_recent_predictions(self, count: int = 10) -> List[Dict]:
        """Get recent predictions"""
        return list(self.prediction_history)[-count:]
    
    def get_recent_alerts(self, count: int = 5) -> List[Dict]:
        """Get recent alerts"""
        return list(self.alert_history)[-count:]
    
    def reset_statistics(self):
        """Reset processing statistics"""
        self.stats = {
            'total_predictions': 0,
            'normal_count': 0,
            'abnormal_count': 0,
            'alerts_triggered': 0,
            'processing_time': [],
            'start_time': time.time()
        }
        self.prediction_history.clear()
        self.alert_history.clear()
    
    def save_session_data(self, filepath: str):
        """Save session data for analysis"""
        session_data = {
            'statistics': self.get_statistics(),
            'predictions': list(self.prediction_history),
            'alerts': list(self.alert_history),
            'config': self.config,
            'session_end_time': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        logger.info(f"Session data saved to {filepath}")

class ECGSimulator:
    """
    Simulate real-time ECG data for testing
    """
    
    def __init__(self, data_path: str = "data/processed/X_data.npy"):
        """Initialize simulator with processed ECG data"""
        self.data = np.load(data_path)
        self.current_index = 0
        logger.info(f"Simulator initialized with {len(self.data)} ECG segments")
    
    def get_next_signal(self, length: int = 360) -> np.ndarray:
        """Get next ECG signal chunk"""
        if self.current_index >= len(self.data):
            self.current_index = 0
        
        # Get a random segment and extract a chunk
        segment = self.data[self.current_index]
        start_idx = np.random.randint(0, len(segment) - length)
        signal_chunk = segment[start_idx:start_idx + length]
        
        self.current_index += 1
        return signal_chunk
    
    def simulate_realtime_stream(self, inference_engine: ECGInferenceEngine, 
                                duration: int = 60, sample_rate: int = 360):
        """
        Simulate real-time ECG streaming
        
        Args:
            inference_engine: ECG inference engine
            duration: Simulation duration in seconds
            sample_rate: Sampling rate
        """
        chunk_size = sample_rate // 10  # 0.1 second chunks
        total_chunks = duration * 10
        
        logger.info(f"Starting real-time simulation for {duration} seconds")
        
        for i in range(total_chunks):
            # Get signal chunk
            signal_chunk = self.get_next_signal(chunk_size)
            
            # Process through inference engine
            predictions = inference_engine.process_signal_chunk(signal_chunk)
            
            # Print predictions
            for pred in predictions:
                if pred['class'] != 0:  # Only print abnormal
                    logger.info(f"Detected: {pred['class_name']} (confidence: {pred['confidence']:.2f})")
            
            # Sleep to simulate real-time
            time.sleep(0.1)
            
            # Print progress
            if (i + 1) % 50 == 0:
                stats = inference_engine.get_statistics()
                logger.info(f"Progress: {i+1}/{total_chunks} chunks processed, "
                          f"Predictions: {stats['total_predictions']}, "
                          f"Alerts: {stats['alerts_triggered']}")
        
        # Final statistics
        final_stats = inference_engine.get_statistics()
        logger.info("Simulation completed!")
        logger.info(f"Final statistics: {final_stats}")

def main():
    """Main inference testing function"""
    # Initialize inference engine with a trained model
    model_path = "models/trained_models/cnn_final.pth"
    
    try:
        inference_engine = ECGInferenceEngine(model_path)
        
        # Initialize simulator
        simulator = ECGSimulator()
        
        # Run simulation
        simulator.simulate_realtime_stream(inference_engine, duration=30)
        
        # Save session data
        inference_engine.save_session_data("logs/simulation_session.json")
        
    except Exception as e:
        logger.error(f"Error in inference: {str(e)}")
        print("Please make sure you have trained a model first!")

if __name__ == "__main__":
    main()
