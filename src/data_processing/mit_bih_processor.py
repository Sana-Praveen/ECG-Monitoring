#!/usr/bin/env python3
"""
MIT-BIH Arrhythmia Database Processing Module
"""

import os
import numpy as np
import pandas as pd
import wfdb
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import signal as sp_signal
import yaml
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MITBIHProcessor:
    """
    Process MIT-BIH Arrhythmia Database for deep learning
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the processor with configuration"""
        self.config = self._load_config(config_path)
        self.fs = self.config['data']['sampling_rate']
        self.window_size = self.config['data']['window_size']
        self.overlap = self.config['data']['overlap']
        self.annotation_mapping = self.config['annotation_mapping']
        self.classes = self.config['classes']
        
        # Create output directories
        self.processed_dir = Path(self.config['data']['processed_data_path'])
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # MIT-BIH record names
        self.record_names = [
            '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
            '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
            '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
            '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
            '222', '223', '228', '230', '231', '232', '233', '234'
        ]
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def download_and_process_record(self, record_name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Download and process a single MIT-BIH record
        
        Args:
            record_name: Record identifier (e.g., '100')
            
        Returns:
            Tuple of (signals, annotations) or None if failed
        """
        try:
            # Download record from PhysioNet
            record = wfdb.rdrecord(record_name, pn_dir='mitdb')
            
            # Download annotations
            annotation = wfdb.rdann(record_name, 'atr', pn_dir='mitdb')
            
            # Extract ECG signals (use lead II - index 0)
            ecg_signal = record.p_signal[:, 0]  # Lead II
            
            return ecg_signal, annotation
            
        except Exception as e:
            logger.error(f"Failed to download record {record_name}: {str(e)}")
            return None
    
    def preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Preprocess ECG signal with filtering and normalization
        
        Args:
            signal: Raw ECG signal
            
        Returns:
            Preprocessed signal
        """
        # Bandpass filter
        if self.config['preprocessing']['filter_type'] == 'bandpass':
            low_freq = self.config['preprocessing']['low_freq']
            high_freq = self.config['preprocessing']['high_freq']
            
            # Design bandpass filter
            nyquist = self.fs / 2
            low = low_freq / nyquist
            high = high_freq / nyquist
            
            b, a = sp_signal.butter(4, [low, high], btype='band')
            filtered_signal = sp_signal.filtfilt(b, a, signal)
        else:
            filtered_signal = signal
        
        # Notch filter for power line interference
        if self.config['preprocessing']['notch_freq']:
            notch_freq = self.config['preprocessing']['notch_freq']
            Q = 30  # Quality factor
            
            b, a = sp_signal.iirnotch(notch_freq, Q, self.fs)
            filtered_signal = sp_signal.filtfilt(b, a, filtered_signal)
        
        # Remove baseline wander
        if self.config['preprocessing']['remove_baseline']:
            # High-pass filter to remove baseline
            b, a = sp_signal.butter(4, 0.5/(self.fs/2), btype='high')
            filtered_signal = sp_signal.filtfilt(b, a, filtered_signal)
        
        # Normalize
        if self.config['preprocessing']['normalize']:
            filtered_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
        
        return filtered_signal
    
    def map_annotations(self, annotations) -> List[str]:
        """
        Map MIT-BIH annotations to AAMI standard classes
        
        Args:
            annotations: wfdb annotation object
            
        Returns:
            List of mapped annotation labels
        """
        mapped_labels = []
        
        for symbol in annotations.symbol:
            if symbol in self.annotation_mapping:
                mapped_labels.append(self.annotation_mapping[symbol])
            else:
                mapped_labels.append('Q')  # Unknown -> Unclassifiable
        
        return mapped_labels
    
    def segment_signal(self, signal: np.ndarray, annotations, 
                      annotation_labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment ECG signal into fixed-length windows around R-peaks
        
        Args:
            signal: Preprocessed ECG signal
            annotations: wfdb annotation object
            annotation_labels: Mapped annotation labels
            
        Returns:
            Tuple of (segments, labels)
        """
        segments = []
        labels = []
        
        # Calculate window parameters
        half_window = self.window_size // 2
        
        for i, (sample, label) in enumerate(zip(annotations.sample, annotation_labels)):
            # Skip if too close to beginning or end
            if sample < half_window or sample >= len(signal) - half_window:
                continue
            
            # Extract segment around R-peak
            segment = signal[sample - half_window:sample + half_window]
            
            if len(segment) == self.window_size:
                segments.append(segment)
                labels.append(self.classes[label])
        
        return np.array(segments), np.array(labels)
    
    def process_all_records(self) -> Dict[str, any]:
        """
        Process all MIT-BIH records and create training dataset
        
        Returns:
            Dictionary with processed data statistics
        """
        all_segments = []
        all_labels = []
        record_stats = {}
        
        logger.info("Processing all MIT-BIH records...")
        
        for record_name in tqdm(self.record_names, desc="Processing records"):
            # Download and process record
            result = self.download_and_process_record(record_name)
            
            if result is None:
                continue
                
            signal, annotations = result
            
            # Preprocess signal
            processed_signal = self.preprocess_signal(signal)
            
            # Map annotations
            mapped_labels = self.map_annotations(annotations)
            
            # Segment signal
            segments, labels = self.segment_signal(processed_signal, annotations, mapped_labels)
            
            if len(segments) > 0:
                all_segments.extend(segments)
                all_labels.extend(labels)
                
                # Track statistics
                record_stats[record_name] = {
                    'num_segments': len(segments),
                    'signal_length': len(signal),
                    'label_distribution': {str(label): int(count) for label, count in 
                                         zip(*np.unique(labels, return_counts=True))}
                }
                
                logger.info(f"Record {record_name}: {len(segments)} segments extracted")
        
        # Convert to numpy arrays
        X = np.array(all_segments)
        y = np.array(all_labels)
        
        # Save processed data
        self._save_processed_data(X, y, record_stats)
        
        # Generate statistics
        stats = self._generate_statistics(X, y, record_stats)
        
        return stats
    
    def _save_processed_data(self, X: np.ndarray, y: np.ndarray, 
                           record_stats: Dict) -> None:
        """Save processed data to files"""
        
        # Save as numpy arrays
        np.save(self.processed_dir / 'X_data.npy', X)
        np.save(self.processed_dir / 'y_data.npy', y)
        
        # Save as pickle for convenience
        with open(self.processed_dir / 'processed_data.pkl', 'wb') as f:
            pickle.dump({
                'X': X,
                'y': y,
                'record_stats': record_stats,
                'config': self.config
            }, f)
        
        logger.info(f"Processed data saved to {self.processed_dir}")
        logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    
    def _generate_statistics(self, X: np.ndarray, y: np.ndarray, 
                           record_stats: Dict) -> Dict:
        """Generate dataset statistics"""
        
        # Overall statistics
        unique_labels, label_counts = np.unique(y, return_counts=True)
        
        stats = {
            'total_segments': len(X),
            'signal_length': X.shape[1],
            'num_classes': len(unique_labels),
            'class_distribution': {
                f"Class_{int(label)}": int(count) 
                for label, count in zip(unique_labels, label_counts)
            },
            'records_processed': len(record_stats),
            'per_record_stats': record_stats
        }
        
        # Save statistics
        with open(self.processed_dir / 'dataset_stats.yaml', 'w') as f:
            yaml.dump(stats, f, default_flow_style=False)
        
        return stats
    
    def load_processed_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load previously processed data"""
        X = np.load(self.processed_dir / 'X_data.npy')
        y = np.load(self.processed_dir / 'y_data.npy')
        
        return X, y

def main():
    """Main processing function"""
    processor = MITBIHProcessor()
    
    # Process all records
    stats = processor.process_all_records()
    
    # Print statistics
    print("\\n" + "="*60)
    print("MIT-BIH PROCESSING COMPLETED")
    print("="*60)
    print(f"Total segments extracted: {stats['total_segments']}")
    print(f"Records processed: {stats['records_processed']}")
    print(f"Signal length: {stats['signal_length']}")
    print(f"Number of classes: {stats['num_classes']}")
    print("\\nClass distribution:")
    for class_name, count in stats['class_distribution'].items():
        print(f"  {class_name}: {count}")
    
    print(f"\\nProcessed data saved to: {Path('data/processed').absolute()}")

if __name__ == "__main__":
    main()
