# =====================
# ✅ Modified `dashboard.py` with Manual Stats Support
# =====================

#!/usr/bin/env python3
"""
Real-time ECG Arrhythmia Monitoring Dashboard
"""

import os
import sys
import time
import threading
import json
import numpy as np
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import plotly.graph_objs as go
import plotly.utils
from collections import deque
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from inference.realtime_inference import ECGInferenceEngine, ECGSimulator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__, template_folder='../../templates', static_folder='../../static')
app.config['SECRET_KEY'] = 'ecg_monitoring_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
inference_engine = None
simulator = None
monitoring_active = False
monitoring_thread = None

# Data storage for dashboard
ecg_data = deque(maxlen=1000)
prediction_data = deque(maxlen=100)
alert_data = deque(maxlen=50)
manual_stats_data = deque(maxlen=50)

stats_data = {
    'total_predictions': 0,
    'normal_count': 0,
    'abnormal_count': 0,
    'alerts_triggered': 0,
    'session_start': time.time()
}

class ECGMonitoringDashboard:
    def __init__(self):
        self.class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unclassifiable']
        self.class_colors = {
            'Normal': '#28a745',
            'Supraventricular': '#ffc107', 
            'Ventricular': '#dc3545',
            'Fusion': '#fd7e14',
            'Unclassifiable': '#6c757d'
        }

    def initialize_engine(self, model_path="models/trained_models/cnn_quick.pth"):
        global inference_engine, simulator
        try:
            inference_engine = ECGInferenceEngine(model_path)
            simulator = ECGSimulator()
            logger.info("✅ Inference engine initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Error initializing inference engine: {str(e)}")
            return False

    def start_monitoring(self):
        global monitoring_active, monitoring_thread
        if not inference_engine:
            return False
        monitoring_active = True
        monitoring_thread = threading.Thread(target=self._monitoring_loop)
        monitoring_thread.daemon = True
        monitoring_thread.start()
        self._emit_stats_realtime()
        logger.info("🎯 Real-time monitoring started")
        return True

    def stop_monitoring(self):
        global monitoring_active
        monitoring_active = False
        logger.info("⏹️ Real-time monitoring stopped")

    def _monitoring_loop(self):
        global ecg_data, prediction_data, alert_data, stats_data
        while monitoring_active:
            try:
                signal_chunk = simulator.get_next_signal(length=360)
                timestamp = time.time()
                for i, value in enumerate(signal_chunk):
                    ecg_data.append({
                        'time': timestamp + i/360,
                        'value': value,
                        'sample_index': len(ecg_data)
                    })
                predictions = inference_engine.process_signal_chunk(signal_chunk)
                for prediction in predictions:
                    prediction_data.append({
                        'timestamp': timestamp,
                        'class': prediction['class'],
                        'class_name': prediction['class_name'],
                        'confidence': prediction['confidence'],
                        'alert': prediction.get('alert', False)
                    })
                    stats_data['total_predictions'] += 1
                    if prediction['class'] == 0:
                        stats_data['normal_count'] += 1
                    else:
                        stats_data['abnormal_count'] += 1
                    if prediction.get('alert', False):
                        alert_data.append({
                            'timestamp': timestamp,
                            'class_name': prediction['class_name'],
                            'confidence': prediction['confidence'],
                            'message': f"⚠️ {prediction['class_name']} detected!"
                        })
                        stats_data['alerts_triggered'] += 1
                        socketio.emit('alert', {
                            'class_name': prediction['class_name'],
                            'confidence': prediction['confidence'],
                            'timestamp': datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
                        })
                socketio.emit('ecg_update', {
                    'ecg_data': list(ecg_data)[-100:],
                    'predictions': list(prediction_data)[-10:],
                    'stats': self._get_current_stats()
                })
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(1)

    def _emit_stats_realtime(self):
        def loop():
            while monitoring_active:
                socketio.emit('stat_update', self._get_current_stats())
                time.sleep(1)
        threading.Thread(target=loop, daemon=True).start()

    def _get_current_stats(self):
        runtime = time.time() - stats_data['session_start']
        return {
            'total_predictions': stats_data['total_predictions'],
            'normal_count': stats_data['normal_count'],
            'abnormal_count': stats_data['abnormal_count'],
            'alerts_triggered': stats_data['alerts_triggered'],
            'runtime': runtime,
            'predictions_per_minute': (stats_data['total_predictions'] / runtime * 60) if runtime > 0 else 0,
            'normal_percentage': (stats_data['normal_count'] / stats_data['total_predictions'] * 100) if stats_data['total_predictions'] > 0 else 0
        }

# ----------------------
# Flask API Endpoints
# ----------------------
# ----------------------
# Flask API Endpoints
# ----------------------

@socketio.on('start_monitoring')
def handle_start_monitoring():
    logger.info("📡 Received request to start monitoring via socket.")
    dashboard = ECGMonitoringDashboard()
    if dashboard.initialize_engine():
        dashboard.start_monitoring()
        emit('monitoring_started', {'status': 'ok'})
    else:
        emit('monitoring_started', {'status': 'error'})


@app.route('/api/manual_stats', methods=['POST'])
def add_manual_stats():
    """Receive manual stats from user input"""
    try:
        data = request.get_json()
        data['timestamp'] = datetime.now().strftime('%H:%M:%S')
        manual_stats_data.append(data)
        socketio.emit('manual_stats_update', data)
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Failed to receive manual stats: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/')
def index():
    return render_template('dashboard.html')
