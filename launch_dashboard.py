#!/usr/bin/env python3
"""
Launch ECG Dashboard
"""
import os
import sys
import webbrowser
import threading
import time

# Add src to path
sys.path.append('src')

def launch_browser():
    """Launch browser after a short delay"""
    time.sleep(3)
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    print("🏥 ECG Arrhythmia Monitor Dashboard")
    print("=" * 50)
    print("Starting the dashboard...")
    print("Dashboard will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Launch browser in a separate thread
    browser_thread = threading.Thread(target=launch_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Import and run the dashboard
    from src.realtime.dashboard import app, socketio
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
