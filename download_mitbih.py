#!/usr/bin/env python3
"""
Script to download MIT-BIH Arrhythmia Database
"""

import os
import wfdb
import logging
from pathlib import Path
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_mitbih_database():
    """
    Download MIT-BIH Arrhythmia Database using wfdb library
    """
    # Create data directory
    data_dir = Path("data/mit-bih-arrhythmia-database")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # MIT-BIH Arrhythmia Database record names
    record_names = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
        '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
        '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
        '222', '223', '228', '230', '231', '232', '233', '234'
    ]
    
    logger.info(f"Downloading MIT-BIH Arrhythmia Database to {data_dir}")
    logger.info(f"Total records to download: {len(record_names)}")
    
    successful_downloads = 0
    failed_downloads = []
    
    for record_name in tqdm(record_names, desc="Downloading MIT-BIH records"):
        try:
            # Download record from PhysioNet
            record = wfdb.rdrecord(
                record_name, 
                pn_dir='mitdb'
            )
            
            # Download annotation
            annotation = wfdb.rdann(
                record_name,
                'atr',
                pn_dir='mitdb'
            )
            
            # Save to local directory
            record_path = data_dir / record_name
            wfdb.wrsamp(
                str(record_path),
                fs=record.fs,
                units=record.units,
                sig_name=record.sig_name,
                p_signal=record.p_signal,
                fmt=record.fmt
            )
            
            # Save annotation
            wfdb.wrann(
                str(record_path),
                'atr',
                annotation.sample,
                annotation.symbol,
                annotation.aux_note
            )
            
            successful_downloads += 1
            logger.info(f"Successfully downloaded record {record_name}")
            
        except Exception as e:
            logger.error(f"Failed to download record {record_name}: {str(e)}")
            failed_downloads.append(record_name)
    
    logger.info(f"Download complete!")
    logger.info(f"Successfully downloaded: {successful_downloads}/{len(record_names)} records")
    
    if failed_downloads:
        logger.warning(f"Failed downloads: {failed_downloads}")
    
    return successful_downloads, failed_downloads

def verify_download():
    """
    Verify that the downloaded files are complete
    """
    data_dir = Path("data/mit-bih-arrhythmia-database")
    
    if not data_dir.exists():
        logger.error("Data directory does not exist. Please run download first.")
        return False
    
    # Check for .dat, .hea, and .atr files
    dat_files = list(data_dir.glob("*.dat"))
    hea_files = list(data_dir.glob("*.hea"))
    atr_files = list(data_dir.glob("*.atr"))
    
    logger.info(f"Found {len(dat_files)} .dat files")
    logger.info(f"Found {len(hea_files)} .hea files")
    logger.info(f"Found {len(atr_files)} .atr files")
    
    # Should have 48 of each file type
    expected_count = 48
    if len(dat_files) == expected_count and len(hea_files) == expected_count and len(atr_files) == expected_count:
        logger.info("✅ All files downloaded successfully!")
        return True
    else:
        logger.warning("⚠️  Some files may be missing.")
        return False

if __name__ == "__main__":
    print("MIT-BIH Arrhythmia Database Downloader")
    print("=" * 50)
    
    # Download the database
    successful, failed = download_mitbih_database()
    
    # Verify download
    print("\nVerifying download...")
    verify_download()
    
    print("\nDownload process completed!")
    print(f"Data saved to: {Path('data/mit-bih-arrhythmia-database').absolute()}")
