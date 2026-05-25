import os
import numpy as np
import pandas as pd
import wfdb
from collections import Counter

class MITBIHLoader:
    def __init__(self, data_path="data/raw/mit-bih"):
        # Updated path to match your structure
        self.records_path = os.path.join(data_path, "records")
        self.annotations_path = os.path.join(data_path, "annotations") 
        
        self.aami_mapping = {
            # Normal
            'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
            # Supraventricular  
            'A': 1, 'a': 1, 'J': 1, 'S': 1,
            # Ventricular
            'V': 2, 'E': 2,
            # Fusion
            'F': 3,
            # Unknown/Other
            '/': 4, 'f': 4, 'Q': 4, '?': 4
        }
        self.class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
    
    def load_record(self, record_name):
        """Load a single MIT-BIH record"""
        try:
            # Try different path combinations
            record_paths = [
                os.path.join(self.records_path, record_name),
                os.path.join("data/raw/mit-bih", record_name),
                os.path.join("data/raw/mit-bih/records", record_name)
            ]
            
            record = None
            annotation = None
            
            for path in record_paths:
                try:
                    record = wfdb.rdrecord(path)
                    annotation = wfdb.rdann(path, 'atr')
                    break
                except:
                    continue
            
            if record is None:
                return None, None, None
                
            # Extract ECG signal (usually lead II)
            ecg_signal = record.p_signal[:, 0]  # First channel
            
            # Get annotations
            beat_locations = annotation.sample
            beat_labels = annotation.symbol
            
            return ecg_signal, beat_locations, beat_labels
            
        except Exception as e:
            print(f"Error loading record {record_name}: {e}")
            return None, None, None
    
    def get_all_records(self):
        """Get list of all available records"""
        records = []
        
        # Check what files actually exist
        if os.path.exists(self.records_path):
            files = os.listdir(self.records_path)
            # Extract record numbers from .dat files
            for file in files:
                if file.endswith('.dat'):
                    record_name = file.replace('.dat', '')
                    records.append(record_name)
        
        print(f"🔍 Found files in {self.records_path}: {len(records)} records")
        print(f"📋 Sample records: {records[:5] if records else 'None'}")
        
        return records
    
    def map_to_aami_classes(self, beat_labels):
        """Map beat labels to AAMI 5-class system"""
        mapped_labels = []
        for label in beat_labels:
            if label in self.aami_mapping:
                mapped_labels.append(self.aami_mapping[label])
            else:
                mapped_labels.append(4)  # Unknown class
        return np.array(mapped_labels)
    
    def analyze_class_distribution(self, all_labels):
        """Analyze class distribution for imbalance handling"""
        if len(all_labels) == 0:
            print("❌ No labels found - check data loading!")
            return {}
            
        counter = Counter(all_labels)
        total = len(all_labels)
        
        print("Class Distribution:")
        for i, class_name in enumerate(self.class_names):
            count = counter.get(i, 0)
            percentage = (count / total) * 100
            print(f"{class_name}: {count} ({percentage:.1f}%)")
        
        return counter