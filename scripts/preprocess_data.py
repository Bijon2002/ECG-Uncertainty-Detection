#!/usr/bin/env python3
"""
MIT-BIH Data Preprocessing Script
"""
import os
import sys
import numpy as np
import pickle
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import ECGPreprocessor
from src.data.mit_bih_loader import MITBIHLoader
from src.data.dataset_splitter import PatientWiseSplitter

def main():
    print("🚀 Starting MIT-BIH Data Preprocessing...")
    
    # Initialize components
    loader = MITBIHLoader()
    preprocessor = ECGPreprocessor()
    splitter = PatientWiseSplitter()
    
    # Get all available records
    records = loader.get_all_records()
    print(f"📊 Found {len(records)} MIT-BIH records")
    
    # Process each patient
    patient_data = {}
    all_labels = []
    
    print("🔄 Processing ECG records...")
    for record in tqdm(records):
        # Load record
        ecg_signal, beat_locations, beat_labels = loader.load_record(record)
        
        if ecg_signal is not None:
            # Preprocess ECG signal
            filtered_signal = preprocessor.bandpass_filter(ecg_signal)
            r_peaks = preprocessor.detect_r_peaks(filtered_signal)
            
            # Extract beats
            beats = preprocessor.extract_beats(filtered_signal, r_peaks)
            
            if len(beats) > 0:
                # Normalize beats
                normalized_beats = preprocessor.normalize_beats(beats)
                
                # Map labels to AAMI classes
                valid_labels = []
                valid_beats = []
                
                for i, r_peak in enumerate(r_peaks):
                    # Find closest annotation
                    closest_idx = np.argmin(np.abs(beat_locations - r_peak))
                    if np.abs(beat_locations[closest_idx] - r_peak) < 50:  # Within tolerance
                        label = beat_labels[closest_idx]
                        aami_label = loader.aami_mapping.get(label, 4)
                        
                        if i < len(normalized_beats):
                            valid_beats.append(normalized_beats[i])
                            valid_labels.append(aami_label)
                            all_labels.append(aami_label)
                
                if len(valid_beats) > 0:
                    patient_data[record] = (valid_beats, valid_labels)
    
    print(f"✅ Processed {len(patient_data)} patients successfully")
    
    # Analyze class distribution
    loader.analyze_class_distribution(all_labels)
    
    # Split patients
    print("🔄 Creating patient-wise splits...")
    patient_splits = splitter.split_patients(patient_data)
    
    print(f"📊 Split Summary:")
    print(f"  Train: {len(patient_splits['train'])} patients")
    print(f"  Val: {len(patient_splits['val'])} patients") 
    print(f"  Test: {len(patient_splits['test'])} patients")
    
    # Create datasets
    datasets = splitter.create_datasets(patient_data, patient_splits)
    
    # Print dataset statistics
    for split_name, (X, y) in datasets.items():
        print(f"  {split_name.capitalize()}: {len(X)} beats, shape: {X.shape}")
    
    # Save processed data
    print("💾 Saving processed datasets...")
    os.makedirs("data/processed", exist_ok=True)
    
    with open("data/processed/datasets.pkl", "wb") as f:
        pickle.dump(datasets, f)
    
    with open("data/processed/patient_splits.pkl", "wb") as f:
        pickle.dump(patient_splits, f)
    
    print("🎉 Preprocessing completed successfully!")
    print("📁 Saved files:")
    print("  - data/processed/datasets.pkl")
    print("  - data/processed/patient_splits.pkl")

if __name__ == "__main__":
    main()