import numpy as np
from sklearn.model_selection import train_test_split

class PatientWiseSplitter:
    def __init__(self, test_size=0.15, val_size=0.15, random_state=42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
    
    def split_patients(self, patient_records):
        """Split patients into train/val/test sets"""
        patients = list(patient_records.keys())
        
        # First split: train+val vs test
        train_val_patients, test_patients = train_test_split(
            patients, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = self.val_size / (1 - self.test_size)
        train_patients, val_patients = train_test_split(
            train_val_patients,
            test_size=val_size_adjusted,
            random_state=self.random_state
        )
        
        return {
            'train': train_patients,
            'val': val_patients, 
            'test': test_patients
        }
    
    def create_datasets(self, patient_data, patient_splits):
        """Create train/val/test datasets from patient splits"""
        datasets = {'train': [], 'val': [], 'test': []}
        
        for split_name, patients in patient_splits.items():
            X_split = []
            y_split = []
            
            for patient in patients:
                if patient in patient_data:
                    beats, labels = patient_data[patient]
                    X_split.extend(beats)
                    y_split.extend(labels)
            
            datasets[split_name] = (np.array(X_split), np.array(y_split))
        
        return datasets