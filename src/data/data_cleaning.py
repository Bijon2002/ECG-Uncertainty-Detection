"""
ECG Data Cleaning Script
Cleans and validates MIT-BIH dataset for training
"""

import os
import numpy as np
import pandas as pd
import wfdb
from collections import Counter

class ECGDataCleaner:
    def __init__(self, data_path='data/raw/mit-bih/records'):
        self.data_path = data_path
        self.records = []
        self.cleaning_report = {
            'total_records': 0,
            'valid_records': 0,
            'corrupted_files': [],
            'poor_quality': [],
            'annotation_issues': [],
            'excluded_records': []
        }

    def get_record_list(self):
        """Get list of available records"""
        record_files = [f.replace('.hea', '') for f in os.listdir(self.data_path)
                       if f.endswith('.hea') and f.split('.')[0].isdigit()]
        self.records = sorted(record_files)
        self.cleaning_report['total_records'] = len(self.records)
        print(f"Found {len(self.records)} records to check")
        return self.records

    def check_file_integrity(self):
        """Check if all required files exist and are readable"""
        print("\nChecking file integrity...")

        for record_id in self.records:
            try:
                dat_file = os.path.join(self.data_path, f"{record_id}.dat")
                hea_file = os.path.join(self.data_path, f"{record_id}.hea")
                atr_file = os.path.join(self.data_path, f"{record_id}.atr")

                if not all([os.path.exists(f) for f in [dat_file, hea_file, atr_file]]):
                    self.cleaning_report['corrupted_files'].append(record_id)
                    print(f"Record {record_id}: Missing files")
                    continue

                record = wfdb.rdrecord(os.path.join(self.data_path, record_id))
                annotation = wfdb.rdann(os.path.join(self.data_path, record_id), 'atr')

                if record.sig_len == 0:
                    self.cleaning_report['corrupted_files'].append(record_id)
                    print(f"Record {record_id}: Empty signal")
                    continue

                if len(annotation.sample) == 0:
                    self.cleaning_report['annotation_issues'].append(record_id)
                    print(f"Record {record_id}: No annotations")
                    continue

                print(f"Record {record_id}: OK ({record.sig_len} samples, "
                      f"{len(annotation.sample)} beats)")

            except Exception as e:
                self.cleaning_report['corrupted_files'].append(record_id)
                print(f"Record {record_id}: Error - {str(e)}")


def main():
    """Main cleaning function"""
    print("Starting ECG Data Cleaning Process")
    print("=" * 50)

    cleaner = ECGDataCleaner()
    cleaner.get_record_list()
    cleaner.check_file_integrity()

    print("\n" + "=" * 50)
    print("Data cleaning complete!")

if __name__ == "__main__":
    main()