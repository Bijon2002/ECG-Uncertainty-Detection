"""
MIT-BIH Arrhythmia Database Download Script
Downloads and organizes the MIT-BIH dataset from PhysioNet using WFDB.
"""

import os
from datetime import datetime
import wfdb
from tqdm import tqdm
import yaml


RAW_ROOT = "data/raw/mit-bih"
RECORDS_DIR = os.path.join(RAW_ROOT, "records")
META_DIR = os.path.join(RAW_ROOT, "metadata")


def create_data_directories():
    """Create necessary data directories"""
    directories = [
        RECORDS_DIR,
        os.path.join(RAW_ROOT, "annotations"),  # optional (WFDB keeps .atr with records)
        META_DIR,
        "data/processed/train",
        "data/processed/validation",
        "data/processed/test",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def create_dataset_info():
    """Create dataset information file"""
    dataset_info = {
        "dataset": "MIT-BIH Arrhythmia Database",
        "source": "PhysioNet",
        "total_records": 48,
        "sampling_rate": 360,
        "duration_per_record": 30,
        "lead_count": 2,
        "download_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": "Files downloaded via wfdb.dl_database(mitdb).",
    }

    os.makedirs(META_DIR, exist_ok=True)
    out_path = os.path.join(META_DIR, "dataset_info.yaml")
    with open(out_path, "w") as f:
        yaml.dump(dataset_info, f, default_flow_style=False)

    print(f"✅ Created dataset information file: {out_path}")


def download_mit_bih_database():
    """Download MIT-BIH Arrhythmia Database"""

    record_list = [
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112,
        113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200,
        201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215,
        217, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234
    ]

    print("🔄 Downloading MIT-BIH Arrhythmia Database...")

    downloaded_records = []
    failed_records = []

    for record_num in tqdm(record_list, desc="Downloading records"):
        record_name = str(record_num)
        try:
            # downloads record_name.{dat,hea,atr} into RECORDS_DIR
            wfdb.dl_database("mitdb", dl_dir=RECORDS_DIR, records=[record_name])
            downloaded_records.append(record_num)
        except Exception as e:
            print(f"❌ Failed to download record {record_num}: {e}")
            failed_records.append(record_num)

    print(f"\n✅ Successfully downloaded {len(downloaded_records)} records")
    if failed_records:
        print(f"❌ Failed to download {len(failed_records)} records: {failed_records}")

    return downloaded_records, failed_records


def verify_download():
    """Verify that files were downloaded correctly"""
    dat_files = [f for f in os.listdir(RECORDS_DIR) if f.endswith(".dat")]
    hea_files = [f for f in os.listdir(RECORDS_DIR) if f.endswith(".hea")]
    atr_files = [f for f in os.listdir(RECORDS_DIR) if f.endswith(".atr")]

    print("\n📊 Download Verification:")
    print(f"Signal files (.dat): {len(dat_files)}")
    print(f"Header files (.hea): {len(hea_files)}")
    print(f"Annotation files (.atr): {len(atr_files)}")

    if len(dat_files) == len(hea_files) == len(atr_files) and len(dat_files) > 0:
        print("✅ All file types match - download appears complete!")
    else:
        print("⚠️ File count mismatch - some files may be missing")

    return len(dat_files), len(hea_files), len(atr_files)


def main():
    print("🚀 Starting MIT-BIH Database Download Process")
    print("=" * 50)

    create_data_directories()
    create_dataset_info()
    download_mit_bih_database()
    verify_download()

    print("\n" + "=" * 50)
    print("✅ Data acquisition complete!")
    print(f"📂 Data location: {RECORDS_DIR}")
    print("📋 Next step: Run the data exploration notebook")


if __name__ == "__main__":
    main()