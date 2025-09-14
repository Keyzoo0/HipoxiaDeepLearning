import wfdb
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

def process_clinical_data(dataset_path):
    print("Processing clinical data...")
    
    # Membaca daftar record dari file RECORDS
    with open(os.path.join(dataset_path, 'RECORDS'), 'r') as f:
        records = [line.strip() for line in f.readlines()]
    
    print(f"Total records to process: {len(records)}")
    
    # Membuat dataframe untuk menyimpan informasi klinis
    clinical_data = []
    
    # Memproses semua record untuk mengekstrak informasi klinis
    for i, record_name in enumerate(records):
        try:
            header = wfdb.rdheader(os.path.join(dataset_path, record_name))
            
            # Mengekstrak informasi klinis dari komentar
            record_info = {'record_name': record_name}
            
            for comment in header.comments:
                if not comment.startswith('--') and ':' in comment:
                    key, value = comment.split(':', 1)
                    record_info[key.strip()] = value.strip()
                elif not comment.startswith('--') and len(comment.split()) >= 2:
                    parts = comment.split(None, 1)
                    if len(parts) == 2:
                        key, value = parts
                        try:
                            # Coba konversi ke float jika memungkinkan
                            record_info[key] = float(value)
                        except ValueError:
                            record_info[key] = value
            
            clinical_data.append(record_info)
            
            # Menampilkan progress setiap 50 record
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(records)} records")
                
        except Exception as e:
            print(f"Error processing record {record_name}: {e}")
    
    print(f"Finished processing {len(clinical_data)} records")
    
    # Membuat DataFrame
    df = pd.DataFrame(clinical_data)
    
    # Membuat label klasifikasi berdasarkan pH
    if 'pH' in df.columns:
        df['pH_category'] = df['pH'].apply(lambda x: 
            0 if x >= 7.20 else  # normal
            1 if x >= 7.10 else  # suspect
            2)                   # hypoxia
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Process CTU-UHB CTG database for hypoxia classification')
    parser.add_argument('--dataset_path', type=str, 
                        default='/home/zainul/joki/HipoxiaDeepLearning/dataset/physionet.org/files/ctu-uhb-ctgdb/1.0.0',
                        help='Path to the dataset')
    parser.add_argument('--output_path', type=str,
                        default='/home/zainul/joki/HipoxiaDeepLearning/processed_data',
                        help='Path to save processed data')
    
    args = parser.parse_args()
    
    # Buat direktori output jika belum ada
    os.makedirs(args.output_path, exist_ok=True)
    
    # Proses data klinis
    clinical_df = process_clinical_data(args.dataset_path)
    
    # Simpan data klinis
    clinical_df.to_csv(os.path.join(args.output_path, 'clinical_data.csv'), index=False)
    print(f"Clinical data saved to {os.path.join(args.output_path, 'clinical_data.csv')}")
    
    print("\nDataset processing completed!")
    print(f"Total records processed: {len(clinical_df)}")

if __name__ == "__main__":
    main()