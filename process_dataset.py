import wfdb
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

def process_clinical_data(dataset_path):
    \"\"\"Mengekstrak informasi klinis dari semua record\"\"\"
    print(\"Processing clinical data...\")
    
    # Membaca daftar record dari file RECORDS
    with open(os.path.join(dataset_path, 'RECORDS'), 'r') as f:
        records = [line.strip() for line in f.readlines()]
    
    print(f\"Total records to process: {len(records)}\")
    
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
                print(f\"Processed {i + 1}/{len(records)} records\")
                
        except Exception as e:
            print(f\"Error processing record {record_name}: {e}\")
    
    print(f\"Finished processing {len(clinical_data)} records\")
    
    # Membuat DataFrame
    df = pd.DataFrame(clinical_data)
    
    # Membuat label klasifikasi berdasarkan pH
    if 'pH' in df.columns:
        df['pH_category'] = df['pH'].apply(lambda x: 
            0 if x >= 7.20 else  # normal
            1 if x >= 7.10 else  # suspect
            2)                   # hypoxia
    
    return df

def process_signal_data(dataset_path, output_path, clinical_df):
    \"\"\"Mengekstrak data sinyal dari semua record\"\"\"
    print(\"Processing signal data...\")
    
    # Membuat direktori untuk menyimpan data sinyal
    signal_output_path = os.path.join(output_path, 'signals')
    os.makedirs(signal_output_path, exist_ok=True)
    
    # Menyiapkan dataframe untuk informasi sinyal
    signal_info = []
    
    # Memproses semua record
    records_to_process = [str(name) for name in clinical_df['record_name'].tolist()]
    
    print(f\"Processing {len(records_to_process)} records...\")
    
    for i, record_name in enumerate(records_to_process):
        try:
            # Membaca record sinyal
            record_path = os.path.join(dataset_path, str(record_name))
            record = wfdb.rdrecord(record_path)
            
            # Mendapatkan data sinyal
            signal_data = record.p_signal  # shape: (samples, channels)
            signal_names = record.sig_name  # ['FHR', 'UC']
            
            # Memastikan urutan sinyal benar
            fhr_index = signal_names.index('FHR') if 'FHR' in signal_names else 0
            uc_index = signal_names.index('UC') if 'UC' in signal_names else 1
            
            # Memisahkan sinyal FHR dan UC
            fhr_signal = signal_data[:, fhr_index]
            uc_signal = signal_data[:, uc_index]
            
            # Normalisasi sinyal
            # FHR: normal range 120-160 bpm, dengan batas 60-200
            # UC: tidak perlu normalisasi khusus
            fhr_normalized = np.clip(fhr_signal, 60, 200)  # Batas nilai FHR yang realistis
            uc_normalized = uc_signal  # UC tidak dinormalisasi karena nilainya relatif
            
            # Menyimpan sinyal ke file numpy
            signal_filename = f'{record_name}_signals.npy'
            signal_dict = {
                'FHR': fhr_normalized, 
                'UC': uc_normalized,
                'sampling_frequency': record.fs
            }
            np.save(os.path.join(signal_output_path, signal_filename), signal_dict)
            
            # Mendapatkan informasi pH dari clinical_df
            ph_value = None
            ph_category = None
            
            # Cari record dalam clinical_df
            matching_records = clinical_df[clinical_df['record_name'] == int(record_name)]
            if len(matching_records) > 0:
                ph_row = matching_records.iloc[0]
                ph_value = ph_row['pH'] if 'pH' in ph_row else None
                ph_category = ph_row['pH_category'] if 'pH_category' in ph_row else None
            
            # Menyimpan informasi sinyal
            signal_info.append({
                'record_name': record_name,
                'signal_filename': signal_filename,
                'duration_seconds': len(fhr_signal) / record.fs,
                'sampling_frequency': record.fs,
                'num_samples': len(fhr_signal),
                'pH': ph_value,
                'pH_category': ph_category
            })
            
            # Menampilkan progress setiap 50 record
            if (i + 1) % 50 == 0:
                print(f\"Processed {i + 1}/{len(records_to_process)}: {record_name}\")
            
        except Exception as e:
            print(f\"Error processing record {record_name}: {e}\")
    
    # Membuat DataFrame dengan informasi sinyal
    signal_df = pd.DataFrame(signal_info)
    
    return signal_df

def create_train_test_splits(signal_df, output_path):
    \"\"\"Membuat pembagian data train/test\"\"\"
    print(\"Creating train/test splits...\")
    
    # Hapus record tanpa label pH
    labeled_data = signal_df.dropna(subset=['pH_category'])
    
    # Pastikan pH_category adalah integer
    labeled_data['pH_category'] = labeled_data['pH_category'].astype(int)
    
    # Pisahkan fitur dan label
    X = labeled_data[['record_name', 'signal_filename', 'duration_seconds', 'sampling_frequency', 'num_samples']]
    y = labeled_data['pH_category']
    
    # Buat pembagian train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Gabungkan kembali dengan informasi sinyal
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Simpan ke file CSV
    train_df.to_csv(os.path.join(output_path, 'train_set.csv'), index=False)
    test_df.to_csv(os.path.join(output_path, 'test_set.csv'), index=False)
    
    print(f\"Train set size: {len(train_df)}\")
    print(f\"Test set size: {len(test_df)}\")
    print(\"Class distribution in train set:\")
    print(train_df['pH_category'].value_counts().sort_index())
    print(\"Class distribution in test set:\")
    print(test_df['pH_category'].value_counts().sort_index())
    
    return train_df, test_df

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
    print(f\"Clinical data saved to {os.path.join(args.output_path, 'clinical_data.csv')}\")
    
    # Proses data sinyal
    signal_df = process_signal_data(args.dataset_path, args.output_path, clinical_df)
    
    # Simpan informasi sinyal
    signal_df.to_csv(os.path.join(args.output_path, 'signal_info.csv'), index=False)
    print(f\"Signal information saved to {os.path.join(args.output_path, 'signal_info.csv')}\")
    
    # Buat pembagian train/test
    train_df, test_df = create_train_test_splits(signal_df, args.output_path)
    
    print(\"\\nDataset processing completed!\")
    print(f\"Total records processed: {len(signal_df)}\")
    print(f\"Records with pH labels: {len(signal_df.dropna(subset=['pH']))}\")

if __name__ == \"__main__\":
    main()