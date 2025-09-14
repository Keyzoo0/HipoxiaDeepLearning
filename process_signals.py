import wfdb
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def process_all_signals(dataset_path, output_path, clinical_df):
    \"\"\"Mengekstrak data sinyal dari semua record\"\"\"
    print(\"Processing all signal data...\")
    
    # Membuat direktori untuk menyimpan data sinyal
    signal_output_path = os.path.join(output_path, 'signals')
    os.makedirs(signal_output_path, exist_ok=True)
    
    # Menyiapkan list untuk informasi sinyal
    signal_info = []
    
    # Memproses semua record
    records_to_process = [str(name) for name in clinical_df['record_name'].tolist()]
    
    print(f\"Processing {len(records_to_process)} records...\")
    
    success_count = 0
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
            fhr_normalized = np.clip(fhr_signal, 60, 200)
            uc_normalized = uc_signal
            
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
                ph_value = ph_row['pH'] if 'pH' in ph_row and pd.notna(ph_row['pH']) else None
                ph_category = ph_row['pH_category'] if 'pH_category' in ph_row and pd.notna(ph_row['pH_category']) else None
            
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
            
            success_count += 1
            
            # Menampilkan progress setiap 50 record
            if (i + 1) % 50 == 0:
                print(f\"Processed {i + 1}/{len(records_to_process)} records ({success_count} successful)\")
            
        except Exception as e:
            print(f\"Error processing record {record_name}: {e}\")
    
    print(f\"Successfully processed {success_count}/{len(records_to_process)} records\")
    
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
    # Path ke dataset dan output
    dataset_path = '/home/zainul/joki/HipoxiaDeepLearning/dataset/physionet.org/files/ctu-uhb-ctgdb/1.0.0'
    output_path = '/home/zainul/joki/HipoxiaDeepLearning/processed_data'
    
    # Memuat data klinis yang sudah diproses
    clinical_df = pd.read_csv(os.path.join(output_path, 'clinical_data.csv'))
    print(f\"Loaded clinical data with {len(clinical_df)} records\")
    
    # Proses data sinyal
    signal_df = process_all_signals(dataset_path, output_path, clinical_df)
    
    # Simpan informasi sinyal
    signal_df.to_csv(os.path.join(output_path, 'signal_info.csv'), index=False)
    print(f\"Signal information saved to {os.path.join(output_path, 'signal_info.csv')}\")
    
    # Buat pembagian train/test
    train_df, test_df = create_train_test_splits(signal_df, output_path)
    
    print(\"\nDataset processing completed!\")
    print(f\"Total records processed: {len(signal_df)}\")
    print(f\"Records with pH labels: {len(signal_df.dropna(subset=['pH']))}\")

if __name__ == \"__main__\":
    main()