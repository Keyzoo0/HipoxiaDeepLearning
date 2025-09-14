import pandas as pd
import numpy as np
import os

def verify_dataset_integrity():
    """
    Memverifikasi integritas dataset yang sudah diproses
    """
    print("Verifying dataset integrity...")
    
    # Path ke direktori dataset
    data_path = '/home/zainul/joki/HipoxiaDeepLearning/processed_data'
    signals_path = os.path.join(data_path, 'signals')
    
    # 1. Memverifikasi dataset klinis matang
    print("\n1. Verifying mature clinical dataset...")
    mature_df = pd.read_csv(os.path.join(data_path, 'mature_clinical_dataset.csv'))
    print(f"   Shape: {mature_df.shape}")
    print(f"   Columns: {len(mature_df.columns)}")
    print(f"   Missing values in label column: {mature_df['label'].isnull().sum()}")
    print(f"   Label distribution:")
    print(f"   {mature_df['label'].value_counts().sort_index()}")
    
    # 2. Memverifikasi train/test sets
    print("\n2. Verifying train/test sets...")
    train_df = pd.read_csv(os.path.join(data_path, 'mature_train_set.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'mature_test_set.csv'))
    
    print(f"   Train set shape: {train_df.shape}")
    print(f"   Test set shape: {test_df.shape}")
    print(f"   Train set label distribution:")
    print(f"   {train_df['label'].value_counts().sort_index()}")
    print(f"   Test set label distribution:")
    print(f"   {test_df['label'].value_counts().sort_index()}")
    
    # 3. Memverifikasi data sinyal
    print("\n3. Verifying signal data...")
    signal_files = [f for f in os.listdir(signals_path) if f.endswith('_signals.npy')]
    print(f"   Total signal files: {len(signal_files)}")
    
    # Memverifikasi beberapa file sinyal secara acak
    sample_files = signal_files[:5] if len(signal_files) >= 5 else signal_files
    print(f"   Verifying {len(sample_files)} sample signal files:")
    
    for signal_file in sample_files:
        try:
            signal_data = np.load(os.path.join(signals_path, signal_file), allow_pickle=True).item()
            fhr_signal = signal_data['FHR']
            uc_signal = signal_data['UC']
            fs = signal_data['sampling_frequency']
            
            print(f"     {signal_file}:")
            print(f"       FHR shape: {fhr_signal.shape}")
            print(f"       UC shape: {uc_signal.shape}")
            print(f"       Sampling frequency: {fs} Hz")
            print(f"       Duration: {len(fhr_signal)/fs:.1f} seconds")
            print(f"       FHR range: [{np.min(fhr_signal):.1f}, {np.max(fhr_signal):.1f}] bpm")
            print(f"       UC range: [{np.min(uc_signal):.1f}, {np.max(uc_signal):.1f}]")
            
        except Exception as e:
            print(f"     Error verifying {signal_file}: {e}")
    
    # 4. Statistik dataset
    print("\n4. Dataset statistics...")
    print(f"   Total records: {len(mature_df)}")
    print(f"   Features: {len(mature_df.columns) - 1}")  # -1 karena label
    print(f"   Label column: 'label'")
    print(f"   Label classes: {sorted(mature_df['label'].unique())}")
    
    # Statistik pH
    if 'pH' in mature_df.columns:
        print(f"   pH statistics:")
        print(f"     Min: {mature_df['pH'].min():.2f}")
        print(f"     Max: {mature_df['pH'].max():.2f}")
        print(f"     Mean: {mature_df['pH'].mean():.2f}")
        print(f"     Std: {mature_df['pH'].std():.2f}")
    
    print("\nDataset integrity verification completed!")

def main():
    verify_dataset_integrity()

if __name__ == "__main__":
    main()