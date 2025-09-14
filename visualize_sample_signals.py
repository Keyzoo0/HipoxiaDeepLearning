import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_and_visualize_signal(signal_file_path, num_points=2000):
    """
    Memuat dan memvisualisasikan sinyal FHR dan UC
    """
    # Memuat data sinyal
    signal_data = np.load(signal_file_path, allow_pickle=True).item()
    
    # Mendapatkan sinyal FHR dan UC
    fhr_signal = signal_data['FHR']
    uc_signal = signal_data['UC']
    fs = signal_data['sampling_frequency']
    
    # Mengambil sebagian data untuk visualisasi
    if len(fhr_signal) > num_points:
        fhr_signal = fhr_signal[:num_points]
        uc_signal = uc_signal[:num_points]
    
    # Membuat waktu (dalam detik)
    time = np.arange(len(fhr_signal)) / fs
    
    # Membuat plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot FHR
    ax1.plot(time, fhr_signal, 'b-', linewidth=0.8)
    ax1.set_ylabel('FHR (bpm)')
    ax1.set_title('Fetal Heart Rate')
    ax1.grid(True, alpha=0.3)
    
    # Plot UC
    ax2.plot(time, uc_signal, 'r-', linewidth=0.8)
    ax2.set_ylabel('UC')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_title('Uterine Contractions')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, (fhr_signal, uc_signal, time)

def main():
    # Path ke direktori data
    data_path = '/home/zainul/joki/HipoxiaDeepLearning/processed_data'
    signals_path = os.path.join(data_path, 'signals')
    
    # Memuat informasi sinyal
    signal_info = pd.read_csv(os.path.join(data_path, 'signal_info.csv'))
    
    # Menampilkan statistik dataset
    print("Dataset Statistics:")
    print(f"Total records: {len(signal_info)}")
    print(f"Records with pH labels: {len(signal_info.dropna(subset=['pH']))")
    
    # Menampilkan distribusi kategori pH
    print("\npH Category Distribution:")
    print(signal_info['pH_category'].value_counts().sort_index())
    
    # Memilih beberapa contoh dari setiap kategori untuk visualisasi
    categories = signal_info['pH_category'].dropna().unique()
    categories.sort()
    
    for category in categories:
        # Memilih satu record dari setiap kategori
        sample_records = signal_info[signal_info['pH_category'] == category].head(1)
        
        for _, record in sample_records.iterrows():
            record_name = record['record_name']
            signal_filename = record['signal_filename']
            ph_value = record['pH']
            ph_category = record['pH_category']
            
            # Membuat path ke file sinyal
            signal_file_path = os.path.join(signals_path, signal_filename)
            
            # Memuat dan memvisualisasikan sinyal
            if os.path.exists(signal_file_path):
                print(f"\nVisualizing record {record_name} (pH: {ph_value}, Category: {ph_category})")
                
                try:
                    fig, (fhr_signal, uc_signal, time) = load_and_visualize_signal(signal_file_path)
                    
                    # Menyimpan plot
                    plot_filename = f"sample_record_{record_name}_category_{int(ph_category)}.png"
                    fig.savefig(os.path.join(data_path, plot_filename), dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    
                    print(f"Plot saved as {plot_filename}")
                    
                    # Menampilkan statistik sinyal
                    print(f"  FHR - Min: {np.min(fhr_signal):.1f} bpm, Max: {np.max(fhr_signal):.1f} bpm, Mean: {np.mean(fhr_signal):.1f} bpm")
                    print(f"  UC  - Min: {np.min(uc_signal):.1f}, Max: {np.max(uc_signal):.1f}, Mean: {np.mean(uc_signal):.1f}")
                    
                except Exception as e:
                    print(f"Error visualizing record {record_name}: {e}")
            else:
                print(f"Signal file not found: {signal_file_path}")

if __name__ == "__main__":
    main()