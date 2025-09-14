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
    print("Total records: {}".format(len(signal_info)))
    print("Records with pH labels: {}".format(len(signal_info.dropna(subset=['pH']))))
    
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
                print("\nVisualizing record {} (pH: {}, Category: {})".format(record_name, ph_value, int(ph_category)))
                
                try:
                    fig, (fhr_signal, uc_signal, time) = load_and_visualize_signal(signal_file_path)
                    
                    # Menyimpan plot
                    plot_filename = "sample_record_{}_category_{}.png".format(record_name, int(ph_category))
                    fig.savefig(os.path.join(data_path, plot_filename), dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    
                    print("Plot saved as {}".format(plot_filename))
                    
                    # Menampilkan statistik sinyal
                    print("  FHR - Min: {:.1f} bpm, Max: {:.1f} bpm, Mean: {:.1f} bpm".format(np.min(fhr_signal), np.max(fhr_signal), np.mean(fhr_signal)))
                    print("  UC  - Min: {:.1f}, Max: {:.1f}, Mean: {:.1f}".format(np.min(uc_signal), np.max(uc_signal), np.mean(uc_signal)))
                    
                except Exception as e:
                    print("Error visualizing record {}: {}".format(record_name, e))
            else:
                print("Signal file not found: {}".format(signal_file_path))

if __name__ == "__main__":
    main()