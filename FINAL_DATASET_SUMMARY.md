# RINGKASAN AKHIR PEMROSESAN DATASET CTU-UHB CTG UNTUK DETEKSI HIPOKSIA JANIN

## Struktur Dataset Akhir

Dataset CTU-UHB CTG telah berhasil diproses menjadi beberapa komponen:

### 1. Dataset Klinis Matang (`mature_clinical_dataset.csv`)
- Jumlah record: 552
- Jumlah fitur: 35 (tidak termasuk label)
- Label: `normal`, `suspect`, `hypoxia`
- Distribusi kelas:
  - Normal: 375 record (68%)
  - Suspect: 121 record (22%)
  - Hipoksia: 56 record (10%)

### 2. Pembagian Train/Test Sets
- **Train Set** (`mature_train_set.csv`):
  - Jumlah record: 441 (80% dari total)
  - Distribusi kelas:
    - Normal: 299 record
    - Suspect: 97 record
    - Hipoksia: 45 record

- **Test Set** (`mature_test_set.csv`):
  - Jumlah record: 111 (20% dari total)
  - Distribusi kelas:
    - Normal: 76 record
    - Suspect: 24 record
    - Hipoksia: 11 record

### 3. Data Sinyal
- Direktori: `/home/zainul/joki/HipoxiaDeepLearning/processed_data/signals/`
- Jumlah file sinyal: 552
- Format file: `.npy` (numpy array)
- Setiap file berisi:
  - Sinyal FHR (Fetal Heart Rate)
  - Sinyal UC (Uterine Contractions)
  - Informasi frekuensi sampling

## Fitur Dataset Klinis

Dataset klinis matang berisi 35 fitur penting yang relevan untuk prediksi hipoksia janin:

1. **Parameter Fetal**:
   - pH: Nilai pH darah tali pusat (6.85-7.47)
   - BDecf: Base deficit
   - pCO2: Tekanan parsial CO2
   - BE: Base excess
   - Apgar1: Skor Apgar pada 1 menit
   - Apgar5: Skor Apgar pada 5 menit

2. **Parameter Persalinan**:
   - Gest.: Usia kehamilan (minggu)
   - Weight(g): Berat janin (gram)
   - Sex: Jenis kelamin janin
   - Delivery descriptors: Informasi persalinan

3. **Parameter Maternal**:
   - Age: Usia ibu
   - Gravidity: Kehamilan ke-
   - Parity: Kelahiran ke-
   - Risk factors: Diabetes, hipertensi, preeklamsia, dll.

## Penggunaan untuk Deep Learning

Dataset ini siap digunakan untuk pengembangan model deep learning dengan:

1. **Input**:
   - Sinyal FHR dan UC dari file `.npy`
   - Atau fitur klinis dari dataset CSV

2. **Output**:
   - Klasifikasi 3 kelas: `normal`, `suspect`, `hypoxia`

3. **Arsitektur Model yang Disarankan**:
   - CNN untuk analisis sinyal waktu nyata
   - LSTM/GRU untuk analisis sekuens temporal
   - Hybrid CNN-LSTM untuk kombinasi spasial-temporal
   - Fully Connected Network untuk fitur klinis

## Validasi Dataset

Dataset telah divalidasi dengan:
- Pemeriksaan integritas data
- Pembagian train/test yang seimbang
- Normalisasi sinyal FHR ke rentang 60-200 bpm
- Penghapusan nilai-nilai ekstrem yang tidak realistis

## Langkah Selanjutnya

Dataset ini siap untuk digunakan dalam pengembangan model deep learning deteksi hipoksia janin. Anda dapat:

1. Menggunakan sinyal FHR/UC untuk model berbasis sinyal
2. Menggunakan fitur klinis untuk model berbasis tabular
3. Menggabungkan kedua pendekatan untuk model hybrid
4. Melakukan evaluasi dengan metrik akurasi, presisi, recall, dan F1-score

Semua file tersedia di direktori `/home/zainul/joki/HipoxiaDeepLearning/processed_data/`.