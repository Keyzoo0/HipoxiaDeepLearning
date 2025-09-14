# Ringkasan Proses Pemrosesan Dataset CTU-UHB CTG untuk Deteksi Hipoksia Janin

## 1. Pengunduhan Dataset

Dataset CTU-UHB Intrapartum Cardiotocography Database (CTGDB) telah berhasil diunduh dari PhysioNet. Dataset ini berisi:
- 552 rekaman CTG
- Sinyal FHR (Fetal Heart Rate) dan UC (Uterine Contractions)
- Frekuensi sampling 4 Hz
- Durasi maksimal 90 menit sebelum persalinan

## 2. Struktur Dataset

Setiap rekaman mengandung:
- File `.dat`: Berisi data sinyal FHR dan UC
- File `.hea`: Berisi header dengan informasi klinis termasuk nilai pH

## 3. Parameter Klinis Penting

Dataset menyediakan parameter klinis berikut:
- **pH**: Indikator utama hipoksia janin (rentang 6.85-7.47)
- **BDecf**: Base deficit
- **pCO2**: Tekanan parsial karbon dioksida
- **BE**: Base excess
- **Apgar1**: Skor Apgar pada 1 menit
- **Apgar5**: Skor Apgar pada 5 menit
- Informasi ibu dan janin (usia gestasi, berat badan, dll.)

## 4. Klasifikasi Hipoksia Berdasarkan pH

Kami mengkategorikan kondisi janin berdasarkan nilai pH arteri tali pusat:
- **Normal** (pH ≥ 7.20): 375 rekaman
- **Suspect** (7.10 ≤ pH < 7.20): 121 rekaman  
- **Hipoksia** (pH < 7.10): 56 rekaman

## 5. Pemrosesan Data

### 5.1 Ekstraksi Data Klinis
- Mengekstrak parameter klinis dari file header
- Membuat dataset klinis dengan 36 kolom
- Menyimpan ke `clinical_data.csv`

### 5.2 Pemrosesan Sinyal
- Mengekstrak sinyal FHR dan UC dari semua 552 rekaman
- Menyimpan sinyal dalam format numpy array
- Menormalisasi nilai FHR ke rentang 60-200 bpm
- Membuat metadata sinyal di `signal_info.csv`

### 5.3 Pembagian Dataset
- Membagi dataset menjadi train (80%) dan test (20%) sets
- Train set: 441 rekaman
- Test set: 111 rekaman
- Pembagian mempertimbangkan distribusi kelas agar seimbang

## 6. Statistik Dataset Akhir

Total rekaman: 552
Rekaman dengan label pH: 552 (100%)

Distribusi kelas:
- Normal (pH ≥ 7.20): 375 (68%)
- Suspect (7.10 ≤ pH < 7.20): 121 (22%)  
- Hipoksia (pH < 7.10): 56 (10%)

## 7. Direktori Output

Data yang dihasilkan tersimpan di `/home/zainul/joki/HipoxiaDeepLearning/processed_data/`:
- `clinical_data.csv`: Dataset klinis lengkap
- `signal_info.csv`: Metadata sinyal
- `signals/`: Direktori berisi file sinyal dalam format numpy
- `train_set.csv`: Informasi data training
- `test_set.csv`: Informasi data testing
- `sample_record_*_category_*.png`: Visualisasi contoh sinyal

## 8. Siap untuk Pelatihan Model

Dataset kini siap digunakan untuk pelatihan model deep learning deteksi hipoksia janin dengan:
- Input: Sinyal FHR dan UC
- Output: Klasifikasi 3 kelas (normal, suspect, hipoksia)
- Pembagian data yang seimbang
- Visualisasi contoh data tersedia