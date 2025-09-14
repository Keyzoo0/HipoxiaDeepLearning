import pandas as pd
import numpy as np

def create_mature_dataset(input_path, output_path):
    """
    Membuat dataset yang lebih matang dengan label yang jelas
    """
    # Membaca data klinis
    clinical_df = pd.read_csv(input_path)
    
    print(f"Original dataset shape: {clinical_df.shape}")
    print(f"Columns: {list(clinical_df.columns)}")
    
    # Mapping kategori pH ke label yang lebih deskriptif
    category_mapping = {
        0: 'normal',
        1: 'suspect',
        2: 'hypoxia'
    }
    
    # Membuat salinan dataframe
    mature_df = clinical_df.copy()
    
    # Menghapus kolom record_name
    if 'record_name' in mature_df.columns:
        mature_df = mature_df.drop('record_name', axis=1)
    
    # Mengganti nilai numerik pH_category dengan label tekstual
    if 'pH_category' in mature_df.columns:
        mature_df['pH_category'] = mature_df['pH_category'].map(category_mapping)
    
    # Memindahkan kolom pH_category ke posisi paling awal
    cols = list(mature_df.columns)
    cols.insert(0, cols.pop(cols.index('pH_category')))
    mature_df = mature_df[cols]
    
    # Mengganti nama kolom pH_category menjadi label
    mature_df = mature_df.rename(columns={'pH_category': 'label'})
    
    # Menghapus baris dengan nilai NaN pada label
    mature_df = mature_df.dropna(subset=['label'])
    
    # Menyimpan dataset matang
    mature_df.to_csv(output_path, index=False)
    
    print(f"Mature dataset shape: {mature_df.shape}")
    print(f"Label distribution:")
    print(mature_df['label'].value_counts())
    
    # Menampilkan beberapa baris pertama
    print(f"\nFirst 5 rows:")
    print(mature_df.head())
    
    return mature_df

def main():
    input_path = '/home/zainul/joki/HipoxiaDeepLearning/processed_data/clinical_data.csv'
    output_path = '/home/zainul/joki/HipoxiaDeepLearning/processed_data/mature_clinical_dataset.csv'
    
    mature_df = create_mature_dataset(input_path, output_path)
    
    print(f"\nMature clinical dataset saved to: {output_path}")

if __name__ == "__main__":
    main()