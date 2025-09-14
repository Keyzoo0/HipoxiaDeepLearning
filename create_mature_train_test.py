import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_mature_train_test_sets():
    """
    Membuat train/test sets dari dataset matang
    """
    # Membaca dataset matang
    mature_df = pd.read_csv('/home/zainul/joki/HipoxiaDeepLearning/processed_data/mature_clinical_dataset.csv')
    
    print(f"Mature dataset shape: {mature_df.shape}")
    print(f"Label distribution:")
    print(mature_df['label'].value_counts().sort_index())
    
    # Memisahkan fitur dan label
    X = mature_df.drop('label', axis=1)
    y = mature_df['label']
    
    # Membuat pembagian train/test dengan stratifikasi
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Menggabungkan kembali fitur dan label
    train_df = pd.concat([y_train, X_train], axis=1)
    test_df = pd.concat([y_test, X_test], axis=1)
    
    # Menyimpan train/test sets
    train_df.to_csv('/home/zainul/joki/HipoxiaDeepLearning/processed_data/mature_train_set.csv', index=False)
    test_df.to_csv('/home/zainul/joki/HipoxiaDeepLearning/processed_data/mature_test_set.csv', index=False)
    
    print(f"\nTrain set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    print(f"\nTrain set label distribution:")
    print(train_df['label'].value_counts().sort_index())
    
    print(f"\nTest set label distribution:")
    print(test_df['label'].value_counts().sort_index())
    
    # Menampilkan beberapa baris pertama dari train set
    print(f"\nFirst 5 rows of train set:")
    print(train_df.head())
    
    return train_df, test_df

def main():
    train_df, test_df = create_mature_train_test_sets()
    
    print(f"\nMature train/test sets created successfully!")

if __name__ == "__main__":
    main()