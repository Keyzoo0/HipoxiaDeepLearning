import numpy as np
import pandas as pd
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DatasetGenerator:
    def __init__(self, base_path='/home/zainul/joki/HipoxiaDeepLearning'):
        self.base_path = Path(base_path)
        self.processed_data_path = self.base_path / 'processed_data'
        self.signals_path = self.processed_data_path / 'signals'
        
    def load_clinical_data(self):
        """Load clinical dataset with labels"""
        clinical_file = self.processed_data_path / 'mature_clinical_dataset.csv'
        if not clinical_file.exists():
            raise FileNotFoundError(f"Clinical data not found: {clinical_file}")
        
        df = pd.read_csv(clinical_file)
        print(f"✅ Loaded clinical data: {len(df)} records")
        print(f"   Distribution: Normal={len(df[df['label']=='normal'])}, "
              f"Suspect={len(df[df['label']=='suspect'])}, "
              f"Hypoxia={len(df[df['label']=='hypoxia'])}")
        return df
    
    def load_signal_data(self, record_id):
        """Load signal data for specific record"""
        signal_file = self.signals_path / f"{record_id}_signals.npy"
        if not signal_file.exists():
            raise FileNotFoundError(f"Signal file not found: {signal_file}")
        
        data = np.load(signal_file, allow_pickle=True).item()
        return {
            'FHR': data['FHR'],
            'UC': data['UC'], 
            'sampling_frequency': data['sampling_frequency'],
            'record_id': record_id
        }
    
    def get_record_mapping(self):
        """Create mapping of record numbers to labels and clinical data"""
        clinical_data = self.load_clinical_data()
        
        # Get all available signal files
        signal_files = list(self.signals_path.glob("*_signals.npy"))
        available_record_ids = [int(f.stem.split('_')[0]) for f in signal_files]
        
        print(f"   Found {len(signal_files)} signal files")
        print(f"   Record ID range: {min(available_record_ids)} - {max(available_record_ids)}")
        
        record_mapping = {}
        
        # Map clinical data to available signal files
        for idx, row in clinical_data.iterrows():
            # Use index-based mapping since we have 552 clinical records
            # and signal files start from 1001
            if idx < len(available_record_ids):
                record_id = available_record_ids[idx]
                
                # Check if signal file exists
                signal_file = self.signals_path / f"{record_id}_signals.npy"
                if signal_file.exists():
                    record_mapping[record_id] = {
                        'label': row['label'],
                        'pH': row['pH'] if 'pH' in row.index and pd.notna(row['pH']) else 7.2,
                        'clinical_data': row.to_dict()
                    }
        
        return record_mapping
    
    def generate_unified_dataset(self):
        """Generate unified dataset for all methods"""
        print("🔄 Generating unified dataset...")
        
        record_mapping = self.get_record_mapping()
        print(f"✅ Found {len(record_mapping)} valid records")
        
        # Create output directories
        output_dir = self.base_path / 'data'
        output_dir.mkdir(exist_ok=True)
        
        # Generate dataset info
        dataset_info = []
        valid_records = []
        
        for record_id, info in record_mapping.items():
            try:
                signal_data = self.load_signal_data(record_id)
                
                # Basic validation
                if len(signal_data['FHR']) > 1000:  # Minimum length check
                    dataset_info.append({
                        'record_id': record_id,
                        'label': info['label'],
                        'pH': info['pH'],
                        'signal_length': len(signal_data['FHR']),
                        'sampling_freq': signal_data['sampling_frequency']
                    })
                    valid_records.append(record_id)
                    
            except FileNotFoundError:
                print(f"⚠️ Warning: Signal file not found for record {record_id}")
                continue
        
        # Save dataset info
        dataset_df = pd.DataFrame(dataset_info)
        dataset_df.to_csv(output_dir / 'dataset_info.csv', index=False)
        
        print(f"✅ Generated dataset info for {len(valid_records)} records")
        print(f"   Saved to: {output_dir / 'dataset_info.csv'}")
        
        # Show distribution
        label_counts = dataset_df['label'].value_counts()
        print(f"\n📊 Label Distribution:")
        for label, count in label_counts.items():
            print(f"   {label}: {count} records ({count/len(dataset_df)*100:.1f}%)")
        
        return dataset_df, valid_records
    
    def get_available_records(self):
        """Get list of all available record IDs for prediction"""
        try:
            dataset_info = pd.read_csv(self.base_path / 'data' / 'dataset_info.csv')
            return sorted(dataset_info['record_id'].tolist())
        except FileNotFoundError:
            print("⚠️ Dataset info not found. Generating now...")
            dataset_df, _ = self.generate_unified_dataset()
            return sorted(dataset_df['record_id'].tolist())
    
    def get_record_info(self, record_id):
        """Get information about specific record"""
        try:
            dataset_info = pd.read_csv(self.base_path / 'data' / 'dataset_info.csv')
            record_row = dataset_info[dataset_info['record_id'] == record_id]
            if len(record_row) == 0:
                return None
            return record_row.iloc[0].to_dict()
        except FileNotFoundError:
            return None
    
    def prepare_data_for_training(self, method='gan'):
        """Prepare data specifically for training different methods"""
        print(f"🔄 Preparing data for {method.upper()} training...")
        
        dataset_info = pd.read_csv(self.base_path / 'data' / 'dataset_info.csv')
        
        # Load all signals and organize by method requirements
        X_data = []
        y_data = []
        record_ids = []
        
        label_map = {'normal': 0, 'suspect': 1, 'hypoxia': 2}
        
        for _, row in dataset_info.iterrows():
            record_id = row['record_id']
            label = row['label']
            
            try:
                signal_data = self.load_signal_data(record_id)
                
                if method == 'gan':
                    # For GAN: use raw FHR signal
                    X_data.append(signal_data['FHR'])
                elif method in ['mobilenet', 'resnet']:
                    # For CNN methods: prepare as 1D signal (will be converted to spectrogram later)
                    X_data.append(signal_data['FHR'])
                
                y_data.append(label_map[label])
                record_ids.append(record_id)
                
            except Exception as e:
                print(f"⚠️ Error loading record {record_id}: {e}")
                continue
        
        # Convert to numpy arrays
        X_data = np.array(X_data, dtype=object)  # Keep as object array for variable lengths
        y_data = np.array(y_data)
        record_ids = np.array(record_ids)
        
        print(f"✅ Prepared {len(X_data)} samples for {method.upper()} training")
        
        # Save prepared data
        method_dir = self.base_path / 'data' / method
        method_dir.mkdir(exist_ok=True)
        
        np.save(method_dir / 'X_data.npy', X_data)
        np.save(method_dir / 'y_data.npy', y_data)
        np.save(method_dir / 'record_ids.npy', record_ids)
        
        print(f"📁 Saved to: {method_dir}/")
        
        return X_data, y_data, record_ids

def main():
    """Main function to generate dataset"""
    print("🚀 Starting Dataset Generation...")
    
    generator = DatasetGenerator()
    
    try:
        # Generate unified dataset
        dataset_df, valid_records = generator.generate_unified_dataset()
        
        # Prepare data for each method
        for method in ['gan', 'mobilenet', 'resnet']:
            X, y, ids = generator.prepare_data_for_training(method)
            print()
        
        print("✅ Dataset generation completed successfully!")
        print(f"\n📋 Summary:")
        print(f"   Total valid records: {len(valid_records)}")
        print(f"   Data prepared for: GAN, MobileNet, ResNet")
        print(f"   Available records: {min(valid_records)} - {max(valid_records)}")
        
    except Exception as e:
        print(f"❌ Error during dataset generation: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()