#!/usr/bin/env python3
"""
Text-based version of main.py without inquirer (for environments that don't support interactive input)
"""

import sys
import os
from pathlib import Path
import traceback
from datetime import datetime

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import all modules
try:
    from generateDataset import DatasetGenerator
    from methods.gan_method.trainingWithGanMethod import main as train_gan
    from methods.gan_method.predictWithGanMethod import GANPredictor
    from methods.mobilenet_method.trainingWithMobileNet import main as train_mobilenet
    from methods.mobilenet_method.predictWithMobileNet import MobileNetPredictor
    from methods.resnet_method.trainingWithResNet import main as train_resnet
    from methods.resnet_method.predictWithResNet import ResNetPredictor
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure all required packages are installed and files are in correct locations.")
    sys.exit(1)

class HipoxiaDeepLearningTextApp:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.dataset_generator = DatasetGenerator(self.base_path)
        
        # Predictors (will be loaded on demand)
        self.gan_predictor = None
        self.mobilenet_predictor = None
        self.resnet_predictor = None
        
        self.available_records = []
        
    def print_banner(self):
        """Print application banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    HIPOXIA DEEP LEARNING SYSTEM                             ║
║                                                                              ║
║  🔬 Multi-Method Fetal Heart Rate Hypoxia Detection System                  ║
║  📊 Methods: GAN Data Augmentation | MobileNet CNN | ResNet Deep Learning   ║
║  🎯 Input: Record Number → Output: Hypoxia Classification + Visualization   ║
║                                                                              ║
║  Dataset: CTU-UHB Intrapartum Cardiotocography Database                     ║
║  Classes: Normal | Suspect | Hypoxia (based on umbilical cord pH)          ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        print(f"🕐 Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def load_available_records(self):
        """Load available records for prediction"""
        try:
            self.available_records = self.dataset_generator.get_available_records()
            if self.available_records:
                print(f"✅ Found {len(self.available_records)} available records")
                print(f"   Record range: {min(self.available_records)} - {max(self.available_records)}")
            else:
                print("⚠️ No records found. Please generate dataset first.")
        except Exception as e:
            print(f"⚠️ Error loading records: {e}")
            self.available_records = []
    
    def get_input(self, prompt, valid_options=None):
        """Get user input with validation"""
        while True:
            try:
                user_input = input(f"{prompt}: ").strip()
                if valid_options and user_input not in valid_options:
                    print(f"❌ Invalid input. Please choose from: {', '.join(valid_options)}")
                    continue
                return user_input
            except (EOFError, KeyboardInterrupt):
                print("\n⚠️ Input interrupted")
                return None
    
    def generate_dataset_menu(self):
        """Dataset generation menu"""
        print("\n" + "="*60)
        print("📁 DATASET GENERATION")
        print("="*60)
        
        try:
            success = self.dataset_generator.main()
            if success:
                print("\n✅ Dataset generation completed successfully!")
                self.load_available_records()
            else:
                print("\n❌ Dataset generation failed!")
            
            input("\nPress Enter to continue...")
            
        except Exception as e:
            print(f"\n❌ Error during dataset generation: {e}")
            traceback.print_exc()
            input("\nPress Enter to continue...")
    
    def training_menu(self):
        """Training method selection menu"""
        print("\n" + "="*60)
        print("🚀 TRAINING METHODS")
        print("="*60)
        
        if not self.available_records:
            print("⚠️ No dataset found. Please generate dataset first.")
            input("Press Enter to continue...")
            return
        
        print("Select training method:")
        print("1. 🤖 GAN Method (Data Augmentation + Classification)")
        print("2. 📱 MobileNet Method (Lightweight CNN)")
        print("3. 🏗️ ResNet Method (Deep Residual Network)")
        print("4. 🔙 Back to Main Menu")
        
        choice = self.get_input("Enter choice (1-4)", ['1', '2', '3', '4'])
        if choice is None or choice == '4':
            return
        
        method_map = {
            '1': ('GAN', train_gan),
            '2': ('MobileNet', train_mobilenet), 
            '3': ('ResNet', train_resnet)
        }
        
        method_name, train_func = method_map[choice]
        
        print(f"\n🔄 Starting {method_name} training...")
        print("-" * 50)
        
        try:
            success = train_func()
            if success:
                print(f"\n✅ {method_name} training completed successfully!")
            else:
                print(f"\n❌ {method_name} training failed!")
                
        except Exception as e:
            print(f"\n❌ Error during {method_name} training: {e}")
            traceback.print_exc()
        
        input("\nPress Enter to continue...")
    
    def prediction_menu(self):
        """Prediction method selection menu"""
        print("\n" + "="*60)
        print("🔍 PREDICTION METHODS")
        print("="*60)
        
        if not self.available_records:
            print("⚠️ No dataset found. Please generate dataset first.")
            input("Press Enter to continue...")
            return
        
        print("Select prediction method:")
        print("1. 🤖 GAN Method Prediction")
        print("2. 📱 MobileNet Method Prediction")
        print("3. 🏗️ ResNet Method Prediction")
        print("4. 📊 Compare All Methods")
        print("5. 🔙 Back to Main Menu")
        
        choice = self.get_input("Enter choice (1-5)", ['1', '2', '3', '4', '5'])
        if choice is None or choice == '5':
            return
        
        if choice == '4':
            self.compare_all_methods()
        else:
            method_map = {
                '1': 'gan',
                '2': 'mobilenet',
                '3': 'resnet'
            }
            self.single_method_prediction(method_map[choice])
    
    def single_method_prediction(self, method):
        """Handle single method prediction"""
        print(f"\n🔍 {method.upper()} PREDICTION")
        print("-" * 50)
        
        print(f"Available records: {len(self.available_records)} total")
        print(f"Record range: {min(self.available_records)} - {max(self.available_records)}")
        
        print("\nSelect record input option:")
        print("1. 📝 Enter specific record number")
        print("2. 🎲 Use random record")
        print("3. 📊 Batch predict (first 5 records)")
        print("4. 🔙 Back to Prediction Menu")
        
        choice = self.get_input("Enter choice (1-4)", ['1', '2', '3', '4'])
        if choice is None or choice == '4':
            return
        
        # Get predictor
        try:
            predictor = self.get_predictor(method)
            if predictor is None:
                print(f"❌ Could not load {method.upper()} predictor. Please train the model first.")
                input("Press Enter to continue...")
                return
        except Exception as e:
            print(f"❌ Error loading {method.upper()} predictor: {e}")
            input("Press Enter to continue...")
            return
        
        # Handle different record options
        try:
            if choice == '1':
                self.manual_record_prediction(predictor, method)
            elif choice == '2':
                self.random_record_prediction(predictor, method)
            elif choice == '3':
                self.batch_record_prediction(predictor, method)
                
        except Exception as e:
            print(f"❌ Error during {method.upper()} prediction: {e}")
            traceback.print_exc()
        
        input("\nPress Enter to continue...")
    
    def manual_record_prediction(self, predictor, method):
        """Handle manual record number input"""
        while True:
            record_input = self.get_input(
                f"Enter record number ({min(self.available_records)}-{max(self.available_records)})"
            )
            
            if record_input is None:
                break
                
            try:
                record_id = int(record_input)
                if record_id not in self.available_records:
                    print(f"❌ Record {record_id} not found. Available range: {min(self.available_records)}-{max(self.available_records)}")
                    continue
            except ValueError:
                print("❌ Please enter a valid number")
                continue
            
            print(f"\n🔍 Predicting record {record_id} using {method.upper()} method...")
            print("-" * 40)
            
            try:
                result = predictor.predict_record(record_id, show_visualizations=True)
                self.display_prediction_result(result, method)
                
                continue_choice = self.get_input("Predict another record? (y/n)", ['y', 'n', 'yes', 'no'])
                if continue_choice is None or continue_choice.lower() in ['n', 'no']:
                    break
                    
            except Exception as e:
                print(f"❌ Error predicting record {record_id}: {e}")
                break
    
    def random_record_prediction(self, predictor, method):
        """Handle random record prediction"""
        import random
        
        record_id = random.choice(self.available_records)
        
        print(f"\n🎲 Randomly selected record: {record_id}")
        print(f"🔍 Predicting using {method.upper()} method...")
        print("-" * 40)
        
        try:
            result = predictor.predict_record(record_id, show_visualizations=True)
            self.display_prediction_result(result, method)
        except Exception as e:
            print(f"❌ Error predicting record {record_id}: {e}")
    
    def batch_record_prediction(self, predictor, method):
        """Handle batch prediction"""
        batch_records = self.available_records[:5]
        
        print(f"\n📊 Batch prediction on records: {batch_records}")
        print(f"🔍 Using {method.upper()} method...")
        print("-" * 40)
        
        try:
            results = predictor.batch_predict(batch_records, save_summary=True)
            
            print(f"\n📊 Batch Results Summary:")
            correct_count = sum(1 for r in results if r['correct_prediction'])
            accuracy = correct_count / len(results) if results else 0
            
            print(f"   Records processed: {len(results)}")
            print(f"   Correct predictions: {correct_count}")
            print(f"   Accuracy: {accuracy:.1%}")
            
            # Show individual results
            print(f"\n📋 Individual Results:")
            for result in results:
                status = "✅" if result['correct_prediction'] else "❌"
                print(f"   Record {result['record_id']}: {result['predicted_label']} "
                      f"({result['confidence']:.1%}) {status}")
                      
        except Exception as e:
            print(f"❌ Error during batch prediction: {e}")
    
    def compare_all_methods(self):
        """Compare predictions from all methods"""
        print("\n📊 COMPARE ALL METHODS")
        print("-" * 50)
        
        record_input = self.get_input(
            f"Enter record number for comparison ({min(self.available_records)}-{max(self.available_records)})"
        )
        
        if record_input is None:
            return
        
        try:
            record_id = int(record_input)
            if record_id not in self.available_records:
                print(f"❌ Record {record_id} not found")
                return
        except ValueError:
            print("❌ Please enter a valid number")
            return
        
        print(f"\n🔍 Comparing all methods for record {record_id}...")
        print("=" * 60)
        
        results = {}
        errors = {}
        
        # Try each method
        for method in ['gan', 'mobilenet', 'resnet']:
            try:
                print(f"\n🔄 Running {method.upper()} prediction...")
                predictor = self.get_predictor(method)
                
                if predictor is not None:
                    result = predictor.predict_record(record_id, show_visualizations=True)
                    results[method] = result
                    print(f"✅ {method.upper()}: {result['predicted_label']} ({result['confidence']:.1%})")
                else:
                    errors[method] = f"Model not available (please train first)"
                    print(f"⚠️ {method.upper()}: Model not available")
                    
            except Exception as e:
                errors[method] = str(e)
                print(f"❌ {method.upper()}: Error - {e}")
        
        # Display comparison summary
        self.display_comparison_summary(results, errors, record_id)
    
    def display_comparison_summary(self, results, errors, record_id):
        """Display comparison summary of all methods"""
        print(f"\n📊 COMPARISON SUMMARY - Record {record_id}")
        print("=" * 60)
        
        if not results:
            print("❌ No successful predictions to compare.")
            return
        
        # Get true label
        record_info = self.dataset_generator.get_record_info(record_id)
        true_label = record_info['label'] if record_info else 'Unknown'
        
        print(f"True Label: {true_label.title()}")
        print("-" * 30)
        
        # Method comparison table
        method_names = {'gan': 'GAN', 'mobilenet': 'MobileNet', 'resnet': 'ResNet'}
        
        for method, method_name in method_names.items():
            if method in results:
                result = results[method]
                status = "✅" if result['correct_prediction'] else "❌"
                print(f"{method_name:10} | {result['predicted_label']:8} | {result['confidence']:6.1%} | {status}")
            elif method in errors:
                print(f"{method_name:10} | {'ERROR':8} | {'---':6} | ❌ ({errors[method][:20]}...)")
        
        print("\n📁 Detailed visualizations saved in results/prediction_results/")
    
    def get_predictor(self, method):
        """Get predictor for specific method (lazy loading)"""
        try:
            if method == 'gan':
                if self.gan_predictor is None:
                    self.gan_predictor = GANPredictor(self.base_path)
                return self.gan_predictor
            elif method == 'mobilenet':
                if self.mobilenet_predictor is None:
                    self.mobilenet_predictor = MobileNetPredictor(self.base_path)
                return self.mobilenet_predictor
            elif method == 'resnet':
                if self.resnet_predictor is None:
                    self.resnet_predictor = ResNetPredictor(self.base_path)
                return self.resnet_predictor
        except Exception as e:
            print(f"Error loading {method} predictor: {e}")
            return None
        
        return None
    
    def display_prediction_result(self, result, method):
        """Display formatted prediction result"""
        print(f"\n📋 {method.upper()} Prediction Result:")
        print("-" * 40)
        print(f"Record ID: {result['record_id']}")
        print(f"Predicted: {result['predicted_label']} ({result['confidence']:.1%} confidence)")
        print(f"True Label: {result['true_label']}")
        
        if result['correct_prediction']:
            print("Status: ✅ CORRECT")
        else:
            print("Status: ❌ INCORRECT")
        
        print(f"\nProbability Distribution:")
        for i, (label, prob) in enumerate(zip(['Normal', 'Suspect', 'Hypoxia'], result['probabilities'])):
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"  {label:8} | {bar} | {prob:.1%}")
    
    def show_system_info(self):
        """Show system and dataset information"""
        print("\n" + "="*60)
        print("ℹ️ SYSTEM INFORMATION")
        print("="*60)
        
        # Dataset info
        print("📊 Dataset Information:")
        if self.available_records:
            print(f"   Available records: {len(self.available_records)}")
            print(f"   Record range: {min(self.available_records)} - {max(self.available_records)}")
            
            # Load a sample to show label distribution
            try:
                dataset_info_file = self.base_path / 'data' / 'dataset_info.csv'
                if dataset_info_file.exists():
                    import pandas as pd
                    df = pd.read_csv(dataset_info_file)
                    label_dist = df['label'].value_counts()
                    print(f"   Label distribution:")
                    for label, count in label_dist.items():
                        pct = count / len(df) * 100
                        print(f"     {label.title()}: {count} ({pct:.1f}%)")
            except Exception as e:
                print(f"     Could not load label distribution: {e}")
        else:
            print("   No dataset loaded")
        
        # Model availability
        print(f"\n🤖 Model Availability:")
        models_dir = self.base_path / 'models'
        for method in ['gan_models', 'mobilenet_models', 'resnet_models']:
            method_dir = models_dir / method
            if method_dir.exists():
                model_files = list(method_dir.glob('*.h5'))
                status = "✅" if model_files else "⚠️ (no .h5 files)"
            else:
                status = "❌"
            print(f"   {method.replace('_models', '').upper()}: {status}")
        
        input("\nPress Enter to continue...")
    
    def main_menu(self):
        """Main application menu"""
        while True:
            print("\n" + "="*60)
            print("🏠 MAIN MENU")
            print("="*60)
            print("1. 📁 Generate Dataset (.npy files)")
            print("2. 🚀 Train Models")
            print("3. 🔍 Predict with Models")
            print("4. ℹ️ System Information")
            print("5. ❌ Exit")
            print("-" * 60)
            
            choice = self.get_input("Select an action (1-5)", ['1', '2', '3', '4', '5'])
            
            if choice is None or choice == '5':
                self.exit_application()
                break
            
            try:
                if choice == '1':
                    self.generate_dataset_menu()
                elif choice == '2':
                    self.training_menu()
                elif choice == '3':
                    self.prediction_menu()
                elif choice == '4':
                    self.show_system_info()
                    
            except KeyboardInterrupt:
                print("\n⚠️ Operation interrupted by user.")
                input("Press Enter to continue...")
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                traceback.print_exc()
                input("Press Enter to continue...")
    
    def exit_application(self):
        """Exit application gracefully"""
        print("\n" + "="*60)
        print("👋 THANK YOU FOR USING HIPOXIA DEEP LEARNING SYSTEM")
        print("="*60)
        print("📊 Session Summary:")
        
        if self.available_records:
            print(f"   Dataset: {len(self.available_records)} records available")
        else:
            print("   Dataset: Not loaded")
        
        print(f"\n🕐 Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🔬 Keep advancing fetal healthcare with AI! 🚀")
        print("\nGoodbye! 👋\n")
    
    def run(self):
        """Run the main application"""
        try:
            self.print_banner()
            
            # Load available records at startup
            print("🔄 Initializing system...")
            self.load_available_records()
            
            # Start main menu
            self.main_menu()
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Application interrupted by user.")
            self.exit_application()
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            traceback.print_exc()
            print("\n💡 Please check your installation and try again.")

def main():
    """Main entry point"""
    app = HipoxiaDeepLearningTextApp()
    app.run()

if __name__ == "__main__":
    # Check for required packages with proper import names
    package_imports = {
        'tensorflow': 'tensorflow',
        'numpy': 'numpy', 
        'pandas': 'pandas',
        'matplotlib': 'matplotlib.pyplot',
        'seaborn': 'seaborn',
        'scikit-learn': 'sklearn',  # Different import name
    }
    
    missing_packages = []
    for package_name, import_name in package_imports.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        sys.exit(1)
    
    main()