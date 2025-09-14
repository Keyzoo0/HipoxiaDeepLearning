#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import inquirer
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

class HipoxiaDeepLearningApp:
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
        
        method_choices = [
            inquirer.List('method',
                         message="Select training method",
                         choices=[
                             ('🤖 GAN Method (Data Augmentation + Classification)', 'gan'),
                             ('📱 MobileNet Method (Lightweight CNN)', 'mobilenet'),
                             ('🏗️ ResNet Method (Deep Residual Network)', 'resnet'),
                             ('🔙 Back to Main Menu', 'back')
                         ])
        ]
        
        method_answer = inquirer.prompt(method_choices)
        if not method_answer or method_answer['method'] == 'back':
            return
        
        selected_method = method_answer['method']
        
        print(f"\n🔄 Starting {selected_method.upper()} training...")
        print("-" * 50)
        
        try:
            if selected_method == 'gan':
                success = train_gan()
            elif selected_method == 'mobilenet':
                success = train_mobilenet()
            elif selected_method == 'resnet':
                success = train_resnet()
            
            if success:
                print(f"\n✅ {selected_method.upper()} training completed successfully!")
            else:
                print(f"\n❌ {selected_method.upper()} training failed!")
                
        except Exception as e:
            print(f"\n❌ Error during {selected_method.upper()} training: {e}")
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
        
        method_choices = [
            inquirer.List('method',
                         message="Select prediction method",
                         choices=[
                             ('🤖 GAN Method Prediction', 'gan'),
                             ('📱 MobileNet Method Prediction', 'mobilenet'),
                             ('🏗️ ResNet Method Prediction', 'resnet'),
                             ('📊 Compare All Methods', 'compare'),
                             ('🔙 Back to Main Menu', 'back')
                         ])
        ]
        
        method_answer = inquirer.prompt(method_choices)
        if not method_answer or method_answer['method'] == 'back':
            return
        
        selected_method = method_answer['method']
        
        if selected_method == 'compare':
            self.compare_all_methods()
        else:
            self.single_method_prediction(selected_method)
    
    def single_method_prediction(self, method):
        """Handle single method prediction"""
        print(f"\n🔍 {method.upper()} PREDICTION")
        print("-" * 50)
        
        # Show available records info
        print(f"Available records: {len(self.available_records)} total")
        print(f"Record range: {min(self.available_records)} - {max(self.available_records)}")
        
        # Record selection options
        record_choices = [
            inquirer.List('record_option',
                         message="Select record input option",
                         choices=[
                             ('📝 Enter specific record number', 'manual'),
                             ('🎲 Use random record', 'random'),
                             ('📊 Batch predict (first 5 records)', 'batch'),
                             ('🔙 Back to Prediction Menu', 'back')
                         ])
        ]
        
        record_answer = inquirer.prompt(record_choices)
        if not record_answer or record_answer['record_option'] == 'back':
            return
        
        record_option = record_answer['record_option']
        
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
            if record_option == 'manual':
                self.manual_record_prediction(predictor, method)
            elif record_option == 'random':
                self.random_record_prediction(predictor, method)
            elif record_option == 'batch':
                self.batch_record_prediction(predictor, method)
                
        except Exception as e:
            print(f"❌ Error during {method.upper()} prediction: {e}")
            traceback.print_exc()
        
        input("\nPress Enter to continue...")
    
    def manual_record_prediction(self, predictor, method):
        """Handle manual record number input"""
        while True:
            record_input = inquirer.text(
                message=f"Enter record number ({min(self.available_records)}-{max(self.available_records)})",
                validate=lambda _, x: x.isdigit() and int(x) in self.available_records
            )
            
            if record_input is None:
                break
                
            record_id = int(record_input)
            
            print(f"\n🔍 Predicting record {record_id} using {method.upper()} method...")
            print("-" * 40)
            
            try:
                result = predictor.predict_record(record_id, show_visualizations=True)
                self.display_prediction_result(result, method)
                
                # Ask if user wants to predict another record
                continue_choice = inquirer.confirm(
                    message="Predict another record?",
                    default=False
                )
                
                if not continue_choice:
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
        
        # Record selection
        record_input = inquirer.text(
            message=f"Enter record number for comparison ({min(self.available_records)}-{max(self.available_records)})",
            validate=lambda _, x: x.isdigit() and int(x) in self.available_records
        )
        
        if record_input is None:
            return
        
        record_id = int(record_input)
        
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
        
        # Consensus analysis
        if len(results) >= 2:
            predictions = [r['predicted_label'] for r in results.values()]
            confidences = [r['confidence'] for r in results.values()]
            
            print(f"\n🤔 Consensus Analysis:")
            
            # Check if all methods agree
            if len(set(predictions)) == 1:
                consensus_pred = predictions[0]
                avg_confidence = sum(confidences) / len(confidences)
                print(f"   All methods agree: {consensus_pred} (avg confidence: {avg_confidence:.1%})")
            else:
                # Count votes
                from collections import Counter
                vote_counts = Counter(predictions)
                majority_pred = vote_counts.most_common(1)[0][0]
                majority_count = vote_counts.most_common(1)[0][1]
                
                print(f"   Majority prediction: {majority_pred} ({majority_count}/{len(predictions)} methods)")
                print(f"   Disagreement detected - manual review recommended")
        
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
    
    def view_results_menu(self):
        """View training results and statistics"""
        print("\n" + "="*60)
        print("📊 TRAINING RESULTS & STATISTICS")
        print("="*60)
        
        results_dir = self.base_path / 'results'
        if not results_dir.exists():
            print("⚠️ No results found. Please run training first.")
            input("Press Enter to continue...")
            return
        
        view_choices = [
            inquirer.List('view_option',
                         message="Select what to view",
                         choices=[
                             ('📈 Training Plots', 'training_plots'),
                             ('🔍 Prediction Results', 'prediction_results'),
                             ('📁 Open Results Directory', 'open_dir'),
                             ('🔙 Back to Main Menu', 'back')
                         ])
        ]
        
        view_answer = inquirer.prompt(view_choices)
        if not view_answer or view_answer['view_option'] == 'back':
            return
        
        view_option = view_answer['view_option']
        
        try:
            if view_option == 'training_plots':
                self.show_training_plots()
            elif view_option == 'prediction_results':
                self.show_prediction_results()
            elif view_option == 'open_dir':
                self.open_results_directory()
        except Exception as e:
            print(f"❌ Error viewing results: {e}")
        
        input("\nPress Enter to continue...")
    
    def show_training_plots(self):
        """Show available training plots"""
        plots_dir = self.base_path / 'results' / 'training_plots'
        
        if not plots_dir.exists():
            print("⚠️ No training plots found.")
            return
        
        plot_files = list(plots_dir.glob('*.png'))
        if not plot_files:
            print("⚠️ No plot files found.")
            return
        
        print(f"\n📈 Available Training Plots:")
        for plot_file in plot_files:
            print(f"   📊 {plot_file.name}")
        
        print(f"\n📁 Plots location: {plots_dir}")
        print("   Open the directory to view plot images")
    
    def show_prediction_results(self):
        """Show prediction results summary"""
        pred_dir = self.base_path / 'results' / 'prediction_results'
        
        if not pred_dir.exists():
            print("⚠️ No prediction results found.")
            return
        
        for method in ['gan_predictions', 'mobilenet_predictions', 'resnet_predictions']:
            method_dir = pred_dir / method
            if method_dir.exists():
                files = list(method_dir.glob('*.png')) + list(method_dir.glob('*.csv'))
                print(f"\n🔍 {method.replace('_', ' ').title()}:")
                print(f"   📁 Location: {method_dir}")
                print(f"   📊 Files: {len(files)} result files")
    
    def open_results_directory(self):
        """Open results directory in file manager"""
        results_dir = self.base_path / 'results'
        
        try:
            import subprocess
            if sys.platform == 'win32':
                subprocess.run(['explorer', str(results_dir)])
            elif sys.platform == 'darwin':
                subprocess.run(['open', str(results_dir)])
            else:
                subprocess.run(['xdg-open', str(results_dir)])
            
            print(f"✅ Opened results directory: {results_dir}")
        except Exception as e:
            print(f"❌ Could not open directory: {e}")
            print(f"📁 Results location: {results_dir}")
    
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
        
        print(f"\n📁 Project Structure:")
        print(f"   Base path: {self.base_path}")
        print(f"   Models: {(self.base_path / 'models').exists()}")
        print(f"   Results: {(self.base_path / 'results').exists()}")
        print(f"   Data: {(self.base_path / 'data').exists()}")
        
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
            
            menu_choices = [
                inquirer.List('action',
                             message="Select an action",
                             choices=[
                                 ('📁 Generate Dataset (.npy files)', 'generate'),
                                 ('🚀 Train Models', 'train'),
                                 ('🔍 Predict with Models', 'predict'),
                                 ('📊 View Results & Statistics', 'results'),
                                 ('ℹ️ System Information', 'info'),
                                 ('❌ Exit', 'exit')
                             ])
            ]
            
            answer = inquirer.prompt(menu_choices)
            
            if not answer or answer['action'] == 'exit':
                self.exit_application()
                break
            
            action = answer['action']
            
            try:
                if action == 'generate':
                    self.generate_dataset_menu()
                elif action == 'train':
                    self.training_menu()
                elif action == 'predict':
                    self.prediction_menu()
                elif action == 'results':
                    self.view_results_menu()
                elif action == 'info':
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
        
        # Check for any generated results
        results_dir = self.base_path / 'results'
        if results_dir.exists():
            result_files = len(list(results_dir.rglob('*.png'))) + len(list(results_dir.rglob('*.csv')))
            print(f"   Results: {result_files} files generated")
        
        print(f"\n🕐 Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n📋 For questions or issues, check the documentation or logs.")
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
            input("Press Enter to exit...")

def main():
    """Main entry point"""
    app = HipoxiaDeepLearningApp()
    app.run()

if __name__ == "__main__":
    # Check for required packages
    required_packages = [
        'tensorflow', 'numpy', 'pandas', 'matplotlib', 
        'seaborn', 'scikit-learn', 'inquirer'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        sys.exit(1)
    
    main()