#!/usr/bin/env python3
"""
Modular Multimodal Hypoxia Detection System
Clean and modular implementation with separated components
"""

import warnings
warnings.filterwarnings('ignore')

# Import modular components
from methods.data_handler import DataHandler
from methods.model_builder import ModelBuilder
from methods.trainer import ModelTrainer
from methods.predictor import ModelPredictor
from methods.visualizer import Visualizer
from methods.interface import Interface

class MultimodalHypoxiaDetector:
    """Main system class that coordinates all modules"""

    def __init__(self, base_path='/home/zainul/joki/HipoxiaDeepLearning'):
        self.base_path = base_path

        print("🚀 Initializing Modular Multimodal Hypoxia Detection System...")
        print("="*60)

        # Initialize all modules
        self.data_handler = DataHandler(base_path)
        self.model_builder = ModelBuilder()
        self.visualizer = Visualizer(base_path, self.model_builder)
        self.trainer = ModelTrainer(base_path, self.data_handler, self.model_builder, self.visualizer)
        self.predictor = ModelPredictor(self.data_handler, self.trainer, self.visualizer, self.model_builder)
        self.interface = Interface(base_path, self.data_handler, self.trainer, self.predictor, self.model_builder)

        print("✅ All modules initialized successfully!")
        print("\n📋 Available Modules:")
        print("   🗃️  DataHandler - Clinical data and signal processing")
        print("   🏗️  ModelBuilder - Neural network architectures")
        print("   🎯 ModelTrainer - Training and evaluation")
        print("   🔮 ModelPredictor - Single and batch predictions")
        print("   📊 Visualizer - Training and prediction visualizations")
        print("   🖥️  Interface - User interface and system status")

    def run(self):
        """Run the interactive system"""
        print(f"\n🎯 Starting interactive system...")
        self.interface.interactive_menu()

    def train_model(self, method='mdnn'):
        """Train a model with specified method"""
        return self.trainer.train_model(method)

    def predict_record(self, record_id, method='mdnn'):
        """Predict hypoxia for a single record"""
        return self.predictor.predict_single_record(record_id, method)

    def compare_methods(self, record_id):
        """Compare all available methods for a record"""
        return self.predictor.compare_all_methods(record_id)

    def show_status(self):
        """Show system status"""
        self.interface.show_system_status()

    def generate_dataset(self):
        """Generate multimodal dataset"""
        return self.data_handler.generate_multimodal_dataset()

def main():
    """Main function"""
    try:
        # Initialize the modular system
        detector = MultimodalHypoxiaDetector()

        # Run interactive menu
        detector.run()

    except KeyboardInterrupt:
        print("\n\n👋 System interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        print("💡 Please check your installation and data files.")

if __name__ == "__main__":
    main()