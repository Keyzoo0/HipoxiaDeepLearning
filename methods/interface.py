#!/usr/bin/env python3
"""
Interface Module
Handles user interface, menu system, and system status
"""

import pandas as pd
from pathlib import Path

class Interface:
    def __init__(self, base_path, data_handler, trainer, predictor, model_builder):
        self.base_path = Path(base_path)
        self.processed_data_path = self.base_path / 'processed_data'
        self.models_path = self.base_path / 'models'

        self.data_handler = data_handler
        self.trainer = trainer
        self.predictor = predictor
        self.model_builder = model_builder

    def select_method(self, action_type="training"):
        """Method selection helper"""
        print(f"\n🔬 SELECT {action_type.upper()} METHOD")
        print("="*40)
        methods = ['gan', 'mobilenet', 'resnet', 'mdnn']
        icons = ['🤖', '📱', '🏗️', '🎯']

        for i, (method, icon) in enumerate(zip(methods, icons), 1):
            method_display = self.model_builder.get_method_display_name(method)
            description = self.model_builder.get_method_description(method)
            print(f"{i}. {icon} {method_display} Method ({description})")

        while True:
            choice = input("Select method (1-4): ").strip()
            if choice == '1':
                return 'gan'
            elif choice == '2':
                return 'mobilenet'
            elif choice == '3':
                return 'resnet'
            elif choice == '4':
                return 'mdnn'
            else:
                print("❌ Invalid choice. Please select 1-4.")

    def show_system_status(self):
        """Show system status and available data"""
        print("\n📋 SYSTEM STATUS")
        print("="*50)

        # Check data files
        clinical_file = self.processed_data_path / 'clinical_dataset.csv'
        signals_dir = self.processed_data_path / 'signals'

        print("📂 Data Files:")
        print(f"   Clinical Dataset: {'✅' if clinical_file.exists() else '❌'} {clinical_file}")
        print(f"   Signals Directory: {'✅' if signals_dir.exists() else '❌'} {signals_dir}")

        if signals_dir.exists():
            signal_files = list(signals_dir.glob("*_signals.npy"))
            print(f"   Signal Files: {len(signal_files)} files")

        # Check trained models
        available_methods = self.trainer.get_available_methods()
        print(f"\n🤖 Trained Models:")
        if available_methods:
            for method in ['mdnn', 'gan', 'mobilenet', 'resnet']:
                pkl_path = self.models_path / f'{method}_multimodal_hypoxia_detector.pkl'
                status = "✅" if method in available_methods else "❌"
                method_display = self.model_builder.get_method_display_name(method)
                print(f"   {status} {method_display} Method")
                if pkl_path.exists():
                    size_mb = pkl_path.stat().st_size / (1024*1024)
                    print(f"      File: {pkl_path.name} ({size_mb:.1f} MB)")
        else:
            # Check old naming convention
            old_pkl_path = self.models_path / 'multimodal_hypoxia_detector.pkl'
            if old_pkl_path.exists():
                print(f"   ✅ Legacy Model (multimodal_hypoxia_detector.pkl)")
            else:
                print(f"   ❌ No trained models found")

        # Load and show clinical data stats
        if clinical_file.exists():
            try:
                df = pd.read_csv(clinical_file)
                print(f"\n📊 Clinical Dataset:")
                print(f"   Total records: {len(df)}")
                if 'label' in df.columns:
                    label_counts = df['label'].value_counts()
                    print(f"   Label distribution:")
                    for label, count in label_counts.items():
                        print(f"     {label}: {count}")
            except Exception as e:
                print(f"   Error reading clinical data: {e}")

        print(f"\n📁 Available Methods for Training:")
        for method in ['mdnn', 'gan', 'mobilenet', 'resnet']:
            method_display = self.model_builder.get_method_display_name(method)
            description = self.model_builder.get_method_description(method)
            print(f"   🔬 {method_display}: {description}")

    def generate_journal_analysis(self):
        """Generate comprehensive journal analysis"""
        print("\n📰 GENERATING COMPREHENSIVE JOURNAL ANALYSIS")
        print("="*60)
        print("🔬 Creating publication-ready visualizations and reports...")
        print("📊 This will generate:")
        print("   • Method comparison charts")
        print("   • Detailed MDNN analysis")
        print("   • Prediction demonstrations")
        print("   • Statistical analysis tables")
        print("   • LaTeX-ready tables")
        print("   • Comprehensive text report")

        # Import and run the journal analysis
        try:
            import sys
            sys.path.append(str(self.base_path))

            # Import the journal analysis class
            from simple_journal_analysis import SimpleJournalAnalysis

            # Create and run analysis
            analyzer = SimpleJournalAnalysis()
            analyzer.run_complete_analysis()

            print("\n✅ JOURNAL ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"📁 All files saved in: {analyzer.journal_path}")
            print("\n📊 Generated Files:")

            # List generated files
            files = list(analyzer.journal_path.glob('*'))
            for file_path in sorted(files):
                size = file_path.stat().st_size / 1024  # KB
                print(f"   📄 {file_path.name} ({size:.1f} KB)")

            print(f"\n🎯 READY FOR JOURNAL SUBMISSION!")
            print("   • High-resolution figures (300 DPI)")
            print("   • Statistical analysis tables")
            print("   • LaTeX format ready")
            print("   • Complete methodology description")

        except ImportError as e:
            print(f"❌ Could not import journal analysis module: {e}")
            print("💡 Make sure simple_journal_analysis.py is available")
        except Exception as e:
            print(f"❌ Error during journal analysis: {e}")

    def interactive_menu(self):
        """Interactive menu for training and prediction"""
        while True:
            print("\n" + "="*60)
            print("🧬 MULTIMODAL HYPOXIA DETECTION SYSTEM")
            print("="*60)
            print("1. 🎯 Train Single Model (Signal + Clinical)")
            print("2. 🚀 Train All Methods (MDNN, GAN, MobileNet, ResNet)")
            print("3. 🔮 Predict Single Record")
            print("4. 📊 Batch Prediction")
            print("5. 🆚 Compare All Methods")
            print("6. 📋 Show System Status")
            print("7. ❌ Exit")

            choice = input("\nSelect option (1-7): ").strip()

            if choice == '1':
                try:
                    method = self.select_method("training")
                    method_display = self.model_builder.get_method_display_name(method)
                    print(f"\n🚀 Starting {method_display} training...")
                    history, accuracy = self.trainer.train_model(method)
                    method_display = self.model_builder.get_method_display_name(method)
                    print(f"\n✅ {method_display} training completed! Final accuracy: {accuracy:.4f}")
                except Exception as e:
                    print(f"❌ Training error: {e}")

            elif choice == '2':
                try:
                    self.train_all_methods()
                except Exception as e:
                    print(f"❌ Training all methods error: {e}")

            elif choice == '3':
                try:
                    available_methods = self.trainer.get_available_methods()
                    if not available_methods:
                        print("❌ No trained models found. Please train a model first.")
                        continue

                    record_id = int(input("Enter record ID: "))

                    if len(available_methods) == 1:
                        method = available_methods[0]
                        method_display = self.model_builder.get_method_display_name(method)
                        print(f"Using available method: {method_display}")
                    else:
                        print(f"Available methods: {', '.join(m.upper() for m in available_methods)}")
                        method = self.select_method("prediction")

                    self.trainer.model = None  # Reset to load correct method
                    result = self.predictor.predict_single_record(record_id, method)
                except ValueError:
                    print("❌ Please enter a valid record ID")
                except Exception as e:
                    print(f"❌ Prediction error: {e}")

            elif choice == '4':
                try:
                    available_methods = self.trainer.get_available_methods()
                    if not available_methods:
                        print("❌ No trained models found. Please train a model first.")
                        continue

                    records_input = input("Enter record IDs (comma-separated): ")
                    record_ids = [int(x.strip()) for x in records_input.split(',')]

                    if len(available_methods) == 1:
                        method = available_methods[0]
                        method_display = self.model_builder.get_method_display_name(method)
                        print(f"Using available method: {method_display}")
                    else:
                        print(f"Available methods: {', '.join(m.upper() for m in available_methods)}")
                        method = self.select_method("batch prediction")

                    self.trainer.model = None  # Reset to load correct method
                    results = self.predictor.predict_batch_records(record_ids, method)

                except ValueError:
                    print("❌ Please enter valid record IDs")
                except Exception as e:
                    print(f"❌ Batch prediction error: {e}")

            elif choice == '5':
                try:
                    record_id = int(input("Enter record ID for comparison: "))
                    self.predictor.compare_all_methods(record_id)
                except ValueError:
                    print("❌ Please enter a valid record ID")
                except Exception as e:
                    print(f"❌ Comparison error: {e}")

            elif choice == '6':
                self.show_system_status()

            elif choice == '7':
                print("👋 Thank you for using the Multimodal Hypoxia Detection System!")
                break

            else:
                print("❌ Invalid choice. Please select 1-7.")

            input("\nPress Enter to continue...")

    def train_all_methods(self):
        """Train all available methods sequentially"""
        print("\n🚀 TRAINING ALL METHODS")
        print("="*50)
        print("Will train: MDNN → GAN → MobileNet → ResNet")
        print("This may take 20-40 minutes depending on your hardware.")

        confirm = input("\nContinue with training all methods? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ Training cancelled.")
            return

        methods = ['mdnn', 'gan', 'mobilenet', 'resnet']
        results = {}

        for i, method in enumerate(methods, 1):
            method_display = self.model_builder.get_method_display_name(method)
            print(f"\n🔄 [{i}/4] Training {method_display} Method...")
            print("="*40)

            try:
                history, accuracy = self.trainer.train_model(method)
                results[method] = {
                    'accuracy': accuracy,
                    'status': 'completed'
                }
                print(f"✅ {method_display} completed! Accuracy: {accuracy:.4f}")

            except Exception as e:
                results[method] = {
                    'accuracy': 0.0,
                    'status': 'failed',
                    'error': str(e)
                }
                print(f"❌ {method_display} failed: {e}")

            # Short pause between methods
            if i < len(methods):
                print("⏳ Preparing next method...")

        # Summary
        print(f"\n📊 TRAINING SUMMARY")
        print("="*50)

        for method, result in results.items():
            method_display = self.model_builder.get_method_display_name(method)
            if result['status'] == 'completed':
                print(f"✅ {method_display:<12}: {result['accuracy']:.4f} accuracy")
            else:
                print(f"❌ {method_display:<12}: Failed ({result.get('error', 'Unknown error')})")

        # Find best method
        completed_methods = {k: v for k, v in results.items() if v['status'] == 'completed'}
        if completed_methods:
            best_method = max(completed_methods.items(), key=lambda x: x[1]['accuracy'])
            best_method_display = self.model_builder.get_method_display_name(best_method[0])
            print(f"\n🏆 Best Method: {best_method_display} ({best_method[1]['accuracy']:.4f} accuracy)")
        else:
            print(f"\n⚠️  No methods completed successfully.")

        print(f"\n✅ All methods training completed!")