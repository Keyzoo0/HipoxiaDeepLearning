#!/usr/bin/env python3
"""
Simple Main Entry Point for Multimodal Hypoxia Detection System
Uses modular architecture for clean, maintainable code
"""

import warnings
warnings.filterwarnings('ignore')

from main_modular import MultimodalHypoxiaDetector

def main():
    """Simple main function - delegates to modular system"""
    try:
        print("🧬 Multimodal Fetal Hypoxia Detection System")
        print("=" * 50)

        # Initialize and run the modular system
        detector = MultimodalHypoxiaDetector()
        detector.run()

    except KeyboardInterrupt:
        print("\n\n👋 System interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        print("💡 Please check your installation and data files.")
        print("💡 Use main_modular.py for detailed debugging.")

if __name__ == "__main__":
    main()