#!/usr/bin/env python3
"""
Automated installation script for HipoxiaDeepLearning dependencies
"""

import subprocess
import sys
import platform
import pkg_resources
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {platform.python_version()}")
        sys.exit(1)
    else:
        print(f"✅ Python version: {platform.python_version()}")

def check_pip():
    """Check if pip is available"""
    try:
        import pip
        print(f"✅ pip is available")
    except ImportError:
        print("❌ pip is not available!")
        print("   Please install pip first")
        sys.exit(1)

def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "--upgrade", package_name
        ])
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e}")
        return False

def check_installed_packages():
    """Check which packages are already installed"""
    installed_packages = [pkg.project_name for pkg in pkg_resources.working_set]
    
    required_packages = [
        'tensorflow', 'numpy', 'pandas', 'scipy', 
        'scikit-learn', 'matplotlib', 'seaborn', 
        'inquirer', 'tqdm', 'psutil'
    ]
    
    missing_packages = []
    installed_required = []
    
    for package in required_packages:
        if any(package.lower() in pkg.lower() for pkg in installed_packages):
            installed_required.append(package)
        else:
            missing_packages.append(package)
    
    return installed_required, missing_packages

def install_requirements():
    """Main installation function"""
    print("🚀 HipoxiaDeepLearning Dependencies Installation")
    print("=" * 55)
    
    # Check system requirements
    check_python_version()
    check_pip()
    
    # Check what's already installed
    print("\n🔍 Checking installed packages...")
    installed, missing = check_installed_packages()
    
    if installed:
        print(f"✅ Already installed: {', '.join(installed)}")
    
    if not missing:
        print("\n🎉 All required packages are already installed!")
        return True
    
    print(f"\n📦 Need to install: {', '.join(missing)}")
    
    # Choose installation method
    base_path = Path(__file__).parent
    requirements_file = base_path / "requirements-minimal.txt"
    
    if requirements_file.exists():
        print(f"\n🔧 Installing from {requirements_file.name}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "-r", str(requirements_file)
            ])
            print("✅ Installation completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Installation failed: {e}")
            print("\n💡 Trying individual package installation...")
    
    # Fallback: install packages individually
    success_count = 0
    for package in missing:
        print(f"\n📦 Installing {package}...")
        if install_package(package):
            print(f"✅ {package} installed successfully")
            success_count += 1
        else:
            print(f"❌ Failed to install {package}")
    
    if success_count == len(missing):
        print(f"\n🎉 All {success_count} packages installed successfully!")
        return True
    else:
        print(f"\n⚠️ {success_count}/{len(missing)} packages installed")
        print("   Some packages may need manual installation")
        return False

def verify_installation():
    """Verify that all packages can be imported"""
    print("\n🔍 Verifying installation...")
    
    test_imports = {
        'tensorflow': 'import tensorflow as tf',
        'numpy': 'import numpy as np',
        'pandas': 'import pandas as pd',
        'scipy': 'import scipy',
        'sklearn': 'import sklearn',
        'matplotlib': 'import matplotlib.pyplot as plt',
        'seaborn': 'import seaborn as sns',
        'inquirer': 'import inquirer',
        'tqdm': 'import tqdm',
        'psutil': 'import psutil'
    }
    
    success_count = 0
    for package, import_cmd in test_imports.items():
        try:
            exec(import_cmd)
            print(f"✅ {package}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {package}: {e}")
    
    if success_count == len(test_imports):
        print(f"\n🎉 All {success_count} packages verified successfully!")
        return True
    else:
        print(f"\n⚠️ {success_count}/{len(test_imports)} packages working")
        return False

def show_system_info():
    """Show system information"""
    print(f"\n💻 System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Python: {platform.python_version()}")
    print(f"   Architecture: {platform.machine()}")
    
    # Check GPU availability (if TensorFlow is installed)
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"   GPU: {len(gpus)} GPU(s) available")
        else:
            print(f"   GPU: No GPU detected (CPU only)")
    except:
        print(f"   GPU: Cannot detect (TensorFlow not available)")

def main():
    """Main function"""
    try:
        # Install packages
        if install_requirements():
            # Verify installation
            if verify_installation():
                show_system_info()
                print("\n🚀 Ready to run HipoxiaDeepLearning system!")
                print("   Run: python3 main.py")
                return True
            else:
                print("\n⚠️ Installation completed but verification failed")
                print("   Please check error messages above")
                return False
        else:
            print("\n❌ Installation failed")
            print("   Please install missing packages manually:")
            print("   pip install -r requirements-minimal.txt")
            return False
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Installation interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)