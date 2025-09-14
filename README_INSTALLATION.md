# HipoxiaDeepLearning System - Installation Guide

## 🚀 Quick Start Installation

### Method 1: Automated Installation (Recommended)
```bash
# Run the automated installer
python3 install_requirements.py
```

### Method 2: Manual Installation
```bash
# Install minimal dependencies
pip install -r requirements-minimal.txt

# Or install full dependencies
pip install -r requirements.txt
```

### Method 3: Individual Package Installation
```bash
pip install tensorflow>=2.10.0
pip install numpy pandas scipy scikit-learn
pip install matplotlib seaborn
pip install inquirer tqdm psutil
```

## 📋 System Requirements

### Minimum Requirements:
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB recommended)
- **Storage**: 5GB free space
- **OS**: Linux, macOS, or Windows

### Recommended Requirements:
- **Python**: 3.9 or 3.10
- **RAM**: 16GB or more
- **GPU**: NVIDIA GPU with CUDA support (optional)
- **Storage**: 10GB free space

## 📦 Dependencies Overview

### Core Libraries:
- **TensorFlow** (>= 2.10.0): Deep learning framework
- **NumPy** (>= 1.21.0): Numerical computing
- **Pandas** (>= 1.3.0): Data manipulation
- **SciPy** (>= 1.7.0): Scientific computing
- **Scikit-learn** (>= 1.0.0): Machine learning utilities

### Visualization:
- **Matplotlib** (>= 3.5.0): Plotting library
- **Seaborn** (>= 0.11.0): Statistical visualization

### Interface:
- **Inquirer** (>= 3.1.0): Interactive CLI menus

### Utilities:
- **tqdm** (>= 4.62.0): Progress bars
- **psutil** (>= 5.8.0): System monitoring

## 🔧 Installation Options

### Option 1: Standard Installation
For most users who want all features:
```bash
pip install -r requirements.txt
```

### Option 2: Minimal Installation
For users who want only essential features:
```bash
pip install -r requirements-minimal.txt
```

### Option 3: Development Installation
For developers who want to modify the code:
```bash
pip install -r requirements.txt
pip install pytest black flake8
```

## 🐛 Troubleshooting

### Common Issues and Solutions:

#### 1. TensorFlow Installation Issues
```bash
# For CPU-only installation
pip install tensorflow-cpu

# For specific TensorFlow version
pip install tensorflow==2.13.0
```

#### 2. Memory Issues
- Close other applications
- Use smaller batch sizes during training
- Consider using CPU-only mode

#### 3. Import Errors
```bash
# Verify installation
python3 -c "import tensorflow; print(tensorflow.__version__)"
python3 -c "import numpy; print(numpy.__version__)"
```

#### 4. Permission Errors (Linux/macOS)
```bash
# Use user installation
pip install --user -r requirements.txt
```

#### 5. Windows-specific Issues
```bash
# Use conda instead of pip
conda install tensorflow numpy pandas matplotlib seaborn scikit-learn
pip install inquirer tqdm psutil
```

## 🖥️ Platform-Specific Instructions

### Ubuntu/Debian Linux:
```bash
# Update system
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip

# Install system dependencies
sudo apt install python3-dev python3-venv

# Install requirements
pip3 install -r requirements.txt
```

### CentOS/RHEL/Fedora:
```bash
# Install Python and pip
sudo yum install python3 python3-pip  # CentOS/RHEL
sudo dnf install python3 python3-pip  # Fedora

# Install requirements
pip3 install -r requirements.txt
```

### macOS:
```bash
# Install Python (if not installed)
brew install python3

# Install requirements
pip3 install -r requirements.txt
```

### Windows:
```bash
# Using pip (in Command Prompt or PowerShell)
pip install -r requirements.txt

# Or using conda
conda create -n hipoxia python=3.9
conda activate hipoxia
conda install tensorflow numpy pandas matplotlib seaborn scikit-learn
pip install inquirer tqdm psutil
```

## 🚀 Verification

After installation, verify everything works:

```bash
# Run the verification script
python3 install_requirements.py

# Or manually test
python3 -c "
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print('✅ All imports successful!')
print(f'TensorFlow version: {tf.__version__}')
print(f'GPU available: {len(tf.config.experimental.list_physical_devices(\"GPU\")) > 0}')
"
```

## 🔄 Virtual Environment (Recommended)

Create an isolated environment for the project:

```bash
# Create virtual environment
python3 -m venv hipoxia_env

# Activate environment
source hipoxia_env/bin/activate  # Linux/macOS
# or
hipoxia_env\Scripts\activate     # Windows

# Install requirements
pip install -r requirements.txt

# When done, deactivate
deactivate
```

## 📈 Performance Optimization

### For GPU Support:
```bash
# Install CUDA-enabled TensorFlow
pip install tensorflow[and-cuda]

# Verify GPU detection
python3 -c "
import tensorflow as tf
print('GPUs Available:', len(tf.config.experimental.list_physical_devices('GPU')))
"
```

### For Better Performance:
```bash
# Install optimized libraries
pip install intel-tensorflow  # Intel CPUs
# or
pip install tensorflow-directml  # DirectML (Windows)
```

## 🆘 Getting Help

If you encounter issues:

1. **Check system requirements** above
2. **Try the automated installer**: `python3 install_requirements.py`
3. **Use minimal installation**: `pip install -r requirements-minimal.txt`
4. **Check error messages** for specific missing dependencies
5. **Create virtual environment** and try again

## ✅ Ready to Use

Once installation is complete, you can run the system:

```bash
# Start the application
python3 main.py

# Follow the interactive menus:
# 1. Generate Dataset
# 2. Train Models  
# 3. Predict with Models
```

Enjoy using the HipoxiaDeepLearning system! 🎯