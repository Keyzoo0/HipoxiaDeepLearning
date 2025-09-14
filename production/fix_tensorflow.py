#!/usr/bin/env python3
"""
Fix TensorFlow and Keras compatibility issues
"""

import subprocess
import sys
import pkg_resources

def uninstall_conflicting_packages():
    """Uninstall potentially conflicting packages"""
    packages_to_remove = ['tensorflow', 'keras', 'tf-keras']
    
    for package in packages_to_remove:
        try:
            print(f"🔄 Uninstalling {package}...")
            subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', package], 
                         capture_output=True, text=True)
        except Exception as e:
            print(f"⚠️ Could not uninstall {package}: {e}")

def install_compatible_tensorflow():
    """Install compatible TensorFlow version"""
    try:
        print("🔄 Installing compatible TensorFlow...")
        
        # Install specific TensorFlow version that works well
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            'tensorflow==2.13.0'
        ])
        
        print("✅ TensorFlow 2.13.0 installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install TensorFlow: {e}")
        return False

def test_tensorflow():
    """Test TensorFlow import"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Test basic functionality
        print("🔄 Testing basic TensorFlow operations...")
        x = tf.constant([[1, 2], [3, 4]])
        y = tf.constant([[1, 1], [0, 1]])
        result = tf.matmul(x, y)
        print("✅ TensorFlow operations working")
        
        # Test GPU availability
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU available: {len(gpus)} device(s)")
        else:
            print("ℹ️ No GPU detected, using CPU")
        
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        return False

def main():
    """Main function to fix TensorFlow issues"""
    print("🔧 Fixing TensorFlow and Keras compatibility issues...")
    print("=" * 60)
    
    try:
        # Step 1: Uninstall conflicting packages
        print("\n📦 Step 1: Removing conflicting packages...")
        uninstall_conflicting_packages()
        
        # Step 2: Install compatible TensorFlow
        print("\n📦 Step 2: Installing compatible TensorFlow...")
        if not install_compatible_tensorflow():
            print("❌ Failed to install TensorFlow")
            return False
        
        # Step 3: Test installation
        print("\n🧪 Step 3: Testing TensorFlow...")
        if not test_tensorflow():
            print("❌ TensorFlow test failed")
            return False
        
        print("\n🎉 TensorFlow installation fixed successfully!")
        print("✅ You can now run: python3 main.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)