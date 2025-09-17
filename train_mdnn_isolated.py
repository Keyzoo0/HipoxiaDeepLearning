#!/usr/bin/env python3
import sys
import signal
import os

# Complete signal isolation
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)

# Add project to path
sys.path.insert(0, "/home/zainul/joki/HipoxiaDeepLearning")

# Import and run isolated trainer
from isolated_trainer_template import train_isolated_model

if __name__ == "__main__":
    train_isolated_model("mdnn", "/home/zainul/joki/HipoxiaDeepLearning", "/home/zainul/joki/HipoxiaDeepLearning/parallel_progress/mdnn_progress.txt")
