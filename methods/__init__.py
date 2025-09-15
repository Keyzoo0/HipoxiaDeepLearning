#!/usr/bin/env python3
"""
Methods Package for Multimodal Hypoxia Detection System
Contains all modular components for clean code organization
"""

from .data_handler import DataHandler
from .model_builder import ModelBuilder, focal_loss_fixed
from .trainer import ModelTrainer
from .predictor import ModelPredictor
from .visualizer import Visualizer
from .interface import Interface

__all__ = [
    'DataHandler',
    'ModelBuilder',
    'ModelTrainer',
    'ModelPredictor',
    'Visualizer',
    'Interface',
    'focal_loss_fixed'
]

__version__ = '1.0.0'
__author__ = 'Multimodal Hypoxia Detection Team'
__description__ = 'Modular components for multimodal fetal hypoxia detection using deep learning'