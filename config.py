#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config file for training and testing
"""

import torch
import os

class Config:
    # ==================== Project Path ====================
    PROJECT_PATH = r'your/project/path'  # Replace with your project path
    
    # ==================== Dataset Path ====================
    TRAIN_DATASET_PATH = os.path.join(PROJECT_PATH, 'your/train/dataset/path')  # Replace with your train dataset path
    VAL_DATASET_PATH = os.path.join(PROJECT_PATH, 'your/val/dataset/path')  # Replace with your val dataset path
    
    # ==================== Audio Parameters ====================
    SAMPLE_RATE = 16000
    DURATION = 10
    TARGET_LENGTH = SAMPLE_RATE * DURATION
    
    # ==================== STFT Parameters ====================
    N_FFT = 512
    HOP_LENGTH = 80
    WIN_LENGTH = 512
    
    # ==================== Training Hyperparameters ====================
    BATCH_SIZE = 4
    ENCODER_LR = 1e-5      # Encoder learning rate
    DECODER_LR = 1e-4      # Decoder learning rate
    NUM_EPOCHS = 50
    WEIGHT_DECAY = 1e-3
    
    # ==================== Loss Function Weights ====================
    SI_SNR_WEIGHT = 1.0
    SPECTRAL_WEIGHT = 0.1
    
    # ==================== Bee Frequency Band Configuration (for spectral loss) ====================
    BEE_FREQ_BAND = (150, 500)
    
    # ==================== Model Save Path ====================
    MODEL_SAVE_PATH = "your/model/save/path"  # Replace with your model save path
    BEST_MODEL_SAVE_PATH = "your/best/model/save/path"  # Replace with your best model save path
    
    # ==================== Pretrained Model Path ====================
    GENERAL_PRETRAINED_PATH = os.path.join(PROJECT_PATH, 'your/pretrained/model/path')  # Replace with your pretrained model path
    
    # ==================== Device Configuration ====================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    
    