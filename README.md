# CBM-Net: Automatic Detection and Separation of Cicada Chorus from Bee Wingbeat Audios
This repository contains the official implementation of the paper "CBM-Net: Automatic Detection and Separation of Cicada Chorus from Bee Wingbeat Audios".

## 📋 Project Overview

An audio separation system based on the CNN14 pre-trained model that separates target signals from background noise in mixed audio (bee wingbeats + cicada chorus).

## 🔗 Model Weights

| Weight Type | Description |
|------------|-------------|
| General Pre-trained Weights | Cnn14_16k model pre-trained on AudioSet |

## 🚀 Quick Start

### 1. Environment Setup

```bash
conda create -n cbmnet python=3.8
conda activate cbmnet
pip install -r requirements.txt
### 2. Configuration
Before training, modify the following settings in config.py:(Required) Project Path:
python
PROJECT_PATH = '/your/project/path'   # Change to your project root directory
(Required) Dataset Paths:
python
TRAIN_DATASET_PATH = os.path.join(PROJECT_PATH, 'your_train.xlsx')
VAL_DATASET_PATH = os.path.join(PROJECT_PATH, 'your_val.xlsx')
(Required) Pre-trained Weights Path:
python
GENERAL_PRETRAINED_PATH = os.path.join(PROJECT_PATH, 'your_pretrained.pth')
(Optional) Training Hyperparameters:
python
BATCH_SIZE = 4          # Adjust based on GPU memory
NUM_EPOCHS = 50         # Number of training epochs
ENCODER_LR = 1e-5       # Encoder learning rate
DECODER_LR = 1e-4       # Decoder learning rate
(Required) Model Save Paths:
python
MODEL_SAVE_PATH = "your_output_dir/model.pth"
BEST_MODEL_SAVE_PATH = "your_output_dir/model_best.pth"
(Optional) Loss Function Weights:
python
SI_SNR_WEIGHT = 1.0           # SI-SNR loss weight
SPECTRAL_WEIGHT = 0.1         # Spectral loss weight
BEE_FREQ_BAND = (150, 500)    # Target audio frequency band (Hz)
### 3. Data Preparation
The Excel file should contain three columns with audio file paths:
mixed_audio	clean_audio	noise_audio
/path/to/mixed_1.wav	/path/to/clean_1.wav	/path/to/noise_1.wav
/path/to/mixed_2.wav	/path/to/clean_2.wav	/path/to/noise2.wav |
mixed_audio: Mixed audio containing both bee wingbeats and cicada chorus
clean_audio: Clean target audio (cicada chorus or bee wingbeats)
noise_audio: Background noise audio
### 4. Start Training
bash
python train.py
During training, the following files will be generated:
training_metrics.xlsx: Training metrics for each epoch
training_trends.png: Visualization of training progress
Model checkpoints saved to the paths specified in config
### 5.Testing
Test Single Audio File
bash
python test.py --mode custom --mixed_audio /path/to/your/mixed_audio.wav
Batch Test from Excel File
bash
python test.py --mode batch --excel_file /path/to/your/file_list.xlsx
The Excel file should contain a column named mixed_audio with paths to audio files.
Test Options
Option	Description	Default
--------	-------------	---------
--mode	Test mode: custom or batch	custom
--model_path	Path to trained model	Config.BEST_MODEL_SAVE_PATH
--mixed_audio	Path to mixed audio (custom mode)	-
--excel_file	Path to Excel file (batch mode)	-
--column_name	Column name in Excel file	mixed_audio
--output_dir	Output directory	test_results
## 📁 Code Structure
### File	Description
train.py	Main training script with model definition, loss functions, and training loop
dataset.py	Dataset class for audio loading and preprocessing
config.py	Configuration file with all adjustable parameters
test.py	Testing script for single or batch audio processing
train_process_visual.py	Training metrics visualization and analysis
requirements.txt	Python dependencies
## 📊 Model Architecture
The CBM-Net model consists of:
Encoder: Cnn14 backbone (pre-trained on AudioSet) for feature extraction
Decoder: Complex mask decoder for predicting time-frequency masks
Loss Function: Combined SI-SNR loss (weight=1.0) and spectral loss (weight=0.1)
