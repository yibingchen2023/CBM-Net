# test.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test for one audio：
python test.py --mode custom --mixed_audio /path/to/your/mixed_audio.wav
test for batch audio:
python test.py --mode batch --excel_file /path/to/your/file_list.xlsx
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import os
from pathlib import Path

from train import Cnn14Separator
from config import Config
from dataset import AudioSeparationDataset, create_separation_data_loaders

def load_test_model(model_path, device):
    """load trained model"""
    print(f"Loading model from: {model_path}")
    model = Cnn14Separator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully")
    return model

def load_audio(file_path, sr=Config.SAMPLE_RATE, duration=Config.DURATION):
    """Load and preprocess audio file"""
    target_length = sr * duration
    
    # load audio
    audio, _ = librosa.load(file_path, sr=sr)
    
    # process audio length
    if len(audio) > target_length:
        # take middle part
        start = (len(audio) - target_length) // 2
        audio = audio[start:start + target_length]
    else:
        # pad
        audio = np.pad(audio, (0, max(0, target_length - len(audio))))
    
    # convert to mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=0)
        
    return torch.FloatTensor(audio)

def rms_normalize(audio, target_rms=0.1):
    """
    # RMS normalization
    """
    if isinstance(audio, torch.Tensor):
        rms = torch.sqrt(torch.mean(audio**2) + 1e-8)
        if rms > 1e-6:  # avoid division by zero
            audio = audio * (target_rms / rms)
    else:
        rms = np.sqrt(np.mean(audio**2) + 1e-8)
        if rms > 1e-6:  # avoid division by zero
            audio = audio * (target_rms / rms)
    return audio

def save_audio(audio, file_path, sr=Config.SAMPLE_RATE):
    """Save audio file"""
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    
    # perform RMS normalization for better listening
    audio = rms_normalize(audio, target_rms=0.1)
    
    # ensure audio does not clip
    max_val = np.max(np.abs(audio))
    if max_val > 0.95:
        audio = audio * (0.95 / max_val)
    
    sf.write(file_path, audio, sr)
    print(f"Audio saved to: {file_path}")

def test_single_audio(model, mixed_path, device, output_dir, normalize_loudness=True, output_prefix=""):
    """Test single audio"""
    print(f"Processing audio: {mixed_path}")
    
    # load mixed audio
    mixed_audio = load_audio(mixed_path).to(device)
    
    # model inference
    with torch.no_grad():
        pred_clean, pred_noise = model(mixed_audio.unsqueeze(0))  # add batch dimension
    
    # save separation results
    mixed_name = Path(mixed_path).stem
    
    # save loudness normalized results
    clean_output_path = os.path.join(output_dir, f"{output_prefix}{mixed_name}_predicted_clean_normalized.wav")
    noise_output_path = os.path.join(output_dir, f"{output_prefix}{mixed_name}_predicted_noise_normalized.wav")
    
    # get original prediction results
    pred_clean_audio = pred_clean.squeeze(0)
    pred_noise_audio = pred_noise.squeeze(0)
    
    # save loudness normalized audio
    save_audio(pred_clean_audio, clean_output_path)
    save_audio(pred_noise_audio, noise_output_path)
    
    return clean_output_path, noise_output_path

def test_custom_audio(model_path, mixed_audio_path, output_dir):
    """Test custom audio"""    
    print("Starting custom audio testing...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define device
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    # Load model
    model = load_test_model(model_path, device)
    
    # Test audio
    pred_clean_path, pred_noise_path = test_single_audio(
        model, mixed_audio_path, device, output_dir
    )
    
    print(f"\nCustom audio testing completed!")
    print(f"Predicted clean audio: {pred_clean_path}")
    print(f"Predicted noise audio: {pred_noise_path}")

def test_batch_from_excel(model_path, excel_file_path, output_dir, column_name='mixed_audio'):
    """Test batch audio from Excel"""
    print("Starting batch audio testing from Excel...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define device
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    # Load model
    model = load_test_model(model_path, device)
    
    # Read Excel file
    df = pd.read_excel(excel_file_path)
    
    if column_name not in df.columns:
        print(f"Error: Excel file does not contain the column '{column_name}'")
        print(f"Available column names: {list(df.columns)}")
        return None
    
    print(f"Found {len(df)} audio files in total")
    
    results = []
    success_count = 0
    
    for i, row in df.iterrows():
        file_path = row[column_name]
        print(f"Processing file {i+1}/{len(df)}: {file_path}")
        
        # Check if file exists  
        if not os.path.exists(file_path):
            print(f"Warning: File does not exist, skipping: {file_path}")
            results.append({
                'index': i,
                'input_file': file_path,
                'status': 'file_not_found',
                'predicted_clean_audio': '',
                'predicted_noise_audio': ''
            })
            continue
        
        try:
            # Test model
            pred_clean_path, pred_noise_path = test_single_audio(
                model, file_path, device, output_dir, output_prefix=f"sample_{i:04d}_"
            )
            
            # Record success result
            results.append({
                'index': i,
                'input_file': file_path,
                'status': 'success',
                'predicted_clean_audio': pred_clean_path,
                'predicted_noise_audio': pred_noise_path
            })
            success_count += 1
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            results.append({
                'index': i,
                'input_file': file_path,
                'status': f'error: {str(e)}',
                'predicted_clean_audio': '',
                'predicted_noise_audio': ''
            })
            continue
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(output_dir, "batch_processing_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nBatch processing results saved to: {results_csv}")
    
    print(f"\nBatch processing completed!")
    print(f"Success processing {success_count}/{len(df)} files")
    print(f"Results saved in: {output_dir}")
    
    return results_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Cnn14 Audio Separation Model')
    parser.add_argument('--mode', type=str, default='custom', 
                        choices=['custom', 'batch'],
                        help='Test mode: custom or batch')
    parser.add_argument('--model_path', type=str, 
                        default=Config.BEST_MODEL_SAVE_PATH,
                        help='Model file path')
    parser.add_argument('--mixed_audio', type=str, 
                        help='Custom mixed audio path (only used in custom mode)')
    parser.add_argument('--excel_file', type=str, 
                        help='Excel file path containing audio file paths (only used in batch mode)')
    parser.add_argument('--column_name', type=str, default='mixed_audio',
                        help='Column name in Excel file containing file paths (only used in batch mode)')
    parser.add_argument('--output_dir', type=str, 
                        default='test_results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    if args.mode == 'custom':
        if not args.mixed_audio:
            print("Please provide mixed audio path (--mixed_audio)")
            exit(1)
        # Test custom audio
        test_custom_audio(
            model_path=args.model_path,
            mixed_audio_path=args.mixed_audio,
            output_dir=args.output_dir
        )
    elif args.mode == 'batch':
        if not args.excel_file:
            print("Please provide Excel file path (--excel_file)")
            exit(1)
        # Batch processing of audio files in Excel
        test_batch_from_excel(
            model_path=args.model_path,
            excel_file_path=args.excel_file,
            output_dir=args.output_dir,
            column_name=args.column_name
        )