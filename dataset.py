# dataset.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset module for audio separation task
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
from config import Config

class AudioSeparationDataset(Dataset):
    def __init__(self, df, sr=Config.SAMPLE_RATE, duration=Config.DURATION, is_train=True):
        """
        Audio separation dataset
        
        Args:
            df: DataFrame containing mixed_audio, clean_audio, noise_audio columns
            sr: sampling rate
            duration: audio duration (seconds)
            is_train: whether it is a training set (for data augmentation)
        """
        self.df = df
        self.sr = sr
        self.duration = duration
        self.target_length = sr * duration
        self.is_train = is_train
        
        print(f"Loading audio separation dataset: {len(self.df)} samples")
        
    def __len__(self):
        return len(self.df)
    
    def _spectrum_preserve_augment(self, audio):
        """Simple spectrum preservation augmentation"""
        if not self.is_train or np.random.random() > 0.3:  # 30% probability
            return audio
        
        audio_np = audio.numpy()
        sr = self.sr
        
        try:
            # Calculate STFT
            n_fft = 512
            hop_length = 80
            stft = librosa.stft(audio_np, n_fft=n_fft, hop_length=hop_length)
            mag = np.abs(stft)
            
            # Slightly adjust the spectral shape (simulate real changes)
            freq_bins = mag.shape[0]
            
            # Randomly select several frequency bands for fine-tuning
            for _ in range(3):
                start = np.random.randint(0, freq_bins-10)
                end = start + np.random.randint(5, 20)
                gain = np.random.uniform(0.8, 1.2)  # ±20% adjustment
                mag[start:end, :] *= gain
            
            # Keep the total energy constant
            original_energy = np.sum(np.abs(stft)**2)
            adjusted = mag * np.exp(1j * np.angle(stft))
            adjusted_energy = np.sum(np.abs(adjusted)**2)
            
            if adjusted_energy > 0:
                scale = np.sqrt(original_energy / adjusted_energy)
                adjusted *= scale
            
            # Inverse STFT
            enhanced = librosa.istft(adjusted, hop_length=hop_length)
            
            # Keep the length the same
            min_len = min(len(audio_np), len(enhanced))
            audio_np = np.zeros_like(audio_np)
            audio_np[:min_len] = enhanced[:min_len]
            
            return torch.FloatTensor(audio_np)
        except:
            return audio  # If error, return original audio

    def __getitem__(self, idx):
        try:
            # Get three audio file paths
            mixed_path = self.df.iloc[idx]['mixed_audio']
            clean_path = self.df.iloc[idx]['clean_audio']
            noise_path = self.df.iloc[idx]['noise_audio']
            
            # Load three audio files
            mixed_audio = self._load_and_process_audio(mixed_path)
            clean_audio = self._load_and_process_audio(clean_path)
            noise_audio = self._load_and_process_audio(noise_path)
            
            # Apply spectrum preservation augmentation to clean audio
            if self.is_train:
                clean_audio = self._spectrum_preserve_augment(clean_audio)
            
            return mixed_audio, (clean_audio, noise_audio)
            
        except Exception as e:
            print(f"Error loading audio files at index {idx}: {e}")
            # Return empty audio
            empty_audio = torch.zeros(self.target_length)
            return empty_audio, (empty_audio.clone(), empty_audio.clone())
    
    def _load_and_process_audio(self, file_path):
        """Load and preprocess audio file"""
        # Load audio
        audio, _ = librosa.load(file_path, sr=self.sr)
        
        # Process audio length
        if len(audio) > self.target_length:
            if self.is_train:
                # Randomly crop during training
                start = np.random.randint(0, len(audio) - self.target_length)
                audio = audio[start:start + self.target_length]
            else:
                # Take the middle part during validation/test
                start = (len(audio) - self.target_length) // 2
                audio = audio[start:start + self.target_length]
        else:
            # Pad
            audio = np.pad(audio, (0, max(0, self.target_length - len(audio))))
        
        # Convert to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=0)
            
        # Data augmentation (only during training)
        if self.is_train:
            # Random gain
            gain = np.random.uniform(0.8, 1.2)
            audio = audio * gain
            
            # Randomly add noise
            if np.random.random() < 0.3:
                noise = np.random.normal(0, 0.005, len(audio))
                audio = audio + noise
        
        return torch.FloatTensor(audio)

def create_separation_data_loaders(train_df, val_df):
    """Create audio separation data loaders for training and validation"""
    
    # Create datasets
    train_dataset = AudioSeparationDataset(train_df, is_train=True)
    val_dataset = AudioSeparationDataset(val_df, is_train=False)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset, val_dataset