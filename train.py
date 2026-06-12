#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
import time
import librosa
from scipy.stats import spearmanr

from dataset import AudioSeparationDataset, create_separation_data_loaders
from config import Config
from train_process_visual import save_training_metrics, generate_training_charts, analyze_training_progress

# From panns_inference import necessary components
from panns_inference.models import ConvBlock

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

def si_sdr(reference, estimation, eps=1e-8):
    """
    Calculate Scale-Invariant Signal-to-Distortion Ratio
    """
    reference_energy = torch.sum(reference ** 2, dim=-1, keepdim=True)
    correlation = torch.sum(reference * estimation, dim=-1, keepdim=True)
    scale = correlation / (reference_energy + eps)
    reference_scaled = scale * reference
    noise = estimation - reference_scaled
    reference_scaled_energy = torch.sum(reference_scaled ** 2, dim=-1)
    noise_energy = torch.sum(noise ** 2, dim=-1)
    sdr = 10 * torch.log10((reference_scaled_energy + eps) / (noise_energy + eps))
    return torch.mean(sdr)

def snr(reference, estimation, eps=1e-8):
    """
    Calculate Signal-to-Noise Ratio
    """
    noise = estimation - reference
    ref_energy = torch.sum(reference ** 2, dim=-1)
    noise_energy = torch.sum(noise ** 2, dim=-1)
    snr_value = 10 * torch.log10((ref_energy + eps) / (noise_energy + eps))
    return torch.mean(snr_value)

def precompute_features(audio, sr=16000):
    """
    Precompute shared features to avoid redundant calculations
    """
    # Calculate STFT - used for all metrics to ensure consistency
    n_fft = 2048
    hop_length = int(sr * 0.01)  # 10ms hop
    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S_power = np.abs(S)**2
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    
    # Get frequency axis
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    return freqs, S_power, S_db

def calculate_indices_optimized(audio, sr=16000):
    """
    Calculate all four indices simultaneously using precomputed features
    """
    freqs, S_power, S_db = precompute_features(audio, sr)
    
    # ACI: Acoustic Complexity Index
    def calculate_aci_from_features():
        freq_idx = np.where((freqs >= 150) & (freqs <= 8000))[0]
        S_power_limited = S_power[freq_idx, :]
        
        # Calculate differences between adjacent time frames
        if S_power_limited.shape[1] < 2:
            return 0.0
        
        # Calculate the sum of absolute differences between adjacent time points
        diffs = np.sum(np.abs(np.diff(S_power_limited, axis=1)), axis=1)
        sums = np.sum(S_power_limited, axis=1)
        
        # Avoid division by zero
        sums = np.where(sums > 0, sums, 1e-10)
        ratios = diffs / sums
        
        # Total ACI value
        aci_total = np.sum(ratios)
        return aci_total
    
    # ADI: Acoustic Diversity Index (Shannon diversity index, not normalized)
    def calculate_adi_from_features():
        # Define bee-specific frequency bands
        min_freq, max_freq, n_bands = 150, 8000, 6
        band_edges = np.logspace(np.log10(min_freq), np.log10(max_freq), n_bands + 1)
        
        band_energies = []
        for i in range(n_bands):
            start_freq = band_edges[i]
            end_freq = band_edges[i+1]
            
            start_idx = np.searchsorted(freqs, start_freq)
            end_idx = np.searchsorted(freqs, end_freq)
            
            if start_idx < len(freqs) and end_idx <= len(freqs) and start_idx < end_idx:
                # Calculate the mean power spectrum, then sum over the frequency band
                power_band = np.mean(S_power[start_idx:end_idx, :], axis=1)  # Mean spectrum
                band_energy = np.sum(power_band)  # Band total energy
                band_energies.append(band_energy)
            else:
                band_energies.append(0.0)
        
        band_energies = np.array(band_energies)
        total_energy = np.sum(band_energies)
        
        if total_energy == 0:
            return 0.0
        
        proportions = band_energies / total_energy
        epsilon = 1e-10
        # Ensure proportions are not zero to avoid log(0)
        proportions = np.clip(proportions, epsilon, 1.0)
        
        # Calculate Shannon diversity index (not normalized)
        adi_value = -np.sum(proportions * np.log(proportions))
        return adi_value
    
    # AEI: Acoustic Evenness Index 
    def calculate_aei_from_features():
        # Use the same frequency band division and energy calculation as ADI
        min_freq, max_freq, n_bands = 150, 8000, 6
        band_edges = np.logspace(np.log10(min_freq), np.log10(max_freq), n_bands + 1)
        
        band_energies = []
        for i in range(n_bands):
            start_freq = band_edges[i]
            end_freq = band_edges[i+1]
            
            start_idx = np.searchsorted(freqs, start_freq)
            end_idx = np.searchsorted(freqs, end_freq)
            
            if start_idx < len(freqs) and end_idx <= len(freqs) and start_idx < end_idx:
                # Calculate the mean power spectrum, then sum over the frequency band
                power_band = np.mean(S_power[start_idx:end_idx, :], axis=1)  # Mean spectrum
                # Calculate the sum of power in the frequency band
                band_energy = np.sum(power_band)  # Band total energy
                band_energies.append(band_energy)
            else:
                band_energies.append(0.0)
        
        band_energies = np.array(band_energies)
        total_energy = np.sum(band_energies)
        
        if total_energy == 0:
            return 0.0
        
        proportions = band_energies / total_energy
        epsilon = 1e-10
        proportions = np.clip(proportions, epsilon, 1.0)
        
        # Calculate Shannon diversity index 
        h_prime = -np.sum(proportions * np.log(proportions))
        
        # Calculate Pielou's Evenness
        # Use more sensitive active band detection
        # Lower threshold to better detect weak energy
        energy_threshold = np.max(proportions) * 0.01  # Relative threshold, not absolute threshold
        n_active_bands = np.sum(proportions > energy_threshold)  # Use relative threshold
        
        if n_active_bands <= 1:
            return 1.0 if n_active_bands == 1 else 0.0
        
        h_prime_max = np.log(n_active_bands)
        
        # AEI = H' / H'_max
        aei_value = h_prime / h_prime_max if h_prime_max > 0 else 0.0
        
        return aei_value
    
    # BI: Bioacoustic Index
    def calculate_bi_from_features():
        freq_idx = np.where((freqs >= 150) & (freqs <= 8000))[0]
        
        if len(freq_idx) == 0:
            return 0.0
        
        # Average power spectrum in time dimension
        mean_spectrum = np.mean(S_power[freq_idx, :], axis=1)
        
        # Ensure numerical stability
        mean_spectrum = np.clip(mean_spectrum, 1e-10, None)
        
        # Use linear domain for integration, not dB domain
        # Find minimum value as baseline
        s_min = np.min(mean_spectrum)
        
        # Calculate difference integral
        diff = mean_spectrum - s_min
        freqs_limited = freqs[freq_idx]
        
        # Use trapezoidal rule for integration
        bi_value = np.trapz(diff, freqs_limited)
        return bi_value
    
    # Calculate all indices
    aci = calculate_aci_from_features()
    adi = calculate_adi_from_features()
    aei = calculate_aei_from_features()
    bi = calculate_bi_from_features()
    
    return aci, adi, aei, bi

def calculate_spearman_correlation_two_signals(clean_audio, enhanced_audio, sr=16000):
    """
    Calculate Spearman correlation coefficients between clean and enhanced audio for all indices
    Optimized version that computes indices more efficiently
    """
    # Ensure all signals have the same length
    min_length = min(len(clean_audio), len(enhanced_audio))
    clean_audio = clean_audio[:min_length]
    enhanced_audio = enhanced_audio[:min_length]
    
    # To calculate Spearman correlation, we need multiple observations
    segment_length = int(sr * 0.5)  # 0.5-second segment
    hop_length = int(sr * 0.25)     # 0.25-second hop
    
    # Check if there is enough audio length for segmentation analysis
    if min_length < segment_length:
        # If audio is too short, directly calculate overall indices
        clean_aci, clean_adi, clean_aei, clean_bi = calculate_indices_optimized(clean_audio, sr)
        enhanced_aci, enhanced_adi, enhanced_aei, enhanced_bi = calculate_indices_optimized(enhanced_audio, sr)
    
        correlations = {}
        index_names = ['ACI', 'ADI', 'AEI', 'BI']
        clean_values = [clean_aci, clean_adi, clean_aei, clean_bi]
        enhanced_values = [enhanced_aci, enhanced_adi, enhanced_aei, enhanced_bi]
        
        for i, name in enumerate(index_names):
            correlations[f'clean_enhanced_{name}'] = {
                'correlation': np.nan,  
                'p_value': np.nan,
                'difference': abs(clean_values[i] - enhanced_values[i]),
                'warning': 'Audio too short for segmentation analysis'
            }
        return correlations
    
    clean_indices = []
    enhanced_indices = []
    
    for i in range(0, min_length - segment_length + 1, hop_length):
        seg_clean = clean_audio[i:i+segment_length]
        seg_enhanced = enhanced_audio[i:i+segment_length]
        
        clean_aci, clean_adi, clean_aei, clean_bi = calculate_indices_optimized(seg_clean, sr)
        enhanced_aci, enhanced_adi, enhanced_aei, enhanced_bi = calculate_indices_optimized(seg_enhanced, sr)
        
        clean_indices.append([clean_aci, clean_adi, clean_aei, clean_bi])
        enhanced_indices.append([enhanced_aci, enhanced_adi, enhanced_aei, enhanced_bi])
    
    clean_indices = np.array(clean_indices)
    enhanced_indices = np.array(enhanced_indices)

    if len(clean_indices) < 2:
        correlations = {}
        index_names = ['ACI', 'ADI', 'AEI', 'BI']
        for name in index_names:
            correlations[f'clean_enhanced_{name}'] = {
                'correlation': np.nan,
                'p_value': np.nan,
                'error': 'Insufficient data points for correlation'
            }
        return correlations
    
    correlations = {}
    index_names = ['ACI', 'ADI', 'AEI', 'BI']
    
    for i, name in enumerate(index_names):
        try:
            clean_enhanced_corr, clean_enhanced_p = spearmanr(clean_indices[:, i], enhanced_indices[:, i])
            correlations[f'clean_enhanced_{name}'] = {
                'correlation': clean_enhanced_corr,
                'p_value': clean_enhanced_p
            }
        except Exception as e:
            correlations[f'clean_enhanced_{name}'] = {
                'correlation': np.nan,
                'p_value': np.nan,
                'error': str(e)
            }
    
    return correlations

def evaluate_bioacoustic_metrics(model, val_loader, device, num_samples=10):
    """
    Calculate bioacoustic metrics Spearman correlation on validation set audio        
    """
    model.eval()
    all_correlations = []
    
    with torch.no_grad():
        count = 0
        for mixed_batch, (clean_batch, noise_batch) in val_loader:
            if count >= num_samples:
                break
                
            mixed_batch = mixed_batch.to(device)
            clean_batch = clean_batch.to(device)
            
            pred_clean, _ = model(mixed_batch)
            
            for i in range(min(len(clean_batch), len(pred_clean))):
                clean_np = clean_batch[i].cpu().numpy()
                pred_np = pred_clean[i].cpu().numpy()
                
                correlations = calculate_spearman_correlation_two_signals(clean_np, pred_np)
                
                corrs = []
                for metric in ['ACI', 'ADI', 'AEI', 'BI']:
                    corr_key = f'clean_enhanced_{metric}'
                    if corr_key in correlations and 'correlation' in correlations[corr_key]:
                        corr_val = correlations[corr_key]['correlation']
                        if not np.isnan(corr_val):
                            corrs.append(corr_val)
                
                if len(corrs) > 0:
                    all_correlations.append(np.mean(corrs)) 
                    
                count += 1
                if count >= num_samples:
                    break
    
    if len(all_correlations) > 0:
        avg_spearman_mean = np.mean(all_correlations)
        return avg_spearman_mean
    else:
        return float('-inf')

class Cnn14Encoder(nn.Module):
    """Removed spectrogram extractor from Cnn14 encoder for audio separation task"""   
    def __init__(self):
        super(Cnn14Encoder, self).__init__()
        
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        
        self.init_weight()
    
    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
    
    def forward(self, input_spectrogram):
        """
        Input: (batch_size, 1, time_steps, freq_bins) 
        """
        x = input_spectrogram  # (batch, 1, time_steps, freq_bins)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        
        return x

class ComplexMaskDecoder(nn.Module):
    """Complex mask decoder"""    
    def __init__(self):
        super(ComplexMaskDecoder, self).__init__()
        # Reduce parameters for better numerical stability
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=(2,2), stride=(2,2)),
            nn.InstanceNorm2d(1024), 
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=(2,2), stride=(2,2)),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=(2,2), stride=(2,2)),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=(2,2), stride=(2,2)),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(2,2), stride=(2,2)),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1)  # Output complex mask (real and imaginary parts)                        
        )
    
    def forward(self, x):
        x = self.decoder(x)
        return x

class Cnn14Separator(nn.Module):
    """Complete Cnn14 separator model"""    
    def __init__(self):
        super(Cnn14Separator, self).__init__()
        self.encoder = Cnn14Encoder()
        self.decoder = ComplexMaskDecoder()
        
        # STFT parameters 
        self.n_fft = Config.N_FFT
        self.hop_length = Config.HOP_LENGTH
        self.win_length = Config.WIN_LENGTH
        
        # Create window function
        self.register_buffer('window', torch.hann_window(self.win_length))
    
    def forward(self, mixed_audio):
        # Ensure input audio numerical stability  
        mixed_audio = torch.clamp(mixed_audio, -0.99, 0.99)  
        
        if torch.isnan(mixed_audio).any() or torch.isinf(mixed_audio).any():
            print("Input contains NaN or Inf, replacing with zeros")
            mixed_audio = torch.where(torch.isnan(mixed_audio) | torch.isinf(mixed_audio), 
                                      torch.zeros_like(mixed_audio), mixed_audio)
        
        # STFT transformation (using complex form)   
        mixed_stft = torch.stft(
            mixed_audio, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            pad_mode='constant',
            center=True
        )  # (batch, freq_bins, time_frames)
        
        # Extract magnitude spectrum for encoder input
        mixed_mag = torch.abs(mixed_stft)  # (batch, freq_bins, time_frames)
        mixed_mag = mixed_mag.unsqueeze(1)  # (batch, 1, freq_bins, time_frames)
        
        # Special handling: add small constant to avoid zero values
        mixed_mag = mixed_mag + 1e-8
        
        if torch.isnan(mixed_mag).any() or torch.isinf(mixed_mag).any():
            print("Magnitude contains NaN or Inf, clipping")
            mixed_mag = torch.clamp(mixed_mag, min=1e-8, max=1e4)
        
        features = self.encoder(mixed_mag)

        if torch.isnan(features).any() or torch.isinf(features).any():
            print("Features contain NaN or Inf, replacing with zeros")
            features = torch.where(torch.isnan(features) | torch.isinf(features), 
                                   torch.zeros_like(features), features)
        
        complex_mask = self.decoder(features)  # (batch, 2, time_frames_downsampled, freq_bins_downsampled)

        if torch.isnan(complex_mask).any() or torch.isinf(complex_mask).any():
            print("Complex mask contains NaN or Inf, clipping")
            complex_mask = torch.clamp(complex_mask, min=-10.0, max=10.0)
        
        # Adjust mask size to match original STFT dimensions
        target_freq = mixed_stft.shape[1]  # Frequency bins
        target_time = mixed_stft.shape[2]  # Time frames
        
        complex_mask = F.interpolate(
            complex_mask, 
            size=(target_freq, target_time), 
            mode='bilinear', 
            align_corners=False
        )

        complex_mask = torch.tanh(complex_mask)  
        
        complex_mask = complex_mask.permute(0, 2, 3, 1)
        
        mask_complex = torch.view_as_complex(complex_mask.contiguous())  # (batch, freq, time)
        
        if torch.isnan(mask_complex).any() or torch.isinf(mask_complex).any():
            print("Mask complex contains NaN or Inf, clipping")
            mask_complex = torch.clamp(mask_complex, min=-1.0, max=1.0)
        
        clean_stft = mixed_stft * mask_complex
        
        if torch.isnan(clean_stft).any() or torch.isinf(clean_stft).any():
            print("Clean STFT contains NaN or Inf, replacing with zeros")
            clean_stft = torch.where(torch.isnan(clean_stft) | torch.isinf(clean_stft), 
                                     torch.zeros_like(clean_stft), clean_stft)
        
        # ISTFT transformation                         
        pred_clean = torch.istft(
            clean_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=mixed_audio.shape[-1],
            center=True
        )
        
        pred_clean = torch.clamp(pred_clean, -0.99, 0.99)
        
        pred_noise = mixed_audio - pred_clean
        pred_noise = torch.clamp(pred_noise, -0.99, 0.99)
        
        return pred_clean, pred_noise

def load_pretrained_weights(model, general_pretrained_path):
    """Load pretrained weights"""
    
    def load_general_pretrained_weights(encoder_model, pretrained_path):
        """Load general Cnn14 pretrained weights"""
        print(f"Loading general pretrained weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Handle weight dictionary
        if 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        else:
            pretrained_dict = checkpoint
        
        # Create new weight dictionary, adapted to our encoder structure
        model_dict = encoder_model.state_dict()
        adapted_dict = {}
        
        for k, v in pretrained_dict.items():
            # Remove unnecessary components (spectrogram extractor, classification head, etc.)
            if k.startswith('spectrogram_extractor') or k.startswith('logmel_extractor'):
                continue
            if k.startswith('spec_augmenter'):
                continue
            if k.startswith('fc_audioset'):
                continue
                
            # Adapt keys to our encoder
            adapted_dict[k] = v
        
        # Update model weights
        model_dict.update(adapted_dict)
        encoder_model.load_state_dict(model_dict)
        
        print("General pretrained weights loaded")
        return encoder_model
    
    # Load pretrained weights if path is provided and exists
    if general_pretrained_path and os.path.exists(general_pretrained_path):
        load_general_pretrained_weights(model.encoder, general_pretrained_path)
    else:
        print("No pretrained weights found, using random initialization")

def si_snr_loss(s1, s2, eps=1e-8):
    """
    Improved SI-SNR loss function, increased numerical stability    
    """
    s1 = torch.clamp(s1, -0.99, 0.99)
    s2 = torch.clamp(s2, -0.99, 0.99)
    
    if torch.isnan(s1).any() or torch.isinf(s1).any():
        print("s1 contains NaN or Inf in si_snr_loss")
        return torch.tensor(0.0, device=s1.device, requires_grad=True)
    
    if torch.isnan(s2).any() or torch.isinf(s2).any():
        print("s2 contains NaN or Inf in si_snr_loss")
        return torch.tensor(0.0, device=s1.device, requires_grad=True)
    
    s1_s2_norm = torch.sum(s1 * s2, dim=-1, keepdim=True) / (torch.sum(s2**2, dim=-1, keepdim=True) + eps)
    s1_proj = s1_s2_norm * s2
    e_noise = s1 - s1_proj
    
    signal_power = torch.sum(s1_proj**2, dim=-1) + eps
    noise_power = torch.sum(e_noise**2, dim=-1) + eps
    
    signal_power = torch.clamp(signal_power, min=eps)
    noise_power = torch.clamp(noise_power, min=eps)
    
    snr_ratio = signal_power / noise_power
    
    snr_ratio = torch.clamp(snr_ratio, min=1e-6, max=1e6)
    
    si_snr = 10 * torch.log10(snr_ratio)
    
    si_snr = torch.clamp(si_snr, min=-100.0, max=100.0)
    
    return -torch.mean(si_snr)

def enhanced_spectral_loss(pred, target, sample_rate=16000):
    """
    Enhanced spectral loss function  
    """
    n_fft = Config.N_FFT  # 512
    hop_length = Config.HOP_LENGTH  # 80
    
    if torch.isnan(pred).any() or torch.isinf(pred).any():
        print("pred contains NaN or Inf in enhanced_spectral_loss")
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    if torch.isnan(target).any() or torch.isinf(target).any():
        print("target contains NaN or Inf in enhanced_spectral_loss")
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    # STFT
    window = torch.hann_window(n_fft).to(pred.device)
    pred_stft = torch.stft(pred, n_fft=n_fft, hop_length=hop_length,
                          window=window, return_complex=True)
    target_stft = torch.stft(target, n_fft=n_fft, hop_length=hop_length,
                            window=window, return_complex=True)
    
    pred_mag = torch.abs(pred_stft) + 1e-8
    target_mag = torch.abs(target_stft) + 1e-8
    
    log_pred = torch.log(pred_mag)
    log_target = torch.log(target_mag)
    base_loss = F.mse_loss(log_pred, log_target)
    
    freq_bins = n_fft // 2 + 1  # 257
    bee_low = int(Config.BEE_FREQ_BAND[0] * n_fft / sample_rate)  
    bee_high = int(Config.BEE_FREQ_BAND[1] * n_fft / sample_rate)  
    
    bee_loss = 0
    if bee_low < bee_high and bee_high < freq_bins:
        pred_bee = pred_mag[:, bee_low:bee_high, :]
        target_bee = target_mag[:, bee_low:bee_high, :]
        
        bee_log_pred = torch.log(pred_bee)
        bee_log_target = torch.log(target_bee)
        bee_mse_loss = F.mse_loss(bee_log_pred, bee_log_target)

        bee_loss = bee_mse_loss
    else:
        bee_loss = torch.tensor(0.0, device=pred.device)

    return bee_loss

def combined_loss(pred_clean, pred_noise, target_clean, target_noise):
    """
    Fixed combination loss function: only using SI-SNR and Spectral Loss
    """
    if torch.isnan(pred_clean).any() or torch.isinf(pred_clean).any():
        print("Warning: NaN/Inf in pred_clean")
        return torch.tensor(float('inf'), device=pred_clean.device, requires_grad=True)
    
    if torch.isnan(target_clean).any() or torch.isinf(target_clean).any():
        print("Warning: NaN/Inf in target_clean")
        return torch.tensor(float('inf'), device=pred_clean.device, requires_grad=True)
    
    pred_clean = torch.clamp(pred_clean, min=-1.0, max=1.0)
    target_clean = torch.clamp(target_clean, min=-1.0, max=1.0)
    
    losses = {}
    
    # 1. SI-SNR loss
    sisnr_clean = si_snr_loss(pred_clean, target_clean)
    if torch.isnan(sisnr_clean) or torch.isinf(sisnr_clean):
        print(f"sisnr_clean is NaN or Inf: {sisnr_clean}")
        sisnr_clean = torch.tensor(0.0, device=sisnr_clean.device, requires_grad=True)
    losses['sisnr'] = sisnr_clean
    
    # 2. Spectral loss
    spectral_loss = enhanced_spectral_loss(pred_clean, target_clean)
    if torch.isnan(spectral_loss) or torch.isinf(spectral_loss):
        print(f"spectral_loss is NaN or Inf: {spectral_loss}")
        spectral_loss = torch.tensor(0.0, device=spectral_loss.device, requires_grad=True)
    losses['spectral'] = spectral_loss
    
    # Calculate total loss - using fixed weights (SI-SNR=1.0, Spectral=0.1)
    total_weight = 1.0 + 0.1  # 1.1
    
    total_loss = 0.0
    if 'sisnr' in losses:
        total_loss += (1.0 / total_weight) * losses['sisnr']
    if 'spectral' in losses:
        total_loss += (0.1 / total_weight) * losses['spectral']
        
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print(f"Warning: NaN/Inf in total loss:")
        for key, value in losses.items():
            print(f"  {key}: {value}")
        return torch.tensor(float('inf'), device=total_loss.device, requires_grad=True)
    
    total_loss = torch.clamp(total_loss, min=-100.0, max=100.0)
    
    return total_loss

def train_model():
    experiment_name = "SI-SNR + Spectral Loss (Fixed)"
    print(f"Starting training of Cnn14 audio separation model: {experiment_name}")
    print(f"Loss function composition: SI-SNR (weight 1.0) + Spectral (weight 0.1)")
    
    start_time = time.time()  
    
    # Add a list to record training history         
    training_history = []
    
    # Define device
    DEVICE = Config.DEVICE
    
    best_spearman_mean = float('-inf')
    
    # Create model      
    model = Cnn14Separator().to(DEVICE)
    
    # Load pretrained weights    
    load_pretrained_weights(
        model, 
        Config.GENERAL_PRETRAINED_PATH
    )
    
    # Set different learning rates for different parts    
    encoder_params = []
    decoder_params = []
    
    # Separate encoder and decoder parameters        
    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    
    optimizer = torch.optim.Adam([
        {'params': encoder_params, 'lr': Config.ENCODER_LR},  # Encoder uses smaller learning rate
        {'params': decoder_params, 'lr': Config.DECODER_LR}   # Decoder uses slightly larger learning rate
    ], weight_decay=Config.WEIGHT_DECAY)
    
    # Use CosineAnnealingLR to dynamically adjust learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
    
    # Read independent dataset
    print("Reading independent train and validation sets...")
    train_df = pd.read_excel(Config.TRAIN_DATASET_PATH)
    val_df = pd.read_excel(Config.VAL_DATASET_PATH)

    # Check for necessary columns
    required_columns = ['mixed_audio', 'clean_audio', 'noise_audio']
    for col in required_columns:
        if col not in train_df.columns:
            raise ValueError(f"Train Excel file does not contain the column '{col}'. Please check the file content!")  
        if col not in val_df.columns:
            raise ValueError(f"Validation Excel file does not contain the column '{col}'. Please check the file content!")

    # Create independent dataset and data loaders
    train_loader, val_loader, train_dataset, val_dataset = create_separation_data_loaders(train_df, val_df)
    
    epoch_times = []  # Used to record the time of each epoch
    
    for epoch in range(Config.NUM_EPOCHS):
        epoch_start_time = time.time() 
        
        # Training  
        model.train()
        train_loss = 0
        train_sdr_clean = 0
        train_snr_clean = 0
        
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.NUM_EPOCHS} [Train]')
        for batch_idx, (mixed_batch, (clean_batch, noise_batch)) in enumerate(train_bar):
            mixed_batch = mixed_batch.to(DEVICE)
            clean_batch = clean_batch.to(DEVICE)
            noise_batch = noise_batch.to(DEVICE)
            
            if torch.isnan(mixed_batch).any() or torch.isinf(mixed_batch).any():
                print("Skipping batch due to NaN/Inf in mixed_batch")
                continue
            
            if torch.isnan(clean_batch).any() or torch.isinf(clean_batch).any():
                print("Skipping batch due to NaN/Inf in clean_batch")
                continue
                
            optimizer.zero_grad()
            pred_clean, pred_noise = model(mixed_batch)
            
            if torch.isnan(pred_clean).any() or torch.isinf(pred_clean).any():
                print("Skipping batch due to NaN/Inf in model output")
                continue
            
            loss = combined_loss(pred_clean, pred_noise, clean_batch, noise_batch)
            
            if torch.isnan(loss) or torch.isinf(loss) or loss > 1e5:
                print(f"Skipping batch due to invalid loss: {loss}")
                continue
            
            loss.backward()
            train_bar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{train_loss/(batch_idx+1):.4f}'})
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            train_loss += loss.item()
            
            # Only calculate clean audio evaluation metrics
            with torch.no_grad():
                train_sdr_clean += si_sdr(clean_batch, pred_clean).item()
                train_snr_clean += snr(clean_batch, pred_clean).item()
            
            # train_bar.set_postfix(loss=loss.item())
        
        # Validation
        model.eval()
        val_loss = 0
        val_sdr_clean = 0
        val_snr_clean = 0
        
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{Config.NUM_EPOCHS} [Val]')
        with torch.no_grad():
            for mixed_batch, (clean_batch, noise_batch) in val_bar:
                mixed_batch = mixed_batch.to(DEVICE)
                clean_batch = clean_batch.to(DEVICE)
                noise_batch = noise_batch.to(DEVICE)
                
                pred_clean, pred_noise = model(mixed_batch)
                
                # Check validation loss calculation
                loss = combined_loss(pred_clean, pred_noise, clean_batch, noise_batch)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    
                    # Only calculate clean audio evaluation metrics
                    val_sdr_clean += si_sdr(clean_batch, pred_clean).item()
                    val_snr_clean += snr(clean_batch, pred_clean).item()
                else:
                    print(f"Invalid validation loss: {loss}")
        
        spearman_mean = evaluate_bioacoustic_metrics(model, val_loader, DEVICE, num_samples=20)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len([x for x in val_loader if True]) 
        
        avg_train_sdr_clean = train_sdr_clean / len(train_loader)
        avg_train_snr_clean = train_snr_clean / len(train_loader)
        
        avg_val_sdr_clean = val_sdr_clean / len([x for x in val_loader if True])
        avg_val_snr_clean = val_snr_clean / len([x for x in val_loader if True])
        
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_sdr_clean': avg_train_sdr_clean,
            'train_snr_clean': avg_train_snr_clean,
            'val_sdr_clean': avg_val_sdr_clean,
            'val_snr_clean': avg_val_snr_clean,
            'spearman_mean': spearman_mean
        }
        
        training_history.append(epoch_data)
        
        save_training_metrics(training_history)
        generate_training_charts(training_history)
        analyze_training_progress(training_history)
        

        print(f'Epoch {epoch+1}/{Config.NUM_EPOCHS}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print(f'  Train SDR: Clean={avg_train_sdr_clean:.2f}dB')
        print(f'  Train SNR: Clean={avg_train_snr_clean:.2f}dB')
        print(f'  Val SDR: Clean={avg_val_sdr_clean:.2f}dB')
        print(f'  Val SNR: Clean={avg_val_snr_clean:.2f}dB')
        print(f'  Spearman Mean (ACI/AEI/ADI/BI): {spearman_mean:.4f}')
        
        scheduler.step()
        
        if spearman_mean > best_spearman_mean:
            best_spearman_mean = spearman_mean
            torch.save(model.state_dict(), Config.BEST_MODEL_SAVE_PATH)
            print(f" Save best model  {best_spearman_mean:.4f}")  #

        epoch_duration = time.time() - epoch_start_time  # Calculate current epoch duration
        epoch_times.append(epoch_duration)

    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    
    print(f"\nAudio separation model training completed: {experiment_name}")
        
    total_training_time = time.time() - start_time 

    print(f"\nTotal training time: {total_training_time/3600:.2f} hours") 
    print(f"Average epoch duration: {np.mean(epoch_times):.2f} seconds")

if __name__ == '__main__':
    train_model()