# train_process_visual.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def save_training_metrics(training_data, filename='training_metrics.xlsx'):
    """save training metrics to Excel file"""    
    df = pd.DataFrame(training_data)
    df.to_excel(filename, index=False)
    print(f"Training metrics saved to {filename}")

def generate_training_charts(training_data, filename='training_trends.png'):
    """generate training process charts"""
    if len(training_data) == 0:
        print("No training data available for chart generation")
        return
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(training_data)
    epochs = df['epoch'].values
    
    # Create charts
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Cnn14 decomposition metrics', fontsize=16)
    
    # Loss curve
    axes[0, 0].plot(epochs, df['train_loss'].values, 'o-', label='train loss', color='blue')
    axes[0, 0].plot(epochs, df['val_loss'].values, 'o-', label='val loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # SDR metric
    axes[0, 1].plot(epochs, df['train_sdr_clean'].values, 'o-', label='train Clean SDR', color='blue')
    axes[0, 1].plot(epochs, df['val_sdr_clean'].values, 'o-', label='val Clean SDR', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('SDR (dB)')
    axes[0, 1].set_title('Clean audio SDR')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # SNR metric
    axes[1, 0].plot(epochs, df['train_snr_clean'].values, 'o-', label='train Clean SNR', color='blue')
    axes[1, 0].plot(epochs, df['val_snr_clean'].values, 'o-', label='val Clean SNR', color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('SNR (dB)')
    axes[1, 0].set_title('Clean audio SNR')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Composite metric
    axes[1, 1].plot(epochs, df['train_loss'].values, 'o-', label='train loss', color='blue')
    axes[1, 1].plot(epochs, df['val_loss'].values, 'o-', label='val loss', color='red')
    axes[1, 1].plot(epochs, df['train_sdr_clean'].values/10, 's--', label='train Clean SDR/10', color='cyan')
    axes[1, 1].plot(epochs, df['val_sdr_clean'].values/10, 's--', label='val Clean SDR/10', color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('value')
    axes[1, 1].set_title('composit metrics')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free up memory
    
    print(f"Training trends chart saved to {filename}")

def analyze_training_progress(training_data):
    """analyze training progress"""
    if len(training_data) < 2:
        return
    
    latest = training_data[-1]
    previous = training_data[-2]
    
    print("\nTraining analysis:")
    if latest['train_loss'] < previous['train_loss']:
        print("1. The model is learning, and the training loss is decreasing")
    else:
        print("1. Training loss is not decreasing, consider adjusting the learning rate or other parameters")
        
    if latest['train_sdr_clean'] > previous['train_sdr_clean']:
        print("2. Clean audio SDR is improving, indicating better separation quality")
    else:
        print("2. Clean audio SDR is not improving, attention needed")
        
    if latest['val_loss'] < previous['val_loss']:
        print("3. Validation loss is decreasing, indicating improvement in model generalization")
    else:
        print("3. Validation loss is not decreasing, attention needed for overfitting")


if __name__ == "__main__":
    # Record training data
    training_data = [
        {
            'epoch': 1,
            'train_loss': 0.9437,
            'val_loss': -0.1971,
            'train_sdr_clean': 7.47,
            'train_sdr_noise': -10.17,
            'train_snr_clean': -1.29,
            'train_snr_noise': -5.29,
            'val_sdr_clean': 11.03,
            'val_sdr_noise': -10.46,
            'val_snr_clean': -1.48,
            'val_snr_noise': -5.27
        },
        {
            'epoch': 2,
            'train_loss': 0.0109,
            'val_loss': -0.4653,
            'train_sdr_clean': 9.92,
            'train_sdr_noise': -9.95,
            'train_snr_clean': -1.91,
            'train_snr_noise': -5.49,
            'val_sdr_clean': 11.48,
            'val_sdr_noise': -10.14,
            'val_snr_clean': -2.37,
            'val_snr_noise': -5.76
        }
    ]
    
    save_training_metrics(training_data)
    generate_training_charts(training_data)
    analyze_training_progress(training_data)