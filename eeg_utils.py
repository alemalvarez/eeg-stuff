import os
from typing import Tuple, Optional
from loguru import logger
import numpy as np
import scipy.io as sio  # type: ignore
from scipy import signal  # type: ignore

def load_files_from_folder(folder_path: str) -> Tuple[list[dict], list[str]]:
    """Load all .mat files from the specified folder."""
    contents = []
    names = []
    try:
        for file in os.listdir(folder_path):
            if file.endswith('.mat'):
                contents.append(sio.loadmat(os.path.join(folder_path, file)))
                names.append(file)
    except Exception as e:
        logger.error(f"Error loading mat files from folder: {e}")
        raise e
    return contents, names
    
def get_nice_data(
    raw_data: dict, 
    name: str,
    positives: list[str] = ['AD', 'MCI'],
) -> Tuple[np.ndarray, dict, bool]:
    """Get the nice data from the MATLAB .mat file."""
    data = raw_data['data']
    signal = data['signal'][0, 0]
    cfg = data['cfg']

    def extract_important_params(cfg_data: dict) -> dict:
        """
        Helper function to extract important parameters from the configuration data.
        
        Args:
            cfg_data: The configuration data from the MATLAB .mat file
            
        Returns:
            dict: A dictionary containing the extracted important parameters
        """
        cfg = cfg_data[0,0][0]
        
        params = {
            'fs': int(cfg['fs'][0][0][0]),  # Sampling rate
            
            # Filtering info
            'filtering': [
                {
                    'type': f['type'][0],
                    'band': f['band'][0].tolist(),
                    'order': int(f['order'][0][0])
                }
                for f in cfg['filtering'][0][0]
            ],
            
            # Trial length in seconds
            'trial_length_secs': float(cfg['trial_length_secs'][0][0][0]),
            
            # Head model info
            'head_model': str(cfg['head_model'][0][0]),

            
            # Source orientation
            'source_orientation': str(cfg['source_orientation'][0][0][0][0]),
            
            # Atlas information
            'atlas': str(cfg['ROIs'][0][0]['Atlas'][0][0][0]),
            
            # Number of discarded ICA components
            'N_discarded_ICA': int(cfg['N_discarded_ICA'][0][0][0])
        }
        
        return params
    
    cfg = extract_important_params(cfg)
    cfg['name'] = name

    positive = any(pos in name for pos in positives)

    return signal, cfg, positive

def get_spectral_density(
    signal_data: np.ndarray, 
    cfg: dict, 
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Power Spectral Density (PSD) for each segment using Welch's method,
    averaging across channels within each segment.

    Args:
        signal_data (np.ndarray): EEG signal with shape (n_segments, n_samples, n_channels).
        cfg (dict): Configuration dictionary containing at least 'fs'.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - f: Frequencies (shape: n_freqs)
            - Pxx_segments: Power Spectral Density averaged across channels for each segment 
                          (shape: n_segments, n_freqs)
    """
    fs = cfg['fs']
    n_segments, n_samples, n_channels = signal_data.shape
    
    freqs = None
    Pxx_segments = []

    for s in range(n_segments):
        Pxx_channels_in_segment = []
        for c in range(n_channels):
            # Compute PSD for segment s, channel c
            # Use n_samples as the segment length for welch calculation on each segment
            f, P = signal.welch(signal_data[s, :, c], fs=fs, nperseg=n_samples, scaling='density')
            
            if freqs is None:
                freqs = f  # Store frequencies from the first calculation
            Pxx_channels_in_segment.append(P)
        
        # Average PSDs across channels for the current segment
        if Pxx_channels_in_segment: # Ensure there are channels
            Pxx_segment_mean = np.mean(Pxx_channels_in_segment, axis=0)
            Pxx_segments.append(Pxx_segment_mean)
        # else: handle case with 0 channels if necessary, though shape implies >= 1

    # Stack segment PSDs into a single array
    Pxx_segments_stacked = np.stack(Pxx_segments, axis=0)  # Shape: (n_segments, n_freqs)

    if freqs is None:
        # Handle case where there are no segments (or channels)
        logger.warning("No segments found to compute PSD.")
        return np.array([]), np.array([])

    return freqs, Pxx_segments_stacked

def plot_segment(
    segment: np.ndarray,
    cfg: dict,
) -> None:
    """
    Plot information about an EEG segment including time domain signal, 
    power spectral density, filter information, and configuration details.
    
    Args:
        segment (np.ndarray): EEG segment with shape (n_samples, n_channels)
        cfg (dict): Configuration dictionary containing parameters like 'fs'
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract necessary parameters
    fs = cfg.get('fs', 0)
    n_samples, n_channels = segment.shape
    
    # Create time vector
    time = np.arange(n_samples) / fs if fs > 0 else np.arange(n_samples)
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Time domain signal for each channel
    for ch in range(n_channels):
        axs[0].plot(time, segment[:, ch], label=f'Channel {ch+1}')
    
    axs[0].set_title('EEG Signal in Time Domain')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude (μV)')
    axs[0].grid(True)
    if n_channels <= 10:  # Only show legend if not too many channels
        axs[0].legend()
    
    # Plot 2: Power Spectral Density using get_spectral_density function
    # Reshape segment to match the expected input shape (1, n_samples, n_channels)
    segment_reshaped = segment.reshape(1, n_samples, n_channels)
    f, Pxx = get_spectral_density(segment_reshaped, cfg)
    
    # Since get_spectral_density returns averaged PSD across channels,
    # we can directly plot it (Pxx shape is (1, n_freqs))
    axs[1].semilogy(f, Pxx[0], label='Average across channels')
    
    axs[1].set_title('Power Spectral Density (Averaged Across Channels)')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('PSD (μV²/Hz)')
    axs[1].grid(True)
    
    # Add filter information if available in cfg
    if 'filtering' in cfg:
        for filter_info in cfg['filtering']:
            if 'type' in filter_info and 'band' in filter_info:
                filter_type = filter_info['type']
                band = filter_info['band']
                
                if filter_type == 'highpass':
                    axs[1].axvline(x=band[0], color='r', linestyle='--', 
                                  label=f"Highpass {band[0]} Hz")
                elif filter_type == 'lowpass':
                    axs[1].axvline(x=band[0], color='g', linestyle='--', 
                                  label=f"Lowpass {band[0]} Hz")
                elif filter_type == 'bandpass' and len(band) >= 2:
                    axs[1].axvspan(band[0], band[1], 
                                  alpha=0.2, color='yellow', label=f"Bandpass {band} Hz")
                elif filter_type == 'notch':
                    axs[1].axvline(x=band[0], color='b', linestyle=':', 
                                  label=f"Notch {band[0]} Hz")
    
    axs[1].legend()
    
    # Add text with configuration information
    info_text = f"Segment Shape: {segment.shape}\n"
    info_text += f"Sampling Rate: {fs} Hz\n"
    info_text += f"Duration: {n_samples/fs:.2f} s\n"
    
    # Add other relevant config info
    for key, value in cfg.items():
        if key not in ['fs', 'filtering'] and not isinstance(value, (dict, list, np.ndarray)):
            info_text += f"{key}: {value}\n"
    
    plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))  # Adjust layout to make room for text
    plt.show()
