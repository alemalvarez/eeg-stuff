import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from typing import List, Dict, Any, Optional

import core.eeg_utils as eeg
from renyi_entropy import calcular_re # Import the RE calculation function

# --- Configuration ---
DATA_FOLDER_PATH = '/Users/alemalvarez/code-workspace/TFG/DATA/BBDDs/HURH'
CLASSICAL_BANDS: Dict[str, List[float]] = {
    "Delta (0.5-4 Hz)": [0.5, 4.0],
    "Theta (4-8 Hz)": [4.0, 8.0],
    "Alpha (8-13 Hz)": [8.0, 13.0],
    "Beta1 (13-19 Hz)": [13.0, 19.0],
    "Beta2 (19-30 Hz)": [19.0, 30.0],
    "Gamma (30-70 Hz)": [30.0, 70.0]
}
SUB_BANDAS_NAMES: List[str] = list(CLASSICAL_BANDS.keys())
Q_PARAM: float = 2.0 # Rényi q parameter (must not be 1.0)

# --- Load Data ---
all_files, names = eeg.load_files_from_folder(DATA_FOLDER_PATH)

if not all_files:
    logger.error(f"No files found in {DATA_FOLDER_PATH}. Exiting.")
    exit()

one_file = all_files[0]
one_name = names[0]
logger.info(f"Processing file: {one_name} for Rényi Entropy (RE) with q={Q_PARAM}")

# --- Preprocessing and PSD Calculation ---
signal, cfg, target = eeg.get_nice_data(raw_data=one_file, name=one_name, comes_from_bbdds=True)
if target is None:
    logger.error(f"File {one_name} has no target. Skipping.")
    exit()
    
n_segments, _, _ = signal.shape
logger.info(f"Signal shape: {signal.shape} (segments, samples, channels)")

f, Pxx = eeg.get_spectral_density(signal, cfg)
logger.success(f"Shape of frequency vector f: {f.shape}")
logger.success(f"Shape of PSD matrix Pxx: {Pxx.shape} (segments, frequencies)")

# --- Determine Overall Band for RE Calculation ---
overall_band_for_re: Optional[List[float]] = None
if 'filtering' in cfg and isinstance(cfg['filtering'], list):
    for filt_item in cfg['filtering']:
        if isinstance(filt_item, dict) and filt_item.get('type') == 'BandPass' and 'band' in filt_item:
            overall_band_for_re = filt_item['band']
            logger.info(f"Using bandpass filter range from cfg for Overall RE: {overall_band_for_re} Hz")
            break
if overall_band_for_re is None and f.size > 0:
    overall_band_for_re = [f.min(), f.max()]
    logger.warning(f"No BandPass filter. Defaulting Overall RE band to: [{f.min():.2f}, {f.max():.2f}] Hz")
elif f.size == 0:
    logger.error("Frequency vector 'f' is empty. Cannot determine overall band for RE.")

# --- RE Calculation --- 
all_re_values_per_segment: List[List[Optional[float]]] = []

Pxx_proc = Pxx.reshape(n_segments, -1) if Pxx.ndim == 2 else Pxx.reshape(1, -1)
n_plot_segments = Pxx_proc.shape[0]

for seg_idx in range(n_plot_segments):
    current_psd_segment = Pxx_proc[seg_idx, :]
    re_values_for_current_segment: List[Optional[float]] = []
    for band_name, band_limits in CLASSICAL_BANDS.items():
        re_val = calcular_re(current_psd_segment, f, band_limits, Q_PARAM)
        re_values_for_current_segment.append(re_val)
        logger.trace(f"Seg {seg_idx+1}, Band '{band_name}': RE(q={Q_PARAM}) = {re_val if re_val is not None else 'None'}")
    
    if overall_band_for_re and f.size > 0:
        overall_re_val = calcular_re(current_psd_segment, f, overall_band_for_re, Q_PARAM)
        re_values_for_current_segment.append(overall_re_val)
        logger.trace(f"Seg {seg_idx+1}, Band 'Overall': RE(q={Q_PARAM}) = {overall_re_val if overall_re_val is not None else 'None'}")
    else:
        re_values_for_current_segment.append(None)
        logger.trace(f"Seg {seg_idx+1}, Band 'Overall': RE(q={Q_PARAM}) = None (band not defined or f empty)")
    all_re_values_per_segment.append(re_values_for_current_segment)

# --- Logging Results ---
LOG_PLOT_BAND_NAMES = SUB_BANDAS_NAMES + [f"Overall RE (q={Q_PARAM})"]
logger.info(f"\n--- Rényi Entropy (RE, q={Q_PARAM}) per Segment ---")
for seg_idx, re_values_segment in enumerate(all_re_values_per_segment):
    logger.info(f"Segment {seg_idx + 1}:")
    for band_name, re_val in zip(LOG_PLOT_BAND_NAMES, re_values_segment):
        logger.info(f"  {band_name}: {re_val:.4f}" if re_val is not None else f"  {band_name}: None")

# --- Visualization ---
try:
    if n_plot_segments > 0 and any(any(v is not None for v in s) for s in all_re_values_per_segment):
        fig_bar, ax_bar = plt.subplots(figsize=(max(10, n_plot_segments * 2.2), 7))
        n_bands_plot = len(LOG_PLOT_BAND_NAMES)
        segment_indices = np.arange(n_plot_segments)
        bar_width = 0.8 / n_bands_plot
        
        cmap = plt.cm.get_cmap('viridis', n_bands_plot)
        colors_list = [cmap(i / (n_bands_plot -1 if n_bands_plot > 1 else 1)) for i in range(n_bands_plot)]

        for band_idx, band_name in enumerate(LOG_PLOT_BAND_NAMES):
            re_for_band = [s[band_idx] if s and len(s) > band_idx and s[band_idx] is not None else np.nan 
                           for s in all_re_values_per_segment]
            re_np = np.array(re_for_band, dtype=float)
            offsets = bar_width * band_idx - (bar_width * (n_bands_plot -1) / 2)
            ax_bar.bar(segment_indices + offsets, re_np, bar_width, label=band_name, color=colors_list[band_idx])

        ax_bar.set_xlabel('Segment Number', fontsize=10)
        ax_bar.set_ylabel(f'Rényi Entropy (q={Q_PARAM})', fontsize=10)
        ax_bar.set_title(f'Rényi Entropy (q={Q_PARAM}) in Bands for {one_name}', fontsize=12)
        ax_bar.set_xticks(segment_indices)
        ax_bar.set_xticklabels([f'{i+1}' for i in segment_indices], fontsize=8)
        ax_bar.legend(title='Frequency Bands', fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
        ax_bar.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout(rect=(0, 0, 0.88, 1))
        plt.show()
    else:
        logger.info("No valid RE data to plot.")
except ImportError:
    logger.warning("Matplotlib not installed. Plotting skipped.")
except Exception as e:
    logger.error(f"Error during visualization: {e}", exc_info=True)

logger.info(f"Rényi Entropy (RE, q={Q_PARAM}) processing finished.") 