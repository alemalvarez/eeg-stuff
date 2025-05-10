import eeg_utils as eeg
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from typing import List
from sklearn.cluster import KMeans # type: ignore

from CalculoMF import calcular_mf
from CalculoSEF95 import calcular_sef95

# Load all files
logger.info("Loading EEG files...")
all_files, names = eeg.load_files_from_folder('/Users/alemalvarez/code-workspace/TFG/DATA')
logger.success(f"Loaded {len(all_files)} files.")

# Lists to store results
mean_mfs: List[float] = []
mean_sef95s: List[float] = []
targets: List[int] = []
subject_names: List[str] = []

# Process each file
for file_idx, (file, name) in enumerate(zip(all_files, names)):
    logger.info(f"Processing file {file_idx + 1}/{len(all_files)}: {name}")
    try:
        # Get signal and configuration
        signal, cfg, target = eeg.get_nice_data(raw_data=file, name=name)
        
        # Get spectral density
        f, Pxx = eeg.get_spectral_density(signal, cfg)
        
        if Pxx.ndim == 1: # If Pxx is 1D, make it 2D with one segment
            Pxx = Pxx[np.newaxis, :]
            logger.debug(f"Pxx for {name} was 1D, reshaped to {Pxx.shape}")
        elif Pxx.ndim != 2 or Pxx.shape[0] == 0:
            logger.warning(f"Pxx for {name} has unexpected shape {Pxx.shape} or is empty, skipping.")
            continue
            
        # Get band of interest from cfg
        banda_interes = None
        if 'filtering' in cfg and isinstance(cfg['filtering'], list):
            for filt in cfg['filtering']:
                if isinstance(filt, dict) and filt.get('type') == 'BandPass' and 'band' in filt:
                    banda_interes = filt['band']
                    break
        
        if banda_interes is None:
            logger.warning(f"No BandPass filter found in cfg for {name}, skipping...")
            continue
        logger.debug(f"Band of interest for {name}: {banda_interes}")

        # Calculate MF and SEF95 for each segment
        segment_mfs: List[float] = []
        segment_sef95s: List[float] = []
        
        for seg_idx in range(Pxx.shape[0]):
            psd_segment = Pxx[seg_idx, :]
            
            mf_segment = calcular_mf(psd_segment, f, banda_interes)
            sef95_segment = calcular_sef95(psd_segment, f, banda_interes)
            
            if mf_segment is not None:
                segment_mfs.append(mf_segment)
            if sef95_segment is not None:
                segment_sef95s.append(sef95_segment)
        
        # Ensure we have valid data for both MF and SEF95 to form pairs
        if len(segment_mfs) > 0 and len(segment_sef95s) > 0:
            # If differing numbers of valid segments, take the minimum length or handle appropriately
            # For simplicity, let's assume we want to average if at least one of each is found.
            # A more robust approach might pair them if calculated per segment, 
            # but here we average them separately then pair the averages.
            current_mean_mf = float(np.mean(segment_mfs))
            current_mean_sef95 = float(np.mean(segment_sef95s))
            
            mean_mfs.append(current_mean_mf)
            mean_sef95s.append(current_mean_sef95)
            targets.append(target) # Keep original target
            subject_names.append(name)
            logger.success(f"Subject {name}: Mean MF = {current_mean_mf:.2f} Hz, Mean SEF95 = {current_mean_sef95:.2f} Hz, Target = {target}")
        else:
            logger.warning(f"Could not calculate MF and/or SEF95 for enough segments in {name} to form a pair, skipping...")

    except Exception as e:
        logger.error(f"Error processing file {name}: {e}", exc_info=True)
        continue

# Convert lists to numpy arrays for easier indexing
mean_mfs_array = np.array(mean_mfs, dtype=float)
mean_sef95s_array = np.array(mean_sef95s, dtype=float)
targets_array = np.array(targets, dtype=int) # Original targets

if len(mean_mfs_array) == 0 or len(mean_sef95s_array) == 0 or len(mean_mfs_array) != len(mean_sef95s_array):
    logger.error("Not enough data or mismatched data lengths for MF and SEF95 to plot. Exiting.")
    exit()

# Prepare data for clustering
X_for_clustering = np.column_stack((mean_mfs_array, mean_sef95s_array))

if X_for_clustering.shape[0] < 2: # Need at least 2 samples for k=2
    logger.error("Not enough samples for clustering. Exiting.")
    exit()

# Apply KMeans clustering
num_clusters = 2 # Starting with 2 clusters
logger.info(f"Applying KMeans clustering with k={num_clusters}...")
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
kmeans.fit(X_for_clustering)
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

logger.success(f"Clustering complete. Found {num_clusters} clusters.")

# Create the 2D scatter plot, colored by cluster
logger.info("Generating 2D scatter plot colored by cluster...")
plt.figure(figsize=(12, 8))

colors = ['purple', 'orange', 'green', 'brown', 'pink'] # Add more if num_clusters > 5

for i in range(num_clusters):
    cluster_mask = cluster_labels == i
    plt.scatter(mean_mfs_array[cluster_mask], 
                mean_sef95s_array[cluster_mask], 
                color=colors[i % len(colors)], 
                label=f'Cluster {i}', 
                alpha=0.7, edgecolors='w')
    # Plot centroids
    plt.scatter(centroids[i, 0], centroids[i, 1], 
                color=colors[i % len(colors)], marker='X', s=200, 
                edgecolors='black', label=f'Cluster {i} Centroid')
    logger.info(f"Cluster {i} Centroid: MF={centroids[i, 0]:.2f}, SEF95={centroids[i, 1]:.2f}")

plt.title(f'Mean MF vs. Mean SEF95 by KMeans Cluster (k={num_clusters})', fontsize=16)
plt.xlabel('Mean Median Frequency (MF) (Hz)', fontsize=12)
plt.ylabel('Mean Spectral Edge Frequency 95% (SEF95) (Hz)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.5)

# Add cluster statistics to the plot
stats_text = f"Total Subjects Clustered: {len(mean_mfs_array)}\n"
for i in range(num_clusters):
    stats_text += f"Cluster {i} Size: {np.sum(cluster_labels == i)}\n"

# Remove last newline if it exists
stats_text = stats_text.strip()

plt.text(0.02, 0.98, stats_text,
         transform=plt.gca().transAxes,
         fontsize=10,
         verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

plt.tight_layout()
logger.info("Displaying plot...")
plt.show()

logger.success("Script finished.") 