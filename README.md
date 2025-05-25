# EEG Data Handling Efficiency Test

This module provides a comprehensive testing framework for evaluating the efficiency of EEG data handling, storage, and machine learning pipelines using PyTorch and Weights & Biases (wandb).

## Overview

The testing pipeline evaluates the efficiency of:

1. **H5 Dataset Creation & Upload**: Creates dummy EEG data and uploads to wandb
2. **Data Loading**: Downloads and loads H5 datasets from wandb
3. **Data Preparation**: Prepares PyTorch datasets from EEG segments and spectral parameters
4. **Model Training**: Trains dummy neural networks on both raw EEG signals and spectral features
5. **Performance Monitoring**: Times all operations and logs metrics to wandb

## Features

### Data Structure
- **Raw EEG Segments**: Simulated multi-channel EEG data (68 channels, 1000 samples per segment)
- **Spectral Parameters**: Feature extracted parameters including:
  - Relative power in frequency bands (delta, theta, alpha, beta1, beta2, gamma)
  - Median frequency, individual alpha frequency, transition frequency
  - Spectral edge frequency (95%)
  - Entropy measures (Rényi, Shannon, Tsallis)
  - Spectral crest factor and centroid

### Machine Learning Models
- **CNN Model**: Simple convolutional neural network for raw EEG classification
- **MLP Model**: Multi-layer perceptron for spectral parameter classification

### Monitoring & Logging
- **Loguru**: Beautiful colored logging with timestamps
- **Weights & Biases**: Experiment tracking and metrics logging
- **Comprehensive Timing**: All major operations are timed and logged

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies
- `torch>=2.0.0` - PyTorch for neural networks
- `numpy>=1.24.0` - Numerical computations
- `h5py>=3.8.0` - HDF5 file format support
- `wandb>=0.15.0` - Experiment tracking
- `loguru>=0.7.0` - Advanced logging

## Usage

### Basic Usage

```python
from test_efficiency import EEGEfficiencyTester

# Initialize the tester
tester = EEGEfficiencyTester(project_name="my-eeg-project")

# Run the complete efficiency test
results = tester.run_full_efficiency_test()

# Results contain timing information for all operations
print(f"Total time: {results['total_time']:.2f} seconds")
```

### Running from Command Line

```bash
python test_efficiency.py
```

### Individual Operations

You can also run individual parts of the pipeline:

```python
tester = EEGEfficiencyTester()

# Create and save dataset
save_time = tester.create_and_save_dataset()

# Load dataset
subjects_data, load_time = tester.download_and_load_dataset()

# Prepare PyTorch datasets
raw_dataloader, spectral_dataloader, prep_time = tester.prepare_datasets(subjects_data)

# Train models
raw_training_time = tester.train_raw_model(raw_dataloader)
spectral_training_time = tester.train_spectral_model(spectral_dataloader)
```

## Data Structure

The H5 file contains the following structure:

```
h5test.h5
├── subjects/
│   ├── subject_001/
│   │   ├── raw_segments          # (47, 1000, 68) - Raw EEG data
│   │   ├── spectral/
│   │   │   ├── psd              # Power spectral density
│   │   │   ├── f                # Frequencies
│   │   │   └── spectral_parameters/
│   │   │       ├── median_frequency
│   │   │       ├── spectral_edge_frequency_95
│   │   │       ├── individual_alpha_frequency
│   │   │       ├── transition_frequency
│   │   │       ├── relative_power_*     # Per frequency band
│   │   │       ├── renyi_entropy
│   │   │       └── ...
│   │   └── attributes           # Metadata (category, sampling_rate, etc.)
│   └── subject_002/
│       └── ...
```

## Model Architectures

### CNN Model (Raw EEG)
- Input: (batch_size, 1000, 68) - Time samples × Channels
- Architecture:
  - Conv1D layers with increasing channels (64 → 128 → 256)
  - ReLU activations and MaxPooling
  - Adaptive average pooling
  - Fully connected classifier (256 → 128 → 2)

### MLP Model (Spectral Features)
- Input: (batch_size, n_features) - Concatenated spectral parameters
- Architecture:
  - Fully connected layers (n_features → 256 → 128 → 64 → 2)
  - ReLU activations and dropout (0.3)

## Metrics Logged to wandb

### Timing Metrics
- `h5_creation_time` - Time to create H5 file
- `h5_upload_time` - Time to upload to wandb
- `h5_download_time` - Time to download from wandb
- `h5_load_time` - Time to load H5 data into memory
- `dataset_prep_time` - Time to prepare PyTorch datasets
- `raw_model_training_time` - CNN training time
- `spectral_model_training_time` - MLP training time
- `total_time` - Total pipeline time

### Data Metrics
- `h5_file_size_mb` - File size in megabytes
- `n_segments_total` - Total number of EEG segments
- `n_spectral_features` - Number of spectral features

### Training Metrics
- `raw_model_loss` - CNN training loss per epoch
- `spectral_model_loss` - MLP training loss per epoch

## Sample Output

```
2024-01-15 10:30:00 | INFO     | test_efficiency:main:465 - 🎯 EEG Data Handling Efficiency Test
2024-01-15 10:30:00 | INFO     | test_efficiency:main:466 - ==================================================
2024-01-15 10:30:00 | INFO     | test_efficiency:__init__:118 - Using device: cuda
2024-01-15 10:30:00 | INFO     | test_efficiency:__init__:121 - Initializing Weights & Biases...
2024-01-15 10:30:01 | INFO     | test_efficiency:run_full_efficiency_test:414 - 🚀 Starting full EEG efficiency test pipeline...
2024-01-15 10:30:01 | INFO     | test_efficiency:create_and_save_dataset:131 - 🔄 Creating H5 dataset...
2024-01-15 10:30:02 | INFO     | test_efficiency:create_and_save_dataset:135 - ✅ H5 dataset created in 0.85 seconds
2024-01-15 10:30:02 | INFO     | test_efficiency:create_and_save_dataset:138 - 📁 H5 file size: 118.45 MB
2024-01-15 10:30:02 | INFO     | test_efficiency:create_and_save_dataset:141 - 📤 Uploading H5 file to wandb...
2024-01-15 10:30:05 | INFO     | test_efficiency:create_and_save_dataset:145 - ✅ File uploaded to wandb in 2.34 seconds
```

## Customization

### Modifying Data Parameters

Edit the constants in `create_dummy_data()` function:

```python
n_samples = 1000   # Samples per segment
n_channels = 68    # EEG channels
n_segments = 47    # Segments per subject
```

### Changing Model Architectures

Modify the `SimpleCNN` or `SpectralMLP` classes:

```python
class SimpleCNN(nn.Module):
    def __init__(self, n_channels: int = 68, n_samples: int = 1000):
        # Modify architecture here
        pass
```

### Adjusting Training Parameters

Change training parameters in the train methods:

```python
n_epochs = 3      # Number of training epochs
batch_size = 32   # Batch size for DataLoader
learning_rate = 0.001  # Optimizer learning rate
```

## Performance Considerations

- **File Size**: H5 files can be large (~100MB for 5 subjects). Consider compression settings.
- **Memory Usage**: Loading all data into memory may require significant RAM for large datasets.
- **GPU Usage**: Models automatically use CUDA if available for faster training.
- **Network Speed**: wandb upload/download times depend on internet connection.

## Troubleshooting

### Common Issues

1. **wandb Authentication**: Ensure you're logged into wandb (`wandb login`)
2. **CUDA Memory**: Reduce batch size if GPU memory is insufficient
3. **File Permissions**: Ensure write permissions for H5 file creation
4. **Package Versions**: Use compatible versions as specified in requirements.txt

### Type Checking

The code includes extensive type hints and handles h5py type checking issues with appropriate `# type: ignore` comments.

## Contributing

To extend the testing framework:

1. Add new spectral parameters in the `Subject` dataclass
2. Implement additional model architectures
3. Add new efficiency metrics to track
4. Extend the wandb logging functionality

## License

MIT License - See original code for details. 