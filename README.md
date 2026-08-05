# Deep Learning for Residential Load Forecasting: A Comprehensive Review and Benchmark Study

This repository accompanies the paper:

> **Deep Learning for Residential Load Forecasting: A Comprehensive Review and Benchmark Study**  
> Reza Nematirad, Anil Pahwa, and Balasubramaniam Natarajan

## Overview

This repository provides a unified benchmarking framework for deep learning-based residential load forecasting, including:

- 23 forecasting models
- 4 residential electricity datasets
- Multiple forecasting horizons
- Multiple input sequence lengths
- Accuracy benchmarking (MAE and RMSE)
- Computational efficiency evaluation
  - Training time
  - GPU memory usage
  - Inference latency

---

## Benchmark Models

The evaluated models are grouped into six architectural families.

### Recurrent Neural Networks (RNNs)

- RNN
- LSTM
- GRU
- BiLSTM
- ResLSTM
### CNN / TCN Models

- TimesNet
- ModernTCN
- Times2D
- ConvLSTM
### Transformer Models
- iTransformer
- PatchTST
- Informer
- Crossformer
### MLP / Linear Models
- DLinear
- SparseTSF
- FITS
- TimeMixer
- HDMixer
### State Space Models (SSMs)
- TimePro
- S-Mamba
### Large Language Models (LLMs)
- TimeLLM-GPT2
- TimeLLM-LLaMA7B
- TEMPO

---

# How to Use This Repository

## 1. Clone the Repository

```bash
git clone https://github.com/Tims2D/Residential-Load-Forecasting-Benchmark.git
cd Residential-Load-Forecasting-Benchmark
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Prepare Your Dataset

Place your dataset inside the dataset/ directory.

Dataset Requirements:
- Dataset should be in CSV format.
- The target variable must be the last column.
- All preceding columns are treated as input features.
- Missing values should be handled before training.

Example:

timestamp,temp,humidity,wind_speed,load

Here, 'load' is the forecasting target.

## 4. Run a Model

Example:

```bash
sh ./scripts/lstm/lstm_15Minute_data.sh
```

You can run other models using their corresponding scripts.

# Repository Structure

## dataset/

Contains all datasets used for benchmarking and user-provided datasets.

## data_provider/

Contains the data handling and preprocessing pipeline used throughout the repository.

The module is responsible for:

- Loading datasets from CSV files
- Preparing training, validation, and test sets
- Applying data normalization and scaling
- Generating input-output forecasting windows
- Creating PyTorch dataloaders for deep learning models
- Supporting both deep learning and traditional machine learning forecasting workflows
- Processing temporal features and metadata required by forecasting models

The `data_factory.py` file serves as the interface between experiments and datasets, while `data_loader.py` implements dataset preprocessing, window generation, scaling, and batching utilities. 【1-d0b6cb】【2-44318f】

## exp/

Contains experiment pipelines:
- Training
- Validation
- Testing
- Checkpoint management
- Metrics calculation

## models/

Contains implementations of all forecasting models.

## layers/

Contains reusable neural network components:
- Attention layers
- Embedding layers
- Convolution blocks
- Decomposition modules
- Transformer layers
- State-space components

## utils/

Contains utility functions:
- Evaluation metrics
- Visualization utilities
- Logging
- Early stopping
- Learning-rate scheduling

## scripts/

Contains ready-to-run shell scripts for all benchmark experiments.

Example:

```bash
sh ./scripts/timemixer/timemixer_15Minute_data.sh
```

## Results

Benchmark outputs, figures, and logs can be stored under:

results/
├── accuracy/
├── efficiency/
└── figures/

## Adding a New Model

1. Add the model implementation to models/.
2. Add custom layers to layers/ if needed.
3. Register the model in the experiment pipeline.
4. Create a new script under scripts/.
5. Run the script for training and evaluation.

