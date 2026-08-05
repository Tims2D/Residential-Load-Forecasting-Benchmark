# Deep Learning for Residential Load Forecasting: A Comprehensive Review and Benchmark Study

This repository accompanies the paper:

> **Deep Learning for Residential Load Forecasting: A Comprehensive Review and Benchmark Study**  
> Reza Nematirad, Anil Pahwa, and Balasubramaniam Natarajan

## Overview

Accurate residential electricity demand forecasting is essential for power system operation, demand response, energy management, and infrastructure planning. Recent advances in deep learning have introduced a wide range of forecasting architectures, yet comprehensive comparisons across datasets, forecasting horizons, and computational requirements remain limited.

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

## Datasets

The benchmark includes four residential electricity consumption datasets spanning a wide range of temporal resolutions.

| Dataset | Resolution |
|----------|------------|
| Load-1 | 1 Hour |
| Load-2 | 15 Minutes |
| Load-3 | 1 Minute |
| Load-4 | 10 Seconds |

The datasets collectively enable evaluation across low-frequency and high-frequency residential forecasting scenarios.

---

## Forecasting Setup

### Input Sequence Lengths

```text
L ∈ {96, 720}
