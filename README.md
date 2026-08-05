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
