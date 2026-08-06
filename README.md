# Deep Learning for Residential Load Forecasting: A Comprehensive Review and Benchmark Study
> **Deep Learning for Residential Load Forecasting: A Comprehensive Review and Benchmark Study**  
> Reza Nematirad, Anil Pahwa, and Balasubramaniam Natarajan
>
> 
<p align="center">
  <img src="figures/time_series_forecasting_models last-1.png" widthitory accompanies the paper:



---

<br>

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

### Benchmark Models

The evaluated models are grouped into six architectural families.

#### Recurrent Neural Networks (RNNs)

- RNN
- LSTM
- GRU
- BiLSTM
- ResLSTM

#### CNN / TCN Models

- TimesNet
- ModernTCN
- Times2D
- ConvLSTM

#### Transformer Models

- iTransformer
- PatchTST
- Informer
- Crossformer

#### MLP / Linear Models

- DLinear
- SparseTSF
- FITS
- TimeMixer
- HDMixer

#### State Space Models (SSMs)

- TimePro
- S-Mamba

#### Large Language Models (LLMs)

- TimeLLM-GPT2
- TimeLLM-LLaMA7B
- TEMPO

</details>

---

<details>
<summary><strong>How to Use This Repository</strong></summary>

<br>

### 1. Clone the Repository

```bash
git clone https://github.com/Tims2D/Residential-Load-Forecasting-Benchmark.git
cd Residential-Load-Forecasting-Benchmark
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Your Dataset

Place your dataset inside the `dataset/` directory.

#### Dataset Requirements

- Dataset should be in CSV format.
- The target variable must be the last column.
- All preceding columns are treated as input features.
- Missing values should be handled before training.

Example:

```csv
timestamp,temp,humidity,wind_speed,load
2024-01-01 00:00,22.1,45,5.2,1.83
2024-01-01 00:15,21.8,47,4.8,1.79
```

Here, `load` is the forecasting target.

### 4. Run a Model

Example:

```bash
sh ./scripts/lstm/lstm_15Minute_data.sh
```

You can run other models using their corresponding scripts.

</details>

---

<details>
<summary><strong>Repository Structure</strong></summary>

<br>

### dataset/

Contains all datasets used for benchmarking and user-provided datasets.

### data_provider/

Contains the data handling and preprocessing pipeline used throughout the repository.

Responsibilities include:

- Loading datasets from CSV files
- Preparing training, validation, and test sets
- Applying data normalization and scaling
- Generating forecasting windows
- Creating PyTorch dataloaders
- Supporting both deep learning and machine learning workflows
- Processing temporal features and metadata

### exp/

Contains the experiment and training framework used for all forecasting models.

Responsibilities include:

- Creating and managing forecasting experiments
- Initializing models
- Training, validation, and testing
- Saving checkpoints
- Computing evaluation metrics
- Generating predictions
- Measuring computational efficiency

### models/

Contains implementations of all forecasting models used in the benchmark.

### layers/

Contains reusable neural network building blocks:

- Attention layers
- Embedding layers
- Convolution blocks
- Decomposition modules
- Transformer layers
- State-space components

### utils/

Contains utility functions:

- Evaluation metrics
- Visualization tools
- Logging utilities
- Early stopping
- Learning-rate scheduling

### scripts/

Contains ready-to-run shell scripts for benchmark experiments.

Example:

```bash
sh ./scripts/timemixer/timemixer_15Minute_data.sh
```

### results/

Stores benchmark outputs, figures, and generated results.

```text
results/
├── accuracy/
├── efficiency/
└── figures/
```

### Adding a New Model

1. Add the model implementation to `models/`.
2. Add custom layers to `layers/` if required.
3. Register the model in the experiment framework.
4. Create a script under `scripts/`.
5. Run the script for training and evaluation.

</details>

---

<details>
<summary><strong>Contribution</strong></summary>

<br>

## Contributions

We welcome contributions that improve this repository, including:

- Implementing new forecasting models
- Improving computational efficiency
- Adding new benchmark datasets
- Enhancing documentation
- Reporting bugs and issues
- Improving reproducibility and experimental consistency

Please ensure that all contributions follow the existing repository structure, coding style, and documentation guidelines.

## Contact

For questions, suggestions, or collaboration opportunities, please contact:

[Reza Nemati Rad](mailto:rezanematirad@gmail.com)

## Acknowledgments

We sincerely appreciate the following open-source projects for providing valuable codebases, implementations, and datasets that have contributed to this work:

- [DLinear](https://github.com/vivva/DLinear)
- [Informer](https://github.com/zhouhaoyi/Informer2020)
- [ModernTCN](https://github.com/luodhhh/ModernTCN)
- [PatchTST](https://github.com/yuqinie98/PatchTST)
- [FITS](https://github.com/VEWOXIC/FITS)
- [SparseTSF](https://github.com/lss-1138/SparseTSF)
- [TimeMixer](https://github.com/kwuking/TimeMixer)
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [iTransformer](https://github.com/thuml/iTransformer)

We thank the authors and contributors of these repositories for making their implementations publicly available and supporting reproducible research in time-series forecasting.
