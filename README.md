# Deep Learning for Residential Load Forecasting: A Comprehensive Review and Benchmark Study

> **Deep Learning for Residential Load Forecasting: A Comprehensive Review and Benchmark Study**  
> Reza Nematirad, Anil Pahwa, and Balasubramaniam Natarajan

<p align="center">
  <img src="figures/time_series_forecasting_t-1.png" width=summary><strong>Overview</strong></summary>

This repository provides a unified benchmarking framework for residential load forecasting using state-of-the-art deep learning models, including:

- Conventional forecasting models and Large Language Models (LLMs)
- Four residential electricity datasets
- Multiple forecasting horizons
- Multiple input sequence lengths
- Accuracy benchmarking using MAE and RMSE
- Computational efficiency evaluation

Since LLM-based approaches require additional dependencies and computational resources, they are separated from conventional models so users can install only the packages required for their experiments.

- **Conventional Models (`Conventional_Models/`)**
  - Recurrent Neural Networks (RNNs)
  - CNN / TCN Models
  - Transformer Models
  - MLP / Linear Models
  - State Space Models (SSMs)

- **Large Language Models (`LLMs/`)**
  - TimeLLM-GPT2 (GPT-2-based LLM)
  - TimeLLM-LLaMA7B (LLaMA-7B-based LLM)
  - TEMPO (GPT-2-based LLM with LoRA fine-tuning)

</details>

---

<details>
<summary><strong>How to Use This Repository</strong></summary>

### Prerequisites

All experiments were developed and tested using **Python 3.10**.

### 1. Clone the Repository

For conventional models (non-LLMs):

```bash
git clone https://github.com/Tims2D/Residential-Load-Forecasting-Benchmark.git
cd Residential-Load-Forecasting-Benchmark/Conventional_Models
```

For LLM-based models:

```bash
git clone https://github.com/Tims2D/Residential-Load-Forecasting-Benchmark.git
cd Residential-Load-Forecasting-Benchmark/LLMs
```

### 2. Install Dependencies

For conventional models (non-LLMs):

```bash
pip install -r requirements.txt
```

For LLM-based models:

```bash
pip install -r requirements_llm.txt
```

### 3. Prepare Your Dataset

Place your dataset inside the `dataset/` directory.

#### Dataset Requirements

- Dataset must be in CSV format.
- The target variable must be the last column.
- All preceding columns are treated as input features.

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

```text
Conventional_Models/
├── data_provider/
├── dataset/
├── exp/
├── layers/
├── models/
├── scripts/
└── utils/

LLMs/
├── data_provider/
├── dataset/
├── exp/
├── layers/
├── models/
├── scripts/
└── utils/
```

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

- `logs/` : Training and evaluation log files.
- `results/` : Saved forecasting outputs, including ground-truth and predicted values.
- `test_results/` : Visualization results generated during testing, such as forecast plots and comparison figures.

</details>

---

<details>
<summary><strong>Contribution</strong></summary>

We welcome contributions that improve this repository, including:

- Implementing new forecasting models
- Improving computational efficiency
- Adding new benchmark datasets
- Enhancing documentation
- Reporting bugs and issues
- Improving reproducibility and experimental consistency

Please ensure that all contributions follow the existing repository structure, coding style, and documentation guidelines.

</details>

---

<details>
<summary><strong>Contact</strong></summary>

For questions, suggestions, or collaboration opportunities, please contact:

**Reza Nematirad**  
📧 rezanematirad@gmail.com

</details>

---

<details>
<summary><strong>Acknowledgments</strong></summary>

We sincerely appreciate the following open-source projects for providing valuable codebases, implementations, and datasets that contributed to this work:

- https://github.com/vivva/DLinear
- https://github.com/zhouhaoyi/Informer2020
- https://github.com/luodhhh/ModernTCN
- https://github.com/yuqinie98/PatchTST
- https://github.com/VEWOXIC/FITS
- https://github.com/lss-1138/SparseTSF
- https://github.com/kwuking/TimeMixer
- https://github.com/thuml/Time-Series-Library
- https://github.com/thuml/iTransformer

We thank the authors and contributors of these repositories for making their implementations publicly available and supporting reproducible research in time-series forecasting.

</details>
