# ALROH: Autism Spectrum Disorder Detection using CNN-LSTM with Dragonfly Optimization

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/license-MIT-green)

ALROH is a hybrid deep learning framework for early detection of Autism Spectrum Disorder (ASD) that combines:
- Convolutional Neural Networks (CNN) for feature extraction
- Long Short-Term Memory (LSTM) networks for sequential learning
- Dragonfly Optimization (DFO) for hyperparameter tuning

## Table of Contents
- [Features](#features)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Reproducing Results](#reproducing-results)

## Features
- Hybrid CNN-LSTM architecture for ASD detection
- Dragonfly Optimization for automatic hyperparameter tuning
- Evaluation on multiple ASD datasets (children and toddlers)
- Comprehensive performance metrics (accuracy, precision, recall, F1-score)
- Comparison with state-of-the-art approaches

## Datasets
The model was evaluated on these publicly available ASD datasets:

| Dataset | Samples | ASD Cases | Non-ASD Cases | Source |
|---------|---------|-----------|---------------|--------|
| Children Dataset | 292 | 141 | 151 | [Thabtah et al.](https://example.com/children_dataset) |
| Toddler Dataset 1 | 1,053 | 728 | 325 | [Thabtah et al.](https://example.com/toddler1_dataset) |
| Toddler Dataset 2 | 506 | 341 | 165 | [Alkahtani et al.](https://example.com/toddler2_dataset) |
| Merged Toddler Dataset | 1,559 | 1,069 | 490 | Combined from above |
| Merged Dataset | 1,851 | 1,210 | 641 | Combined all datasets |

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/alroh-model.git
cd alroh-model
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation
Place your dataset files in the `data/` directory and run:
```bash
python src/data_preprocessing.py --dataset data/children_dataset.csv
```

### Training the Model
Train the ALROH model with default parameters:
```bash
python src/train.py --dataset data/processed/children_dataset.npy
```

### Hyperparameter Optimization
Run Dragonfly Optimization for hyperparameter tuning:
```bash
python src/dfo_optimization.py --dataset data/processed/children_dataset.npy
```

### Evaluation
Evaluate the trained model:
```bash
python src/evaluate.py --model models/alroh_model.h5 --dataset data/processed/children_dataset_test.npy
```

## Project Structure
```
alroh-model/
├── data/                   # Dataset files ( UCI Machine Learning Repository. 2025. Autistic Spectrum Disorder Screening Data for Children. Available at https://archive.ics.uci.edu/dataset/419/autistic+spectrum+disorder+screening+data+for+children  (accessed 12 February 2025).
Autism screening data for toddlers. 2025. Kaggle.com. Available at https://www.kaggle.com/datasets/fabdelja/autism-screening-for-toddlers (accessed 12 February 2025). 
ASD Screening Data for Toddlers in Saudi Arabia. 2022. Kaggle.com. Available at https://www.kaggle.com/datasets/asdpredictioninsaudi/asd-screening-data-for-toddlers-in-saudi-arabia  (accessed 12 February 2025).)

alroh-model/
│
├── data/
│   ├── raw/                # Original dataset files (CSV format)
│   │   ├── children_dataset.csv
│   │   ├── toddler1_dataset.csv
│   │   └── toddler2_dataset.csv
│   │
│   └── processed/          # Processed datasets (numpy format)
│       ├── children_processed.npz
│       ├── toddler1_processed.npz
│       └── toddler2_processed.npz
│
├── models/                 # Saved model weights
│   ├── alroh_children.h5
│   ├── alroh_toddler1.h5
│   └── alroh_toddler2.h5
│
├── notebooks/              # Jupyter notebooks for exploration
│   ├── data_exploration.ipynb
│   └── model_prototyping.ipynb
│
├── results/                # Evaluation results and figures
│   ├── metrics/
│   │   ├── children_metrics.json
│   │   └── comparative_results.csv
│   │
│   └── figures/
│       ├── training_curves.png
│       └── roc_curves.png
│
├── src/
│   ├── data_preprocessing.py  # Data loading and preprocessing
│   ├── model.py            # ALROH model architecture
│   ├── dfo.py              # Dragonfly Optimization implementation
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── utils.py            # Utility functions
│
├── tests/
│   ├── test_data_processing.py
│   ├── test_model.py
│   └── test_dfo.py
│
├── requirements.txt        # Python dependencies
├── LICENSE
└── README.md
```

## Reproducing Results
To reproduce the results from the paper:

1. Download all datasets and place them in `data/raw/`
2. Run the preprocessing pipeline:
```bash
python src/data_preprocessing.py --all
```
3. Train models for all datasets:
```bash
python src/run_experiments.py
```
4. Generate comparison tables:
```bash
python src/generate_results.py
```

