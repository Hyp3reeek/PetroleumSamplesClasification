# Petroleum Samples Classification and Stability Analysis

Portfolio machine learning project focused on crude oil sample classification and mixture stability prediction.

## Why This Project
- Solves a practical industrial problem: automatic classification of petroleum samples from physicochemical features.
- Compares multiple model families in one workflow (custom algorithms + library-based methods + neural network).
- Extends beyond classification with a regression pipeline and a simple web app for stability prediction.

## What I Built
- End-to-end training and evaluation script for feature-subset experiments across multiple classifiers.
- Custom implementations of:
	- Decision Tree
	- Random Forest
	- KNN
- SVM wrapper using scikit-learn.
- Feed-forward neural network classifier implemented in PyTorch.
- Data processing and visualization utilities (PCA, LDA, t-SNE, UMAP).
- Stability modeling pipeline for mixture data using RandomForestRegressor.
- Streamlit app for interactive stability index prediction.

## Tech Stack
- Python
- Pandas, NumPy
- scikit-learn
- PyTorch
- Matplotlib
- SciPy, UMAP (umap-learn), Joblib
- Streamlit

## Key Files
- main.py: benchmark of classifiers on different feature subsets.
- models/: custom and wrapped model implementations.
- data.py + helper.py: loading, preprocessing helpers, and visualizations.
- clasification.py: classification of data/data2.csv into predicted classes.
- mixture_stability.py: training of stability regressor and model export.
- stability_app.py: interactive prediction interface.

## Project Documentation
Detailed technical report (PL):
[Project Report](./Petroleum_Classification_Project_Report.pdf)

## Quick Start
1. Clone repository

```bash
git clone https://github.com/Hyp3reeek/PetroleumSamplesClasification.git
cd PetroleumSamplesClasification
```

2. Create and activate virtual environment

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Install dependencies

```bash
pip install numpy pandas scikit-learn matplotlib torch scipy umap-learn joblib streamlit sympy
```

4. Run selected pipeline

```bash
python main.py
python clasification.py
python mixture_stability.py
streamlit run stability_app.py
```

## Context
Developed as an academic project at Gdansk University of Technology, with emphasis on applied machine learning for real-world, data-heavy use cases.
