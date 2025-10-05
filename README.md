# CSE572 Flood Prediction

A machine learning project for predicting flood probability based on various environmental and geographical features. This project is part of the Kaggle Playground Series S4E5 competition.

## Project Overview

This project implements an end-to-end machine learning pipeline for flood prediction, including:
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Model training and evaluation (Linear Regression, Random Forest, XGBoost)
- Model comparison and selection

## Dataset

The dataset contains 20 numeric features related to environmental and geographical conditions, with a target variable `FloodProbability` (continuous, range: 0.285-0.725).

**Key characteristics:**
- No missing values or duplicates
- All features are numeric with low variance (17-20 unique values)
- No data leakage detected

## Project Structure

```
Flood_Prediction/
├── data/
│   ├── train.csv              # Training dataset
│   ├── test.csv               # Test dataset
│   ├── sample_submission.csv  # Sample submission format
│   └── submission.csv         # Generated predictions
├── notebooks/
│   ├── 01_eda.ipynb                      # Exploratory Data Analysis
│   ├── 02_preprocessing_modeling.ipynb   # Preprocessing & Modeling Pipeline
│   ├── create_pipeline_notebook.py       # Notebook generation script
│   └── rf_memory_optimized.py            # Memory-optimized Random Forest
├── reports/
│   └── figures/
│       ├── feature_importance.png        # Feature importance visualization
│       └── model_comparison.png          # Model performance comparison
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone the Repository

```bash
git clone git@github.com-uni:Sreekar-Manuka/CSE572_Flood_Prediction.git
cd CSE572_Flood_Prediction
```

### Step 2: Create a Virtual Environment

Creating a virtual environment ensures that project dependencies are isolated from your system Python installation.

**On Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` appear in your terminal prompt, indicating the virtual environment is active.

### Step 3: Install Dependencies

With the virtual environment activated, install all required packages:

```bash
pip install -r requirements.txt
```

This will install:
- pandas (data manipulation)
- numpy (numerical computing)
- matplotlib (plotting)
- seaborn (statistical visualization)
- scikit-learn (machine learning)
- xgboost (gradient boosting)
- jupyter (notebook interface)
- notebook (Jupyter notebook server)

### Step 4: Launch Jupyter Notebook

Start the Jupyter Notebook server:

```bash
jupyter notebook
```

This will open a browser window with the Jupyter interface. If it doesn't open automatically, copy the URL from the terminal (usually `http://localhost:8888/`).

## Running the Notebooks

### Notebook 01: Exploratory Data Analysis (`01_eda.ipynb`)

**Purpose:** Understand the dataset structure, distributions, and relationships.

**How to run:**
1. In the Jupyter interface, navigate to `notebooks/`
2. Click on `01_eda.ipynb` to open it
3. Run all cells sequentially by clicking **Cell → Run All** or press `Shift + Enter` for each cell

**What it does:**
- Loads and inspects the training data
- Analyzes feature distributions and statistics
- Checks for missing values, duplicates, and data quality issues
- Visualizes correlations between features
- Examines the target variable (FloodProbability) distribution
- Performs data leakage detection

**Expected output:**
- Summary statistics tables
- Distribution plots for all features
- Correlation heatmap
- Data quality report

**Runtime:** ~2-3 minutes

---

### Notebook 02: Preprocessing & Modeling (`02_preprocessing_modeling.ipynb`)

**Purpose:** Build and evaluate machine learning models for flood prediction.

**How to run:**
1. In the Jupyter interface, navigate to `notebooks/`
2. Click on `02_preprocessing_modeling.ipynb` to open it
3. Run all cells sequentially by clicking **Cell → Run All** or press `Shift + Enter` for each cell

**What it does:**
- Loads training and test data
- Performs data preprocessing and feature scaling
- Trains three baseline models:
  - Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor
- Evaluates models using cross-validation
- Compares model performance (RMSE, R² score)
- Generates predictions on test data
- Creates submission file (`data/submission.csv`)
- Visualizes feature importance and model comparison

**Expected output:**
- Model performance metrics (RMSE, R² scores)
- Cross-validation results
- Feature importance plot (`reports/figures/feature_importance.png`)
- Model comparison plot (`reports/figures/model_comparison.png`)
- Submission file ready for Kaggle (`data/submission.csv`)

**Runtime:** ~5-10 minutes (depending on your system)

---

## Model Performance

Based on the analysis, the models achieve the following performance:

| Model | RMSE | R² Score |
|-------|------|----------|
| Linear Regression | ~0.XXX | ~0.XXX |
| Random Forest | ~0.XXX | ~0.XXX |
| XGBoost | ~0.XXX | ~0.XXX |

*(Run the notebooks to see actual results)*

## Submission

After running `02_preprocessing_modeling.ipynb`, you'll find the submission file at:
```
data/submission.csv
```

This file is ready to be uploaded to the Kaggle competition.

## Troubleshooting

### Issue: Jupyter command not found
**Solution:** Make sure your virtual environment is activated and Jupyter is installed:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install jupyter notebook
```

### Issue: Module not found errors
**Solution:** Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: Kernel dies or memory errors
**Solution:** The Random Forest model can be memory-intensive. Try:
- Closing other applications
- Using the memory-optimized script: `notebooks/rf_memory_optimized.py`
- Reducing the number of trees in Random Forest (`n_estimators` parameter)

### Issue: Data files not found
**Solution:** Ensure you're running notebooks from the project root directory and the `data/` folder contains:
- `train.csv`
- `test.csv`
- `sample_submission.csv`

## Deactivating the Virtual Environment

When you're done working on the project, deactivate the virtual environment:

```bash
deactivate
```

## Additional Notes

- **Data Size:** The training dataset (`train.csv`) is ~56 MB
- **Python Version:** Tested with Python 3.8+
- **OS Compatibility:** Works on Linux, macOS, and Windows

## Authors

Sreekar Manuka

## License

This project is for educational purposes as part of CSE572 coursework.

## Acknowledgments

- Kaggle Playground Series S4E5 for the dataset
- CSE572 course staff for guidance
