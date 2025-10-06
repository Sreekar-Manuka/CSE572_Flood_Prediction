# CSE572 Multi-Modal Flood Prediction

A machine learning project for predicting flood probability based on various environmental and geographical features. This project is part of the Kaggle Playground Series S4E5 competition.

## Project Overview

This project implements an end-to-end machine learning pipeline for flood prediction, including:
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Model training and evaluation (Linear Regression, Random Forest, XGBoost)
- Model comparison and selection

# Commands to run the project


## Step 1: Clone the repository

```bash
    git clone git@github.com:Sreekar-Manuka/CSE572_Flood_Prediction.git
    
```

## Step 2: Create a virtual environment

```bash

    python -m venv venv
    source venv/bin/activate

```     

## Step 3: Install dependencies

```bash
    pip install -r requirements.txt
```
## Step 4: Run the notebooks

```bash
    jupyter notebook
```
## Step 5: Run EDA

```bash
    jupyter notebook notebooks/01_eda.ipynb 
```
## Step 6: Run Preprocessing and Modeling

```bash
    jupyter notebook notebooks/02_preprocessing_modeling.ipynb 
```
## Troubleshooting

- If you get an error about missing dependencies, run `pip install -r requirements.txt` again.

- If you get an error about missing jupyter command 
```bash
    source venv/bin/activate  
    pip install jupyter notebook
```


