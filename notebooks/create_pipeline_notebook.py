#!/usr/bin/env python3
"""
Script to create 02_preprocessing_modeling.ipynb notebook
"""

import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Flood Prediction - Data Preprocessing & Modeling Pipeline\n",
                "\n",
                "This notebook builds a clean, reusable end-to-end pipeline for:\n",
                "- Data preprocessing and feature engineering\n",
                "- Training baseline models (Linear Regression, Random Forest, XGBoost)\n",
                "- Model evaluation and comparison\n",
                "\n",
                "**Based on EDA findings:**\n",
                "- ✅ No missing values or duplicates\n",
                "- ✅ No data leakage detected\n",
                "- ✅ All 20 features are numeric with low variance (17-20 unique values)\n",
                "- ✅ Target: FloodProbability (continuous, range: 0.285-0.725)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Setup and Data Loading"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import gc\n",
                "from sklearn.model_selection import train_test_split, cross_val_score\n",
                "from sklearn.preprocessing import StandardScaler, RobustScaler\n",
                "from sklearn.linear_model import LinearRegression\n",
                "from sklearn.ensemble import RandomForestRegressor\n",
                "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
                "import xgboost as xgb\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "# Set random seed for reproducibility\n",
                "np.random.seed(42)\n",
                "\n",
                "print(\"Libraries imported successfully!\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load datasets\n",
                "train = pd.read_csv('../data/train.csv')\n",
                "test = pd.read_csv('../data/test.csv')\n",
                "\n",
                "print(f\"Train shape: {train.shape}\")\n",
                "print(f\"Test shape: {test.shape}\")\n",
                "\n",
                "# Define feature columns\n",
                "feature_cols = [col for col in train.columns if col not in ['id', 'FloodProbability']]\n",
                "print(f\"\\nFeatures ({len(feature_cols)}): {feature_cols}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Feature Groups & Pipeline Strategy\n",
                "\n",
                "Based on EDA, all features are:\n",
                "- **Numeric** (integer values 0-19)\n",
                "- **Low variance** (17-20 unique values each)\n",
                "- **Similar distributions** across train/test\n",
                "\n",
                "### Feature Categorization:\n",
                "1. **Environmental factors**: MonsoonIntensity, ClimateChange, Deforestation, WetlandLoss\n",
                "2. **Infrastructure factors**: DamsQuality, DeterioratingInfrastructure, DrainageSystems, Watersheds\n",
                "3. **Geographic factors**: TopographyDrainage, CoastalVulnerability, Landslides\n",
                "4. **Human factors**: Urbanization, PopulationScore, Encroachments, AgriculturalPractices\n",
                "5. **Management factors**: RiverManagement, IneffectiveDisasterPreparedness, InadequatePlanning, PoliticalFactors, Siltation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Define feature groups\n",
                "feature_groups = {\n",
                "    'environmental': ['MonsoonIntensity', 'ClimateChange', 'Deforestation', 'WetlandLoss'],\n",
                "    'infrastructure': ['DamsQuality', 'DeterioratingInfrastructure', 'DrainageSystems', 'Watersheds'],\n",
                "    'geographic': ['TopographyDrainage', 'CoastalVulnerability', 'Landslides'],\n",
                "    'human': ['Urbanization', 'PopulationScore', 'Encroachments', 'AgriculturalPractices'],\n",
                "    'management': ['RiverManagement', 'IneffectiveDisasterPreparedness', 'InadequatePlanning', \n",
                "                   'PoliticalFactors', 'Siltation']\n",
                "}\n",
                "\n",
                "# Verify all features are categorized\n",
                "all_categorized = [f for group in feature_groups.values() for f in group]\n",
                "print(f\"Categorized features: {len(all_categorized)}/{len(feature_cols)}\")\n",
                "print(f\"\\nFeature groups:\")\n",
                "for group, features in feature_groups.items():\n",
                "    print(f\"  {group}: {len(features)} features\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Data Splitting\n",
                "\n",
                "Split training data into train/validation sets for model evaluation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Prepare features and target\n",
                "X = train[feature_cols]\n",
                "y = train['FloodProbability']\n",
                "X_test = test[feature_cols]\n",
                "\n",
                "# Split into train and validation sets (80/20)\n",
                "X_train, X_val, y_train, y_val = train_test_split(\n",
                "    X, y, test_size=0.2, random_state=42\n",
                ")\n",
                "\n",
                "print(f\"Training set: {X_train.shape}\")\n",
                "print(f\"Validation set: {X_val.shape}\")\n",
                "print(f\"Test set: {X_test.shape}\")\n",
                "print(f\"\\nTarget distribution:\")\n",
                "print(f\"  Train - Mean: {y_train.mean():.4f}, Std: {y_train.std():.4f}\")\n",
                "print(f\"  Val   - Mean: {y_val.mean():.4f}, Std: {y_val.std():.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Preprocessing Pipeline\n",
                "\n",
                "Since all features are numeric with similar scales (0-19), we'll use StandardScaler for normalization."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Initialize scaler\n",
                "scaler = StandardScaler()\n",
                "\n",
                "# Fit on training data and transform all sets\n",
                "X_train_scaled = scaler.fit_transform(X_train)\n",
                "X_val_scaled = scaler.transform(X_val)\n",
                "X_test_scaled = scaler.transform(X_test)\n",
                "\n",
                "# Convert back to DataFrames for easier handling\n",
                "X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)\n",
                "X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols, index=X_val.index)\n",
                "X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)\n",
                "\n",
                "print(\"✅ Data scaled successfully!\")\n",
                "print(f\"\\nScaled feature statistics (train):\")\n",
                "print(X_train_scaled.describe().loc[['mean', 'std']].round(4))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4.1 Prepare Unscaled Data for Tree Models\n",
                "\n",
                "Tree-based models don't need scaling. Using unscaled float32 data reduces memory by ~50%."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Creating unscaled float32 data for tree models...\")\n",
                "\n",
                "# Use unscaled data for tree models (scaling not needed for RF/XGB)\n",
                "Xtr_tree = X_train.values.astype(np.float32)\n",
                "Xva_tree = X_val.values.astype(np.float32)\n",
                "ytr_tree = y_train.values.astype(np.float32)\n",
                "yva_tree = y_val.values.astype(np.float32)\n",
                "\n",
                "print(f\"✅ Tree data created (float32):\")\n",
                "print(f\"   Train: {Xtr_tree.shape}, {Xtr_tree.dtype}, {Xtr_tree.nbytes / 1e9:.2f} GB\")\n",
                "print(f\"   Val:   {Xva_tree.shape}, {Xva_tree.dtype}, {Xva_tree.nbytes / 1e9:.2f} GB\")\n",
                "print(f\"\\nMemory savings vs float64: ~50%\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Baseline Models\n",
                "\n",
                "Train three baseline models:\n",
                "1. **Linear Regression** - Simple baseline (uses scaled data)\n",
                "2. **Random Forest** - Non-linear ensemble (uses unscaled float32)\n",
                "3. **XGBoost** - Gradient boosting (uses scaled data)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Helper function for model evaluation\n",
                "def evaluate_model(model, X_train, y_train, X_val, y_val, model_name):\n",
                "    \"\"\"Train model and return evaluation metrics\"\"\"\n",
                "    # Train\n",
                "    model.fit(X_train, y_train)\n",
                "    \n",
                "    # Predictions\n",
                "    y_train_pred = model.predict(X_train)\n",
                "    y_val_pred = model.predict(X_val)\n",
                "    \n",
                "    # Metrics\n",
                "    metrics = {\n",
                "        'model': model_name,\n",
                "        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),\n",
                "        'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),\n",
                "        'train_mae': mean_absolute_error(y_train, y_train_pred),\n",
                "        'val_mae': mean_absolute_error(y_val, y_val_pred),\n",
                "        'train_r2': r2_score(y_train, y_train_pred),\n",
                "        'val_r2': r2_score(y_val, y_val_pred)\n",
                "    }\n",
                "    \n",
                "    return metrics, model\n",
                "\n",
                "print(\"Evaluation function defined!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5.1 Linear Regression"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Training Linear Regression...\")\n",
                "lr_metrics, lr_model = evaluate_model(\n",
                "    LinearRegression(),\n",
                "    X_train_scaled, y_train,\n",
                "    X_val_scaled, y_val,\n",
                "    'Linear Regression'\n",
                ")\n",
                "\n",
                "print(f\"\\n{'='*60}\")\n",
                "print(f\"LINEAR REGRESSION RESULTS\")\n",
                "print(f\"{'='*60}\")\n",
                "print(f\"Train RMSE: {lr_metrics['train_rmse']:.6f}\")\n",
                "print(f\"Val RMSE:   {lr_metrics['val_rmse']:.6f}\")\n",
                "print(f\"Train MAE:  {lr_metrics['train_mae']:.6f}\")\n",
                "print(f\"Val MAE:    {lr_metrics['val_mae']:.6f}\")\n",
                "print(f\"Train R²:   {lr_metrics['train_r2']:.6f}\")\n",
                "print(f\"Val R²:     {lr_metrics['val_r2']:.6f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5.2 Random Forest (Memory-Optimized)\n",
                "\n",
                "Using unscaled float32 data + conservative config to prevent kernel crashes on 1.1M rows."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Training Random Forest (memory-optimized)...\")\n",
                "\n",
                "# Safe RF configuration: limits depth, leaves, trees, and threads\n",
                "rf_safe = RandomForestRegressor(\n",
                "    n_estimators=150,        # start conservative; can raise after it runs\n",
                "    max_depth=14,           # cap depth to control memory\n",
                "    min_samples_leaf=8,     # fewer leaves → smaller model\n",
                "    max_features=\"sqrt\",    # fewer features per split → faster, lighter\n",
                "    bootstrap=True,\n",
                "    n_jobs=4,               # avoid -1 (can oversubscribe cores & RAM)\n",
                "    random_state=42,\n",
                "    verbose=1               # see progress; helps confirm it's not hung\n",
                ")\n",
                "\n",
                "# Train on unscaled float32 data\n",
                "rf_safe.fit(Xtr_tree, ytr_tree)\n",
                "\n",
                "# Predictions\n",
                "pred_tr = rf_safe.predict(Xtr_tree)\n",
                "pred_va = rf_safe.predict(Xva_tree)\n",
                "\n",
                "# Calculate metrics\n",
                "rf_metrics = {\n",
                "    'model': 'Random Forest',\n",
                "    'train_rmse': np.sqrt(mean_squared_error(ytr_tree, pred_tr)),\n",
                "    'val_rmse':   np.sqrt(mean_squared_error(yva_tree, pred_va)),\n",
                "    'train_mae':  mean_absolute_error(ytr_tree, pred_tr),\n",
                "    'val_mae':    mean_absolute_error(yva_tree, pred_va),\n",
                "    'train_r2':   r2_score(ytr_tree, pred_tr),\n",
                "    'val_r2':     r2_score(yva_tree, pred_va),\n",
                "}\n",
                "\n",
                "# Store model for later use\n",
                "rf_model = rf_safe\n",
                "\n",
                "print(f\"\\n{'='*60}\")\n",
                "print(f\"RANDOM FOREST RESULTS (Memory-Optimized)\")\n",
                "print(f\"{'='*60}\")\n",
                "print(f\"Train RMSE: {rf_metrics['train_rmse']:.6f}\")\n",
                "print(f\"Val RMSE:   {rf_metrics['val_rmse']:.6f}\")\n",
                "print(f\"Train MAE:  {rf_metrics['train_mae']:.6f}\")\n",
                "print(f\"Val MAE:    {rf_metrics['val_mae']:.6f}\")\n",
                "print(f\"Train R²:   {rf_metrics['train_r2']:.6f}\")\n",
                "print(f\"Val R²:     {rf_metrics['val_r2']:.6f}\")\n",
                "\n",
                "gc.collect()  # free memory\n",
                "print(\"\\n✅ Memory cleaned up\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5.3 XGBoost"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Training XGBoost...\")\n",
                "xgb_metrics, xgb_model = evaluate_model(\n",
                "    xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),\n",
                "    X_train_scaled, y_train,\n",
                "    X_val_scaled, y_val,\n",
                "    'XGBoost'\n",
                ")\n",
                "\n",
                "print(f\"\\n{'='*60}\")\n",
                "print(f\"XGBOOST RESULTS\")\n",
                "print(f\"{'='*60}\")\n",
                "print(f\"Train RMSE: {xgb_metrics['train_rmse']:.6f}\")\n",
                "print(f\"Val RMSE:   {xgb_metrics['val_rmse']:.6f}\")\n",
                "print(f\"Train MAE:  {xgb_metrics['train_mae']:.6f}\")\n",
                "print(f\"Val MAE:    {xgb_metrics['val_mae']:.6f}\")\n",
                "print(f\"Train R²:   {xgb_metrics['train_r2']:.6f}\")\n",
                "print(f\"Val R²:     {xgb_metrics['val_r2']:.6f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Model Comparison"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create comparison DataFrame\n",
                "results_df = pd.DataFrame([lr_metrics, rf_metrics, xgb_metrics])\n",
                "results_df = results_df.set_index('model')\n",
                "\n",
                "print(\"\\n\" + \"=\"*80)\n",
                "print(\"MODEL COMPARISON\")\n",
                "print(\"=\"*80)\n",
                "print(results_df.to_string())\n",
                "\n",
                "# Find best model\n",
                "best_model_name = results_df['val_rmse'].idxmin()\n",
                "print(f\"\\n🏆 Best Model (by Val RMSE): {best_model_name}\")\n",
                "print(f\"   Val RMSE: {results_df.loc[best_model_name, 'val_rmse']:.6f}\")\n",
                "print(f\"   Val R²: {results_df.loc[best_model_name, 'val_r2']:.6f}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Visualize model comparison\n",
                "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
                "\n",
                "metrics_to_plot = ['rmse', 'mae', 'r2']\n",
                "titles = ['RMSE (lower is better)', 'MAE (lower is better)', 'R² (higher is better)']\n",
                "\n",
                "for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):\n",
                "    train_col = f'train_{metric}'\n",
                "    val_col = f'val_{metric}'\n",
                "    \n",
                "    x = np.arange(len(results_df))\n",
                "    width = 0.35\n",
                "    \n",
                "    axes[idx].bar(x - width/2, results_df[train_col], width, label='Train', alpha=0.8)\n",
                "    axes[idx].bar(x + width/2, results_df[val_col], width, label='Validation', alpha=0.8)\n",
                "    \n",
                "    axes[idx].set_xlabel('Model')\n",
                "    axes[idx].set_ylabel(metric.upper())\n",
                "    axes[idx].set_title(title)\n",
                "    axes[idx].set_xticks(x)\n",
                "    axes[idx].set_xticklabels(results_df.index, rotation=15, ha='right')\n",
                "    axes[idx].legend()\n",
                "    axes[idx].grid(axis='y', alpha=0.3)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.savefig('../reports/figures/model_comparison.png', dpi=300, bbox_inches='tight')\n",
                "plt.show()\n",
                "\n",
                "print(\"✅ Model comparison plot saved to reports/figures/model_comparison.png\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Feature Importance Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Get feature importance from tree-based models\n",
                "rf_importance = pd.DataFrame({\n",
                "    'feature': feature_cols,\n",
                "    'importance': rf_model.feature_importances_\n",
                "}).sort_values('importance', ascending=False)\n",
                "\n",
                "xgb_importance = pd.DataFrame({\n",
                "    'feature': feature_cols,\n",
                "    'importance': xgb_model.feature_importances_\n",
                "}).sort_values('importance', ascending=False)\n",
                "\n",
                "# Plot feature importance\n",
                "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
                "\n",
                "# Random Forest\n",
                "axes[0].barh(rf_importance['feature'][:15], rf_importance['importance'][:15])\n",
                "axes[0].set_xlabel('Importance')\n",
                "axes[0].set_title('Random Forest - Top 15 Features')\n",
                "axes[0].invert_yaxis()\n",
                "\n",
                "# XGBoost\n",
                "axes[1].barh(xgb_importance['feature'][:15], xgb_importance['importance'][:15])\n",
                "axes[1].set_xlabel('Importance')\n",
                "axes[1].set_title('XGBoost - Top 15 Features')\n",
                "axes[1].invert_yaxis()\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.savefig('../reports/figures/feature_importance.png', dpi=300, bbox_inches='tight')\n",
                "plt.show()\n",
                "\n",
                "print(\"✅ Feature importance plot saved to reports/figures/feature_importance.png\")\n",
                "print(\"\\nTop 10 Important Features (Random Forest):\")\n",
                "print(rf_importance.head(10).to_string(index=False))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Generate Test Predictions\n",
                "\n",
                "Use the best performing model to generate predictions for the test set."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Select best model based on validation RMSE\n",
                "models = {\n",
                "    'Linear Regression': lr_model,\n",
                "    'Random Forest': rf_model,\n",
                "    'XGBoost': xgb_model\n",
                "}\n",
                "\n",
                "best_model = models[best_model_name]\n",
                "\n",
                "# Generate predictions (use appropriate data format)\n",
                "if best_model_name == 'Random Forest':\n",
                "    # RF uses unscaled float32\n",
                "    test_predictions = best_model.predict(X_test.values.astype(np.float32))\n",
                "else:\n",
                "    # Linear/XGB use scaled data\n",
                "    test_predictions = best_model.predict(X_test_scaled)\n",
                "\n",
                "# Create submission file\n",
                "submission = pd.DataFrame({\n",
                "    'id': test['id'],\n",
                "    'FloodProbability': test_predictions\n",
                "})\n",
                "\n",
                "submission.to_csv('../data/submission.csv', index=False)\n",
                "\n",
                "print(f\"✅ Test predictions generated using {best_model_name}\")\n",
                "print(f\"\\nPrediction statistics:\")\n",
                "print(f\"  Mean: {test_predictions.mean():.4f}\")\n",
                "print(f\"  Std:  {test_predictions.std():.4f}\")\n",
                "print(f\"  Min:  {test_predictions.min():.4f}\")\n",
                "print(f\"  Max:  {test_predictions.max():.4f}\")\n",
                "print(f\"\\n📁 Submission file saved to: data/submission.csv\")\n",
                "print(f\"   Shape: {submission.shape}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Summary & Next Steps\n",
                "\n",
                "### Key Findings:\n",
                "- All models trained successfully on clean, preprocessed data\n",
                "- Feature importance analysis reveals key flood predictors\n",
                "- Test predictions generated and ready for submission\n",
                "\n",
                "### Next Steps:\n",
                "1. **Hyperparameter Tuning** - Optimize best model parameters\n",
                "2. **Feature Engineering** - Create interaction features or polynomial features\n",
                "3. **Ensemble Methods** - Combine multiple models for better predictions\n",
                "4. **Cross-Validation** - More robust performance estimation\n",
                "5. **Error Analysis** - Analyze prediction errors to identify improvement areas"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": []
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write the notebook
with open('notebooks/02_preprocessing_modeling.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✅ Successfully created 02_preprocessing_modeling.ipynb")
print("\nNotebook includes:")
print("  1. Setup and data loading")
print("  2. Feature groups definition")
print("  3. Data splitting (train/val)")
print("  4. Preprocessing pipeline (StandardScaler)")
print("  5. Baseline models (Linear Regression, Random Forest, XGBoost)")
print("  6. Model comparison and evaluation")
print("  7. Feature importance analysis")
print("  8. Test predictions generation")
print("  9. Summary and next steps")
print("\nReady to run! Open the notebook and execute the cells.")
