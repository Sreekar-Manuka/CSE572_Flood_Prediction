"""
Memory-Optimized RandomForest Implementation for Flood Prediction
==================================================================

This script contains drop-in code cells to replace the problematic RF section
in your notebook. Copy each section into separate cells in order.

ISSUE: RF crashes on 1.1M rows due to:
- Using scaled float64 data (doubles memory)
- Too many trees/threads
- Memory spikes from large matrices

SOLUTION: Use unscaled float32 data + conservative RF config
"""

# ============================================================================
# CELL 1: Create Unscaled Float32 Data for Tree Models
# ============================================================================
# Insert this AFTER the scaling cell (after cell that creates X_train_scaled)

print("Creating unscaled float32 data for tree models...")

# Use unscaled data for tree models (scaling not needed for RF)
Xtr_tree = X_train.values.astype(np.float32)
Xva_tree = X_val.values.astype(np.float32)
ytr_tree = y_train.values.astype(np.float32)
yva_tree = y_val.values.astype(np.float32)

print(f"✅ Tree data created (float32):")
print(f"   Train: {Xtr_tree.shape}, {Xtr_tree.dtype}, {Xtr_tree.nbytes / 1e9:.2f} GB")
print(f"   Val:   {Xva_tree.shape}, {Xva_tree.dtype}, {Xva_tree.nbytes / 1e9:.2f} GB")
print(f"\nMemory savings vs float64: ~50%")


# ============================================================================
# CELL 2: Memory-Safe Random Forest Configuration
# ============================================================================
# Replace the existing RF training cell with this

import gc

print("Training Random Forest (memory-optimized)...")

# Safe RF configuration: limits depth, leaves, trees, and threads
rf_safe = RandomForestRegressor(
    n_estimators=150,        # start conservative; can raise after it runs
    max_depth=14,           # cap depth to control memory
    min_samples_leaf=8,     # fewer leaves → smaller model
    max_features="sqrt",    # fewer features per split → faster, lighter
    bootstrap=True,
    n_jobs=4,               # avoid -1 (can oversubscribe cores & RAM)
    random_state=42,
    verbose=1               # see progress; helps confirm it's not hung
)

# Train on unscaled float32 data
rf_safe.fit(Xtr_tree, ytr_tree)

# Predictions
pred_tr = rf_safe.predict(Xtr_tree)
pred_va = rf_safe.predict(Xva_tree)

# Calculate metrics
rf_metrics = {
    'model': 'Random Forest',
    'train_rmse': np.sqrt(mean_squared_error(ytr_tree, pred_tr)),
    'val_rmse':   np.sqrt(mean_squared_error(yva_tree, pred_va)),
    'train_mae':  mean_absolute_error(ytr_tree, pred_tr),
    'val_mae':    mean_absolute_error(yva_tree, pred_va),
    'train_r2':   r2_score(ytr_tree, pred_tr),
    'val_r2':     r2_score(yva_tree, pred_va),
}

# Store model for later use
rf_model = rf_safe

print(f"\n{'='*60}")
print(f"RANDOM FOREST RESULTS (Memory-Optimized)")
print(f"{'='*60}")
print(f"Train RMSE: {rf_metrics['train_rmse']:.6f}")
print(f"Val RMSE:   {rf_metrics['val_rmse']:.6f}")
print(f"Train MAE:  {rf_metrics['train_mae']:.6f}")
print(f"Val MAE:    {rf_metrics['val_mae']:.6f}")
print(f"Train R²:   {rf_metrics['train_r2']:.6f}")
print(f"Val R²:     {rf_metrics['val_r2']:.6f}")

gc.collect()  # free memory
print("\n✅ Memory cleaned up")


# ============================================================================
# CELL 3 (OPTIONAL): Incremental Tree Building (if still crashes)
# ============================================================================
# Use this ONLY if the above still crashes. Replace CELL 2 with this.

import gc

print("Training Random Forest with warm_start (chunked)...")

# Build trees in chunks to avoid memory spikes
rf_chunked = RandomForestRegressor(
    n_estimators=0,         # start empty; we'll add trees
    warm_start=True,        # allows incremental building
    max_depth=14, 
    min_samples_leaf=8, 
    max_features="sqrt",
    bootstrap=True, 
    n_jobs=4, 
    random_state=42,
    verbose=1
)

# Add trees in chunks: 50 + 50 + 50 + 50 = 200 total
for add in [50, 50, 50, 50]:
    rf_chunked.n_estimators += add
    rf_chunked.fit(Xtr_tree, ytr_tree)
    print(f"✅ Built up to: {rf_chunked.n_estimators} trees")
    gc.collect()

# Predictions
pred_tr = rf_chunked.predict(Xtr_tree)
pred_va = rf_chunked.predict(Xva_tree)

# Calculate metrics
rf_metrics = {
    'model': 'Random Forest',
    'train_rmse': np.sqrt(mean_squared_error(ytr_tree, pred_tr)),
    'val_rmse':   np.sqrt(mean_squared_error(yva_tree, pred_va)),
    'train_mae':  mean_absolute_error(ytr_tree, pred_tr),
    'val_mae':    mean_absolute_error(yva_tree, pred_va),
    'train_r2':   r2_score(ytr_tree, pred_tr),
    'val_r2':     r2_score(yva_tree, pred_va),
}

rf_model = rf_chunked

print(f"\n{'='*60}")
print(f"RANDOM FOREST RESULTS (Chunked)")
print(f"{'='*60}")
print(f"Train RMSE: {rf_metrics['train_rmse']:.6f}")
print(f"Val RMSE:   {rf_metrics['val_rmse']:.6f}")
print(f"Train MAE:  {rf_metrics['train_mae']:.6f}")
print(f"Val MAE:    {rf_metrics['val_mae']:.6f}")
print(f"Train R²:   {rf_metrics['train_r2']:.6f}")
print(f"Val R²:     {rf_metrics['val_r2']:.6f}")


# ============================================================================
# CELL 4 (OPTIONAL): Smoke Test on Subset
# ============================================================================
# Use this to test settings on a smaller subset first

print("Running smoke test on 200K sample...")

# Sample 200K rows for quick test
idx = y_train.sample(200_000, random_state=42).index
Xtr_sample = X_train.loc[idx].values.astype(np.float32)
ytr_sample = y_train.loc[idx].values.astype(np.float32)

rf_test = RandomForestRegressor(
    n_estimators=150, max_depth=14, min_samples_leaf=8,
    max_features="sqrt", n_jobs=4, random_state=42, verbose=1
)

rf_test.fit(Xtr_sample, ytr_sample)
print("✅ Smoke test passed! Safe to run on full dataset.")


# ============================================================================
# CELL 5 (ALTERNATIVE): Extra Trees (more memory-efficient)
# ============================================================================
# If RF keeps crashing, use Extra Trees instead (same API, better memory)

from sklearn.ensemble import ExtraTreesRegressor
import gc

print("Training Extra Trees (RF alternative)...")

et_model = ExtraTreesRegressor(
    n_estimators=300,       # can use more trees than RF
    max_depth=14, 
    min_samples_leaf=8,
    max_features="sqrt", 
    n_jobs=4, 
    random_state=42,
    verbose=1
)

et_model.fit(Xtr_tree, ytr_tree)

pred_tr = et_model.predict(Xtr_tree)
pred_va = et_model.predict(Xva_tree)

et_metrics = {
    'model': 'Extra Trees',
    'train_rmse': np.sqrt(mean_squared_error(ytr_tree, pred_tr)),
    'val_rmse':   np.sqrt(mean_squared_error(yva_tree, pred_va)),
    'train_mae':  mean_absolute_error(ytr_tree, pred_tr),
    'val_mae':    mean_absolute_error(yva_tree, pred_va),
    'train_r2':   r2_score(ytr_tree, pred_tr),
    'val_r2':     r2_score(yva_tree, pred_va),
}

print(f"\n{'='*60}")
print(f"EXTRA TREES RESULTS")
print(f"{'='*60}")
print(f"Train RMSE: {et_metrics['train_rmse']:.6f}")
print(f"Val RMSE:   {et_metrics['val_rmse']:.6f}")
print(f"Train MAE:  {et_metrics['train_mae']:.6f}")
print(f"Val MAE:    {et_metrics['val_mae']:.6f}")
print(f"Train R²:   {et_metrics['train_r2']:.6f}")
print(f"Val R²:     {et_metrics['val_r2']:.6f}")

gc.collect()


# ============================================================================
# CELL 6 (ALTERNATIVE): Histogram Gradient Boosting
# ============================================================================
# Very memory-efficient, often matches/beats RF on tabular data

from sklearn.ensemble import HistGradientBoostingRegressor
import gc

print("Training Histogram Gradient Boosting...")

hgb_model = HistGradientBoostingRegressor(
    max_depth=8, 
    learning_rate=0.1, 
    max_iter=300,
    l2_regularization=1.0, 
    random_state=42,
    verbose=1
)

hgb_model.fit(Xtr_tree, ytr_tree)

pred_tr = hgb_model.predict(Xtr_tree)
pred_va = hgb_model.predict(Xva_tree)

hgb_metrics = {
    'model': 'HistGradientBoosting',
    'train_rmse': np.sqrt(mean_squared_error(ytr_tree, pred_tr)),
    'val_rmse':   np.sqrt(mean_squared_error(yva_tree, pred_va)),
    'train_mae':  mean_absolute_error(ytr_tree, pred_tr),
    'val_mae':    mean_absolute_error(yva_tree, pred_va),
    'train_r2':   r2_score(ytr_tree, pred_tr),
    'val_r2':     r2_score(yva_tree, pred_va),
}

print(f"\n{'='*60}")
print(f"HISTOGRAM GRADIENT BOOSTING RESULTS")
print(f"{'='*60}")
print(f"Train RMSE: {hgb_metrics['train_rmse']:.6f}")
print(f"Val RMSE:   {hgb_metrics['val_rmse']:.6f}")
print(f"Train MAE:  {hgb_metrics['train_mae']:.6f}")
print(f"Val MAE:    {hgb_metrics['val_mae']:.6f}")
print(f"Train R²:   {hgb_metrics['train_r2']:.6f}")
print(f"Val R²:     {hgb_metrics['val_r2']:.6f}")

gc.collect()


# ============================================================================
# GRADUAL CAPACITY INCREASE (after basic version works)
# ============================================================================
"""
Once CELL 2 runs successfully, you can gradually increase capacity:

Step 1: n_estimators: 150 → 250
rf_safe = RandomForestRegressor(
    n_estimators=250,  # increased
    max_depth=14, min_samples_leaf=8, max_features="sqrt",
    bootstrap=True, n_jobs=4, random_state=42, verbose=1
)

Step 2: n_estimators: 250 → 400
rf_safe = RandomForestRegressor(
    n_estimators=400,  # increased
    max_depth=14, min_samples_leaf=8, max_features="sqrt",
    bootstrap=True, n_jobs=4, random_state=42, verbose=1
)

Step 3: Relax min_samples_leaf: 8 → 5
rf_safe = RandomForestRegressor(
    n_estimators=400, max_depth=14, 
    min_samples_leaf=5,  # relaxed
    max_features="sqrt", bootstrap=True, n_jobs=4, random_state=42, verbose=1
)

⚠️ Only change ONE parameter at a time and keep n_jobs=4
"""


# ============================================================================
# IMPORT ADDITIONS (add to top of notebook if needed)
# ============================================================================
"""
Add these imports to your first cell if not already present:

import gc
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
"""
