#!/usr/bin/env python3
"""
Script to add leakage column removal and data quality confirmation cells
"""

import json

# Read the notebook
with open('notebooks/01_eda.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the index of the last leakage check cell (the summary cell)
insert_index = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'LEAKAGE CHECK SUMMARY' in ''.join(cell.get('source', [])):
        insert_index = i + 1
        break

if insert_index is None:
    print("Could not find leakage summary cell")
    exit(1)

# Create new cells to insert
new_cells = [
    # Code cell: Drop leakage columns
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 7. Drop leakage columns if any exist\n",
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"7. DROPPING LEAKAGE COLUMNS\")\n",
            "print(\"=\"*80)\n",
            "\n",
            "# Collect all columns to drop\n",
            "columns_to_drop = []\n",
            "\n",
            "# Add high correlation features (>0.95)\n",
            "high_corr_features = [f for f, s in sorted_corr if abs(s['correlation']) > 0.95]\n",
            "columns_to_drop.extend(high_corr_features)\n",
            "\n",
            "# Add perfect predictors\n",
            "columns_to_drop.extend(perfect_predictors)\n",
            "\n",
            "# Remove duplicates\n",
            "columns_to_drop = list(set(columns_to_drop))\n",
            "\n",
            "if columns_to_drop:\n",
            "    print(f\"\\n⚠️ Dropping {len(columns_to_drop)} leakage column(s):\")\n",
            "    for col in columns_to_drop:\n",
            "        print(f\"   - {col}\")\n",
            "    \n",
            "    # Drop from train\n",
            "    train_original_shape = train.shape\n",
            "    train = train.drop(columns=columns_to_drop, errors='ignore')\n",
            "    \n",
            "    # Drop from test\n",
            "    test_original_shape = test.shape\n",
            "    test = test.drop(columns=columns_to_drop, errors='ignore')\n",
            "    \n",
            "    # Update feature_cols\n",
            "    feature_cols = [col for col in feature_cols if col not in columns_to_drop]\n",
            "    \n",
            "    print(f\"\\n✅ Columns dropped successfully!\")\n",
            "    print(f\"   Train shape: {train_original_shape} → {train.shape}\")\n",
            "    print(f\"   Test shape: {test_original_shape} → {test.shape}\")\n",
            "    print(f\"   Remaining features: {len(feature_cols)}\")\n",
            "else:\n",
            "    print(\"\\n✅ No leakage columns to drop!\")\n",
            "    print(f\"   Train shape: {train.shape}\")\n",
            "    print(f\"   Test shape: {test.shape}\")\n",
            "    print(f\"   All {len(feature_cols)} features retained\")"
        ]
    },
    # Code cell: Confirm data quality
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 8. Final Data Quality Confirmation\n",
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"8. FINAL DATA QUALITY CONFIRMATION\")\n",
            "print(\"=\"*80)\n",
            "\n",
            "# Check for missing values\n",
            "print(\"\\n📊 Missing Values Check:\")\n",
            "print(\"-\" * 80)\n",
            "train_missing = train.isnull().sum().sum()\n",
            "test_missing = test.isnull().sum().sum()\n",
            "\n",
            "if train_missing == 0 and test_missing == 0:\n",
            "    print(\"✅ No missing values in train or test datasets\")\n",
            "else:\n",
            "    print(f\"⚠️ Missing values found:\")\n",
            "    print(f\"   Train: {train_missing} missing values\")\n",
            "    print(f\"   Test: {test_missing} missing values\")\n",
            "    \n",
            "    if train_missing > 0:\n",
            "        print(\"\\n   Train missing by column:\")\n",
            "        missing_cols = train.isnull().sum()[train.isnull().sum() > 0]\n",
            "        for col, count in missing_cols.items():\n",
            "            print(f\"      {col}: {count} ({count/len(train)*100:.2f}%)\")\n",
            "\n",
            "# Check for duplicate rows\n",
            "print(\"\\n📊 Duplicate Rows Check:\")\n",
            "print(\"-\" * 80)\n",
            "train_dupes = train.duplicated().sum()\n",
            "test_dupes = test.duplicated().sum()\n",
            "\n",
            "if train_dupes == 0 and test_dupes == 0:\n",
            "    print(\"✅ No duplicate rows in train or test datasets\")\n",
            "else:\n",
            "    print(f\"⚠️ Duplicate rows found:\")\n",
            "    print(f\"   Train: {train_dupes} duplicate rows ({train_dupes/len(train)*100:.2f}%)\")\n",
            "    print(f\"   Test: {test_dupes} duplicate rows ({test_dupes/len(test)*100:.2f}%)\")\n",
            "\n",
            "# Final dataset summary\n",
            "print(\"\\n📊 Final Dataset Summary:\")\n",
            "print(\"-\" * 80)\n",
            "print(f\"Train dataset: {train.shape[0]:,} rows × {train.shape[1]} columns\")\n",
            "print(f\"Test dataset:  {test.shape[0]:,} rows × {test.shape[1]} columns\")\n",
            "print(f\"Features:      {len(feature_cols)} features\")\n",
            "print(f\"Target:        FloodProbability (continuous, range: [{train['FloodProbability'].min():.4f}, {train['FloodProbability'].max():.4f}])\")\n",
            "\n",
            "# Data quality score\n",
            "quality_score = 100\n",
            "if train_missing > 0 or test_missing > 0:\n",
            "    quality_score -= 20\n",
            "if train_dupes > 0 or test_dupes > 0:\n",
            "    quality_score -= 20\n",
            "if len(columns_to_drop) > 0:\n",
            "    quality_score -= 10\n",
            "\n",
            "print(f\"\\n📈 Data Quality Score: {quality_score}/100\")\n",
            "\n",
            "if quality_score == 100:\n",
            "    print(\"   ✅ Excellent! Dataset is clean and ready for modeling\")\n",
            "elif quality_score >= 80:\n",
            "    print(\"   ✓ Good! Minor issues addressed, ready for modeling\")\n",
            "elif quality_score >= 60:\n",
            "    print(\"   ⚡ Fair! Some issues remain, proceed with caution\")\n",
            "else:\n",
            "    print(\"   ⚠️ Poor! Significant data quality issues need attention\")\n",
            "\n",
            "print(\"\\n\" + \"=\"*80)"
        ]
    }
]

# Insert new cells
for i, cell in enumerate(new_cells):
    notebook['cells'].insert(insert_index + i, cell)

# Write back to file
with open('notebooks/01_eda.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✅ Successfully added leakage action cells to 01_eda.ipynb")
print(f"   - Added {len(new_cells)} new cells")
print("   - Cell 1: Drop leakage columns from train and test")
print("   - Cell 2: Final data quality confirmation (missing/duplicates)")
print("\nThe notebook now includes:")
print("   ✓ Leakage detection")
print("   ✓ Automatic column removal")
print("   ✓ Missing value confirmation")
print("   ✓ Duplicate row confirmation")
print("   ✓ Final data quality score")
