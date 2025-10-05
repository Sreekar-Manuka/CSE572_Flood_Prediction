#!/usr/bin/env python3
"""
Script to add data leakage check cells to 01_eda.ipynb
"""

import json

# Read the notebook
with open('notebooks/01_eda.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the index where we want to insert (after section 8, before section 9)
insert_index = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown' and '## 9. Key Insights & Next Steps' in ''.join(cell['source']):
        insert_index = i
        break

if insert_index is None:
    print("Could not find insertion point")
    exit(1)

# Create new cells to insert
new_cells = [
    # Markdown cell for section 9
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Data Leakage Checks\n",
            "\n",
            "Checking for potential data leakage issues that could inflate model performance:\n",
            "- Suspiciously high correlations with target\n",
            "- Perfect predictors\n",
            "- Train/test distribution differences\n",
            "- Constant or near-constant features\n",
            "- Features containing future information"
        ]
    },
    # Code cell 1: Correlation analysis
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Check for suspiciously high correlations with target\n",
            "print(\"=\"*80)\n",
            "print(\"1. CORRELATION ANALYSIS - Checking for suspiciously high correlations\")\n",
            "print(\"=\"*80)\n",
            "\n",
            "from scipy.stats import pearsonr\n",
            "\n",
            "correlations = {}\n",
            "for col in feature_cols:\n",
            "    corr, p_value = pearsonr(train[col], train['FloodProbability'])\n",
            "    correlations[col] = {'correlation': corr, 'p_value': p_value}\n",
            "\n",
            "# Sort by absolute correlation\n",
            "sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)\n",
            "\n",
            "print(f\"\\n{'Feature':<40} {'Correlation':>12} {'P-value':>12} {'Status':>15}\")\n",
            "print(\"-\" * 80)\n",
            "\n",
            "for feature, stats in sorted_corr:\n",
            "    corr = stats['correlation']\n",
            "    p_val = stats['p_value']\n",
            "    \n",
            "    if abs(corr) > 0.95:\n",
            "        status = \"⚠️ SUSPICIOUS\"\n",
            "    elif abs(corr) > 0.8:\n",
            "        status = \"⚡ Very High\"\n",
            "    elif abs(corr) > 0.5:\n",
            "        status = \"✓ High\"\n",
            "    elif abs(corr) > 0.3:\n",
            "        status = \"✓ Moderate\"\n",
            "    else:\n",
            "        status = \"✓ Low\"\n",
            "    \n",
            "    print(f\"{feature:<40} {corr:>12.4f} {p_val:>12.2e} {status:>15}\")"
        ]
    },
    # Code cell 2: Perfect predictors
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Check for perfect predictors (potential leakage)\n",
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"2. PERFECT PREDICTOR CHECK\")\n",
            "print(\"=\"*80)\n",
            "\n",
            "perfect_predictors = []\n",
            "for col in feature_cols:\n",
            "    grouped = train.groupby(col)['FloodProbability'].agg(['mean', 'std', 'count'])\n",
            "    \n",
            "    if (grouped['std'].fillna(0) == 0).all():\n",
            "        perfect_predictors.append(col)\n",
            "        print(f\"\\n⚠️ {col}: Perfect predictor detected!\")\n",
            "        print(f\"   Each unique value maps to exactly one target value\")\n",
            "\n",
            "if not perfect_predictors:\n",
            "    print(\"\\n✓ No perfect predictors found\")"
        ]
    },
    # Code cell 3: Distribution check
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Check train/test distribution similarity\n",
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"3. TRAIN/TEST DISTRIBUTION COMPARISON\")\n",
            "print(\"=\"*80)\n",
            "\n",
            "from scipy.stats import ks_2samp\n",
            "\n",
            "print(f\"\\n{'Feature':<40} {'KS Statistic':>15} {'P-value':>12} {'Status':>15}\")\n",
            "print(\"-\" * 80)\n",
            "\n",
            "distribution_issues = []\n",
            "for col in feature_cols:\n",
            "    ks_stat, p_value = ks_2samp(train[col], test[col])\n",
            "    \n",
            "    if p_value < 0.01:\n",
            "        status = \"⚠️ Different\"\n",
            "        distribution_issues.append(col)\n",
            "    elif p_value < 0.05:\n",
            "        status = \"⚡ Slightly Diff\"\n",
            "    else:\n",
            "        status = \"✓ Similar\"\n",
            "    \n",
            "    print(f\"{col:<40} {ks_stat:>15.4f} {p_value:>12.4f} {status:>15}\")\n",
            "\n",
            "if distribution_issues:\n",
            "    print(f\"\\n⚠️ {len(distribution_issues)} feature(s) with significantly different distributions\")"
        ]
    },
    # Code cell 4: Constant features
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 4. Check for constant or near-constant features\n",
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"4. CONSTANT/NEAR-CONSTANT FEATURE CHECK\")\n",
            "print(\"=\"*80)\n",
            "\n",
            "low_variance_features = []\n",
            "for col in feature_cols:\n",
            "    unique_ratio = train[col].nunique() / len(train)\n",
            "    \n",
            "    if unique_ratio < 0.01:\n",
            "        low_variance_features.append(col)\n",
            "        print(f\"\\n⚠️ {col}: Only {train[col].nunique()} unique values ({unique_ratio:.2%} of data)\")\n",
            "        print(f\"   Top values: {train[col].value_counts().head(3).to_dict()}\")\n",
            "\n",
            "if not low_variance_features:\n",
            "    print(\"\\n✓ No low-variance features detected\")"
        ]
    },
    # Code cell 5: Future information
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 5. Check for features that might contain future information\n",
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"5. FUTURE INFORMATION LEAKAGE CHECK\")\n",
            "print(\"=\"*80)\n",
            "\n",
            "consequence_keywords = ['score', 'probability', 'risk', 'impact', 'damage', 'loss', 'severity']\n",
            "potential_leakage = []\n",
            "\n",
            "for col in feature_cols:\n",
            "    col_lower = col.lower()\n",
            "    if any(keyword in col_lower for keyword in consequence_keywords):\n",
            "        potential_leakage.append(col)\n",
            "\n",
            "if potential_leakage:\n",
            "    print(\"\\n⚠️ Features that might contain consequence information:\")\n",
            "    for col in potential_leakage:\n",
            "        corr = correlations[col]['correlation']\n",
            "        print(f\"   - {col} (correlation: {corr:.4f})\")\n",
            "    print(\"\\n   ⚠️ Review these to ensure they don't contain post-flood information\")\n",
            "else:\n",
            "    print(\"\\n✓ No obvious consequence-based features detected\")"
        ]
    },
    # Code cell 6: Summary
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 6. Summary and Recommendations\n",
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"LEAKAGE CHECK SUMMARY\")\n",
            "print(\"=\"*80)\n",
            "\n",
            "issues_found = []\n",
            "\n",
            "# Collect all issues\n",
            "high_corr_features = [f for f, s in sorted_corr if abs(s['correlation']) > 0.95]\n",
            "if high_corr_features:\n",
            "    issues_found.append(f\"⚠️ {len(high_corr_features)} feature(s) with correlation >0.95: {', '.join(high_corr_features)}\")\n",
            "\n",
            "if perfect_predictors:\n",
            "    issues_found.append(f\"⚠️ {len(perfect_predictors)} perfect predictor(s): {', '.join(perfect_predictors)}\")\n",
            "\n",
            "if distribution_issues:\n",
            "    issues_found.append(f\"⚠️ {len(distribution_issues)} feature(s) with different train/test distributions\")\n",
            "\n",
            "if low_variance_features:\n",
            "    issues_found.append(f\"⚠️ {len(low_variance_features)} low-variance feature(s)\")\n",
            "\n",
            "if potential_leakage:\n",
            "    issues_found.append(f\"⚠️ {len(potential_leakage)} feature(s) with potential consequence information\")\n",
            "\n",
            "if issues_found:\n",
            "    print(\"\\n⚠️ POTENTIAL ISSUES DETECTED:\\n\")\n",
            "    for i, issue in enumerate(issues_found, 1):\n",
            "        print(f\"{i}. {issue}\")\n",
            "    \n",
            "    print(\"\\n📋 RECOMMENDATIONS:\")\n",
            "    print(\"   1. Investigate flagged features for data leakage\")\n",
            "    print(\"   2. Verify feature definitions with domain experts\")\n",
            "    print(\"   3. Consider removing or transforming suspicious features\")\n",
            "    print(\"   4. Document any features kept despite high correlations\")\n",
            "    print(\"   5. Monitor model performance on validation set vs test set\")\n",
            "else:\n",
            "    print(\"\\n✅ No major leakage issues detected!\")\n",
            "    print(\"   Dataset appears clean for modeling\")\n",
            "\n",
            "print(\"\\n\" + \"=\"*80)"
        ]
    }
]

# Update the section 9 markdown to section 10
notebook['cells'][insert_index]['source'] = [
    "## 10. Key Insights & Next Steps\n",
    "\n",
    "### Summary:\n",
    "- Dataset size: 1.1M training samples, 745K test samples\n",
    "- 20 input features + 1 target (FloodProbability)\n",
    "- Target is continuous (regression task) with values between 0 and 1\n",
    "- Data leakage checks completed\n",
    "\n",
    "### Next Steps:\n",
    "1. Address any data leakage issues identified\n",
    "2. Feature engineering (if needed)\n",
    "3. Train baseline models (Linear Regression, Random Forest, XGBoost)\n",
    "4. Hyperparameter tuning\n",
    "5. Model evaluation and selection\n",
    "6. Generate predictions for test set"
]

# Insert new cells
for i, cell in enumerate(new_cells):
    notebook['cells'].insert(insert_index + i, cell)

# Write back to file
with open('notebooks/01_eda.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✅ Successfully added data leakage check cells to 01_eda.ipynb")
print(f"   - Added {len(new_cells)} new cells")
print("   - Updated section numbering (Section 9 → Section 10)")
print("\nYou can now open the notebook and run the new cells!")
