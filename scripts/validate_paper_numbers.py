#!/usr/bin/env python3
"""
Generate a comprehensive numerical validation report for the paper.
"""
import pandas as pd
import re
from pathlib import Path

# Load reference data from summary_v2.csv
df = pd.read_csv('outputs/logs/summary_v2.csv')
cv_latest = df[df['mode'] == 'cv'].tail(4)

print("=" * 100)
print("PAPER NUMERICAL VERIFICATION REFERENCE TABLE")
print("=" * 100)
print("\n### MASTER VALUES FROM summary_v2.csv (rows 20-23, CV mode, 5×3 repeats):\n")

methods_data = {}
for idx, row in cv_latest.iterrows():
    exp = row['exp']
    acc_mean = float(row['accuracy_mean'])
    acc_low = float(row['accuracy_ci95_low'])
    acc_high = float(row['accuracy_ci95_high'])
    ba_mean = float(row['balanced_accuracy_mean'])
    f1_mean = float(row['macro_f1_mean'])
    
    methods_data[exp] = {
        'accuracy_mean': acc_mean,
        'accuracy_ci95_low': acc_low,
        'accuracy_ci95_high': acc_high,
        'balanced_accuracy_mean': ba_mean,
        'macro_f1_mean': f1_mean,
    }
    
    print(f"{exp.upper():12} | Accuracy: {acc_mean:.4f} (95% CI: [{acc_low:.4f}, {acc_high:.4f}])")
    print(f"             | Balanced Acc: {ba_mean:.4f} | Macro-F1: {f1_mean:.4f}")
    print()

# Load class-level data
class_analysis = pd.read_csv('outputs/logs/class_error_analysis.csv')

print("\n### CLASS-LEVEL METRICS (fold-level mean):\n")
for _, row in class_analysis.iterrows():
    method = row['method'].upper()
    luma_recall = float(row['LumA_recall'])
    lumb_recall = float(row['LumB_recall'])
    basal_recall = float(row['Basal_recall'])
    print(f"{method:12} | LumA Recall: {luma_recall:.4f} | LumB Recall: {lumb_recall:.4f} | Basal Recall: {basal_recall:.4f}")

print("\n### STABILITY TEST RESULTS (from outputs/logs/stability/):\n")
# Try to load stability results
try:
    stability = pd.read_csv('outputs/logs/stability/stability_sweeps_summary.csv')
    print("Feature-dimension sensitivity:")
    for _, row in stability[stability['sweep_type'] == 'feature_dim'].iterrows():
        dim = int(row['param_value'])
        acc = float(row['accuracy_mean'])
        print(f"  top_var={dim}: Accuracy = {acc:.4f}")
    print("\nRepeats convergence:")
    for _, row in stability[stability['sweep_type'] == 'repeat_convergence'].iterrows():
        repeats = int(row['param_value'])
        acc = float(row['accuracy_mean'])
        print(f"  repeats={repeats}: Accuracy = {acc:.4f}")
except FileNotFoundError:
    print("  (Stability file not found yet)")

print("\n" + "=" * 100)
print("CHECKLIST: Search paper sections for these values and verify consistency:")
print("=" * 100)
print("""
1. ABSTRACT (main.tex):
   [ ] Concat Accuracy: 0.9012 [0.8609, 0.9415]
   [ ] RNA-only Accuracy: 0.8655
   [ ] Stacking Accuracy: 0.8488
   [ ] MOFA Accuracy: 0.7857
   [ ] Sample handling: HER2 < 2 (dropped)

2. DATA SECTION (sections/02-data.tex):
   [ ] Sample counts for each class
   [ ] HER2 dropped mention

3. RESULTS SECTION (sections/09-results-visualization.tex):
   [ ] Main table (method_comparison_table.tex) values
   [ ] Class-level recall rates (LumB recall comparisons)
   [ ] Stability test results (if mentioned)

4. DISCUSSION (sections/10-discussion.tex):
   [ ] Any quantitative claims cite correct numbers
""")
