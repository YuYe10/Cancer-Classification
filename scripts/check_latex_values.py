#!/usr/bin/env python3
"""
Check LaTeX numerical consistency against summary_v2.csv
"""
import pandas as pd
import re

# Load summary data
summary = pd.read_csv('outputs/logs/summary_v2.csv')

# Extract main method results (last 4 rows should be the main methods in CV mode)
main_methods = summary[summary['mode'] == 'cv'].tail(4)

print("=" * 80)
print("SUMMARY_V2.CSV MASTER VALUES FOR PAPER VERIFICATION")
print("=" * 80)
print("\n### Main Methods Results (repeated CV, 5 folds x 3 repeats):")
for _, row in main_methods.iterrows():
    exp = row['exp']
    acc_mean = row['accuracy_mean']
    acc_ci_low = row['accuracy_ci95_low']
    acc_ci_high = row['accuracy_ci95_high']
    bal_acc_mean = row['balanced_accuracy_mean']
    macro_f1_mean = row['macro_f1_mean']
    print(f"  {exp}:")
    print(f"    - Accuracy: {acc_mean:.4f} (95% CI: [{acc_ci_low:.4f}, {acc_ci_high:.4f}])")
    print(f"    - Balanced Acc: {bal_acc_mean:.4f}")
    print(f"    - Macro-F1: {macro_f1_mean:.4f}")

print("\n### Sample counts (HER2 dropped?):")
# Try to find sample info
if 'sample_count' in summary.columns:
    print(summary[['exp', 'sample_count']].tail(4))

print("\n### Key reference points for paper:")
print("1. Total samples in dataset: Check datasets/*.csv")
print("2. CV protocol: 5 folds x 3 repeats (repeated stratified)")
print("3. Dropped labels: HER2 (insufficient samples)")
print("4. Effective labels: LumA, LumB, Basal (3-class classification)")

print("\nTo check paper consistency:")
print("- Read abstract, introduction, data, results sections")
print("- Compare all numerical citations to this reference table")
print("- Mark any discrepancies with [VERIFY] tag in paper")
