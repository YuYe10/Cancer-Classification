#!/usr/bin/env python3
import pandas as pd
import sys

df = pd.read_csv('outputs/logs/class_error_analysis.csv')

latex_lines = [
    r'\begin{table}[H]',
    r'\centering',
    r'\caption{Class-level recall and F1 scores (fold-level mean)}',
    r'\label{tab:class-recall-detailed}',
    r'\small',
    r'\begin{tabular}{lrrrrrrrr}',
    r'\toprule',
    r'Method & LumA Recall & LumB Recall & Basal Recall & LumA F1 & LumB F1 & Basal F1 \\',
    r'\midrule',
]

for _, row in df.iterrows():
    method = row['method']
    luma_recall = f"{row['LumA_recall']:.4f}"
    lumb_recall = f"{row['LumB_recall']:.4f}"
    basal_recall = f"{row['Basal_recall']:.4f}"
    luma_f1 = f"{row['LumA_f1']:.4f}"
    lumb_f1 = f"{row['LumB_f1']:.4f}"
    basal_f1 = f"{row['Basal_f1']:.4f}"
    latex_lines.append(f'{method} & {luma_recall} & {lumb_recall} & {basal_recall} & {luma_f1} & {lumb_f1} & {basal_f1} \\\\')

latex_lines.extend([
    r'\bottomrule',
    r'\end{tabular}',
    r'\end{table}',
])

output = '\n'.join(latex_lines)
print(output)

# Also save to file
with open('outputs/logs/class_recall_table.tex', 'w') as f:
    f.write(output)
