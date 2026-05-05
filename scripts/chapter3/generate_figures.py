#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第3章：方法对比与框架 - 图表生成脚本（美化版）

优化要点：
1. 全局字号提升
2. 关键信息用高对比色块/粗体突出
3. 精简冗余文字
4. 增大图表尺寸
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Serif CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 13
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11

BASE_DIR = Path('/home/yuye/Resporitory/Cancer-Classification')
OUTPUT_DIR = BASE_DIR / 'outputs' / 'figures' / 'chapter3'


def generate_fig1_fusion_methods_comparison():
    """图1: 四种融合策略结构对比图（美化版）"""
    print("\n生成图1: 四种融合策略结构对比图")

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.suptitle('四种多组学融合策略结构对比', fontsize=20, fontweight='bold', y=0.98)

    methods = [
        ('RNA-only 单模态', 'rna', axes[0,0]),
        ('Concat 早期融合', 'concat', axes[0,1]),
        ('MOFA 潜在因子融合', 'mofa', axes[1,0]),
        ('Stacking 晚期集成', 'stacking', axes[1,1])
    ]

    colors = {'rna': '#2E5AAC', 'meth': '#D9730D', 'concat': '#4A9B4A',
              'mofa': '#D4A017', 'stacking': '#888888', 'classifier': '#7BAFD4'}

    for title, method, ax in methods:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title(title, fontsize=15, fontweight='bold', pad=12)

        if method == 'rna':
            draw_box(ax, 3, 7.5, 4, 1.2, 'RNA-seq\n表达矩阵', colors['rna'])
            draw_arrow(ax, 5, 7.5, 5, 5.8)
            draw_box(ax, 3, 4.3, 4, 1.2, 'SVM / XGB\n分类器', colors['classifier'])
            draw_arrow(ax, 5, 4.3, 5, 2.8)
            draw_box(ax, 3, 1.5, 4, 1.0, '预测输出\n(3类)', '#E7E6E6')

        elif method == 'concat':
            draw_box(ax, 1, 8.0, 3, 1, 'RNA-seq', colors['rna'])
            draw_box(ax, 6, 8.0, 3, 1, 'Methylation', colors['meth'])
            draw_arrow(ax, 2.5, 8.0, 2.5, 6.5)
            draw_arrow(ax, 7.5, 8.0, 7.5, 6.5)
            draw_box(ax, 2.5, 5.3, 5, 1.2, '特征拼接 (Concat)\n[4000 dim]', colors['concat'])
            draw_arrow(ax, 5, 5.3, 5, 3.5)
            draw_box(ax, 3, 2.0, 4, 1.2, '分类器\n(SVM)', colors['classifier'])
            draw_arrow(ax, 5, 2.0, 5, 1.0)
            draw_box(ax, 3, 0.5, 4, 0.8, '预测输出', '#E7E6E6')

        elif method == 'mofa':
            draw_box(ax, 1, 8.0, 3, 1, 'RNA-seq', colors['rna'])
            draw_box(ax, 6, 8.0, 3, 1, 'Methylation', colors['meth'])
            draw_arrow(ax, 2.5, 8.0, 3.5, 6.5)
            draw_arrow(ax, 7.5, 8.0, 6.5, 6.5)
            draw_box(ax, 3, 5.3, 4, 1.2, 'MOFA模型\n(因子分解)', colors['mofa'])
            draw_arrow(ax, 5, 5.3, 5, 3.5)
            draw_box(ax, 3, 2.2, 4, 1.2, '潜在因子 Z\n(15-20 factors)', '#D4A017')
            draw_arrow(ax, 5, 2.2, 5, 1.0)
            draw_box(ax, 3, 0.5, 4, 0.9, '分类器 + 输出', colors['classifier'])

        elif method == 'stacking':
            draw_box(ax, 0.5, 8.0, 3.5, 1, 'Base Model 1\n(RNA SVM)', colors['rna'])
            draw_box(ax, 6, 8.0, 3.5, 1, 'Base Model 2\n(Meth XGB)', colors['meth'])
            draw_arrow(ax, 2.25, 8.0, 3.5, 6.5)
            draw_arrow(ax, 7.75, 8.0, 6.5, 6.5)
            draw_box(ax, 3, 5.0, 4, 1.3, 'OOF预测概率\n(元特征)', '#888888')
            draw_arrow(ax, 5, 5.0, 5, 3.3)
            draw_box(ax, 3, 1.8, 4, 1.3, 'Meta Learner\n(Logistic / XGB)', colors['classifier'])
            draw_arrow(ax, 5, 1.8, 5, 1.0)
            draw_box(ax, 3, 0.5, 4, 0.8, '最终预测', '#E7E6E6')

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'fig1_fusion_methods_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    plt.close()
    print(f"✓ 图1已保存至: {output_path}")
    return output_path


def generate_fig2_method_pipeline_overview():
    """图2: 完整实验流程概览图（美化版）"""
    print("\n生成图2: 实验流程概览图")

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    ax.text(8, 8.5, '多组学乳腺癌亚型分类实验完整流程',
            ha='center', fontsize=20, fontweight='bold')

    def draw_box(x, y, w, h, text, color):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black', linewidth=2.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=11, fontweight='bold')

    def draw_arrow(x1, y1, x2, y2):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle='->,head_length=8,head_width=6',
                                color='#333333', linewidth=2.5)
        ax.add_patch(arrow)

    steps = [
        (0.5, 5.5, 2.8, 1.8, '数据准备\n(RNA+Meth+Label)', '#D9E1F2'),
        (3.8, 5.5, 2.8, 1.8, '预处理\n(Normalize+FS)', '#B4C7E7'),
        (7.1, 5.5, 2.8, 1.8, '融合策略\n(5种方法)', '#C6EFCE'),
        (10.4, 5.5, 2.8, 1.8, '交叉验证\n(5×10 CV)', '#FFEB9C'),
        (13.7, 5.5, 1.5, 1.8, '结果\n输出', '#E2EFDA'),
    ]

    for x, y, w, h, text, color in steps:
        draw_box(x, y, w, h, text, color)

    for i in range(len(steps)-1):
        x1 = steps[i][0] + steps[i][2]
        y1 = steps[i][1] + steps[i][3]/2
        x2 = steps[i+1][0]
        y2 = steps[i+1][1] + steps[i+1][3]/2
        draw_arrow(x1, y1, x2, y2)

    # 精简底部说明
    details = [
        "• 数据源: TCGA-BRCA",
        "• 样本: ~150例, 3亚型",
        "• 特征: Top-2000/modality",
        "• 方法: RNA/Meth/Concat/MOFA/Stacking",
        "• 评估: Acc / BalAcc / Macro-F1 + 95%CI"
    ]
    for i, detail in enumerate(details):
        ax.text(i * 3.1 + 0.5, 3.8, detail, fontsize=11, ha='left', fontweight='bold')

    # 性能范围框
    perf_box = FancyBboxPatch((2, 1.2), 12, 1.8,
                               boxstyle="round,pad=0.3",
                               facecolor='#FFF8DC',
                               edgecolor='#DAA520',
                               linewidth=2.5,
                               alpha=0.9)
    ax.add_patch(perf_box)
    ax.text(8, 2.5, '性能范围: 83.67% (Meth-only) ←→ 90.64% (Concat-SVM)',
            ha='center', fontsize=13, fontweight='bold', color='#333333')
    ax.text(8, 1.7, '关键发现: 简单融合(Concat) ≥ 复杂融合(MOFA/Stacking) | RNA单模态已达89.78%',
            ha='center', fontsize=12, color='#555555', fontweight='bold')

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'fig2_method_pipeline_overview.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    plt.close()
    print(f"✓ 图2已保存至: {output_path}")
    return output_path


def draw_box(ax, x, y, w, h, text, color):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor='black', linewidth=2, alpha=0.9)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=10, fontweight='bold')


def draw_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='->,head_length=6,head_width=4',
                            color='#333333', linewidth=2)
    ax.add_patch(arrow)


def main():
    print("\n" + "="*60)
    print("第3章：方法对比与框架 - 图表生成（美化版）")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_files = []

    try:
        generated_files.append(generate_fig1_fusion_methods_comparison())
        generated_files.append(generate_fig2_method_pipeline_overview())
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n✓ 第3章完成! 共{len(generated_files)}个图表")
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
