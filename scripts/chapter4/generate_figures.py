#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第4章：实验设计与统计框架 - 图表生成脚本（美化版）

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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
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
OUTPUT_DIR = BASE_DIR / 'outputs' / 'figures' / 'chapter4'


def generate_fig1_cv_protocol():
    """图1: 重复分层5折交叉验证协议示意图（美化版）"""
    print("\n生成图1: CV协议示意图")

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.5, '重复分层5折交叉验证协议 (5 folds × 10 repeats)',
            ha='center', fontsize=18, fontweight='bold')

    def draw_box(x, y, w, h, text, color):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                             facecolor=color, edgecolor='black', linewidth=2, alpha=0.9)
        ax.add_patch(box)
        if text:
            ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                    fontsize=11, fontweight='bold')

    draw_box(0.5, 5.5, 13, 1.2, '完整数据集 (n≈150样本, 3类: LumA / LumB / Basal)', '#D9E1F2')

    fold_labels = ['Fold 1\n(测试)', 'Fold 2\n(训练)', 'Fold 3\n(训练)',
                   'Fold 4\n(训练)', 'Fold 5\n(训练)']
    fold_colors = ['#D9730D', '#B4C7E7', '#B4C7E7', '#B4C7E7', '#B4C7E7']

    for i, (label, color) in enumerate(zip(fold_labels, fold_colors)):
        x = 0.8 + i * 2.5
        draw_box(x, 3.8, 2.3, 1.3, label, color)

    ax.annotate('', xy=(7, 3.5), xytext=(7, 5.5),
                arrowprops=dict(arrowstyle='->', lw=2.5))
    ax.text(8.8, 4.6, '分层随机划分\n(保持类别比例)', fontsize=11, color='#333333', fontweight='bold')

    ax.text(7, 2.8, '↓  重复 10 次独立随机划分  ↓', ha='center', fontsize=13,
            fontweight='bold', color='#7030A0')

    draw_box(2.5, 0.8, 9, 1.6,
             '总测试折数: 50 (5 folds × 10 repeats)\n'
             '统计指标: 均值 ± 标准差 | 95% CI: μ ± 1.96·σ/√n\n'
             '评估指标: Accuracy | Balanced Accuracy | Macro-F1',
             '#FFF2CC')

    params_text = (
        "关键参数:\n"
        "· 外层折数: K=5\n"
        "· 重复次数: R=10\n"
        "· 总测试折: N=50\n"
        "· 分层策略: Stratified\n"
        "· 内层CV: 5折"
    )
    props = dict(boxstyle='round,pad=0.4', facecolor='#E7E6E6', edgecolor='#999', linewidth=1.5, alpha=0.95)
    ax.text(11.5, 2.8, params_text, fontsize=11, va='top', bbox=props, fontweight='bold')

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'fig1_cv_protocol.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    plt.close()
    print(f"✓ 图1已保存至: {output_path}")
    return output_path


def generate_fig2_experiment_matrix():
    """图2: 实验矩阵设计总览图（美化版）"""
    print("\n生成图2: 实验矩阵总览")

    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(7, 9.5, '实验设计矩阵总览 (共36组实验)',
            ha='center', fontsize=20, fontweight='bold')

    def draw_matrix_box(x, y, w, h, title, items, color):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                             facecolor=color, edgecolor='black', linewidth=2.5, alpha=0.8)
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 0.4, title, ha='center', va='top',
                fontsize=14, fontweight='bold')
        for i, item in enumerate(items):
            ax.text(x + 0.3, y + h - 0.9 - i*0.5, f"• {item}",
                    fontsize=11, va='top', fontweight='bold')

    draw_matrix_box(0.5, 4.2, 4.2, 4.5,
                    '主线对比 (6组)',
                    ['RNA-only (SVM)',
                     'Meth-only (XGB)',
                     'Concat-SVM (SOTA)',
                     'Concat-XGB',
                     'MOFA-20 factors',
                     'Stacking (XGB+LR)'],
                    '#D9E1F2')

    draw_matrix_box(4.9, 4.2, 4.2, 4.5,
                    '敏感性分析 (15组)',
                    ['特征维度: 100/300/500/1000/1500',
                     'CV折数: 3/5/10',
                     'CV重复: 3/5/10/15',
                     'MOFA因子: 10/15/20/25'],
                    '#E2EFDA')

    draw_matrix_box(9.3, 4.2, 4.2, 4.5,
                    '消融 & Stacking (20组)',
                    ['去除RNA组件',
                     '去除Meth组件',
                     'RF代替SVM',
                     '类别平衡策略',
                     'Stacking变体: 7种配置'],
                    '#FFF2CC')

    stats_box = FancyBboxPatch((1, 0.4), 12, 3,
                                boxstyle="round,pad=0.4",
                                facecolor='#F5F5F5',
                                edgecolor='#333333',
                                linewidth=2.5,
                                alpha=0.95)
    ax.add_patch(stats_box)

    stats_text = (
        "实验规模统计:\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "主线实验: 6种方法 × 1配置 = 6组  |  "
        "敏感性分析: 4类 × 多配置 ≈ 14-15组  |  "
        "消融+Stacking: ≈ 20组\n"
        "总计: 41组独立实验配置  |  "
        "每组: 5-fold × 10-repeat CV  |  "
        "最大测试折数: 75 (repeat15)\n"
        "评估口径: Accuracy / Balanced Accuracy / Macro-F1 (+ 95% CI)"
    )
    ax.text(7, 2.5, stats_text, ha='center', va='center',
            fontsize=11, fontweight='bold')

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'fig2_experiment_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    plt.close()
    print(f"✓ 图2已保存至: {output_path}")
    return output_path


def main():
    print("\n" + "="*60)
    print("第4章：实验设计与统计框架 - 图表生成（美化版）")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_files = []

    try:
        generated_files.append(generate_fig1_cv_protocol())
        generated_files.append(generate_fig2_experiment_matrix())
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n✓ 第4章完成! 共{len(generated_files)}个图表")
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
