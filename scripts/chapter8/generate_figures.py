#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第8章：讨论与分析 - 图表生成脚本（美化版）

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
OUTPUT_DIR = BASE_DIR / 'outputs' / 'figures' / 'chapter8'


def generate_fig1_bias_variance_tradeoff():
    print("\n生成图1: 偏差-方差权衡图")

    fig, ax1 = plt.subplots(figsize=(13, 8))

    complexity = np.linspace(1, 10, 100)
    bias_squared = 0.15 * np.exp(-0.3 * complexity) + 0.01
    variance = 0.005 * (complexity ** 1.8)
    noise = 0.02 * np.ones_like(complexity)
    total_error = bias_squared + variance + noise

    ax1.plot(complexity, bias_squared, 'b-', linewidth=3, label='Bias² (偏差²)', color='#2E5AAC')
    ax1.plot(complexity, variance, 'r-', linewidth=3, label='Variance (方差)', color='#D9730D')
    ax1.plot(complexity, total_error, 'k--', linewidth=3, label='Total Error (总误差)')
    ax1.axhline(y=noise[0], color='gray', linestyle=':', linewidth=2, label='Irreducible Noise (不可约误差)')

    optimal_idx = np.argmin(total_error)
    optimal_complexity = complexity[optimal_idx]
    optimal_error = total_error[optimal_idx]

    ax1.scatter([optimal_complexity], [optimal_error], s=250, c='gold',
               marker='*', edgecolors='black', linewidths=2.5, zorder=5,
               label=f'最优点 (Complexity≈{optimal_complexity:.1f})')

    methods_position = {
        'Meth-only': (2, 0.085, '#D9730D'),
        'RNA-only': (3, 0.055, '#2E5AAC'),
        'Concat-SVM': (5, 0.042, '#4A9B4A'),
        'MOFA-20': (7, 0.058, '#D4A017'),
        'Stacking': (9, 0.095, '#888888'),
    }

    for method, (x_pos, y_pos, color) in methods_position.items():
        ax1.annotate(method, xy=(x_pos, y_pos), xytext=(x_pos + 0.3, y_pos + 0.015),
                     fontsize=11, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=color, lw=2))
        ax1.scatter([x_pos], [y_pos], s=150, c=color, edgecolors='black',
                   linewidths=2, zorder=4, alpha=0.9)

    ax1.set_xlabel('模型复杂度 (Model Complexity)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('误差 (Error)', fontsize=13, fontweight='bold')
    ax1.set_title('偏差-方差权衡解释方法性能差异\n(Bias-Variance Tradeoff Analysis)',
                  fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlim(0, 10.5)
    ax1.set_ylim(0, 0.16)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    textstr = ('关键洞察:\n'
               '• Concat-SVM位于最优区域\n'
               '• Stacking因高方差导致总误差上升\n'
               '• 简单方法在小样本下更稳健')
    props = dict(boxstyle='round,pad=0.5', facecolor='#FFF8DC', edgecolor='#DAA520', linewidth=2, alpha=0.9)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=props, fontweight='bold')

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'fig1_bias_variance_tradeoff.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    plt.close()
    print(f"✓ 图1已保存至: {output_path}")
    return output_path


def generate_fig2_complexity_vs_performance():
    print("\n生成图2: 复杂度-性能关系图")

    methods_data = [
        ('Meth-only', 2, 0.8367, '单模态基线'),
        ('RNA-only', 3, 0.8978, '单模态基线'),
        ('Concat-SVM', 5, 0.9064, '早期融合 SOTA'),
        ('MOFA-20', 7, 0.9000, '潜在因子融合'),
        ('Stacking', 9, 0.8556, '晚期集成'),
    ]

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['#D9730D', '#2E5AAC', '#4A9B4A', '#D4A017', '#888888']
    markers = ['o', 's', '^', 'D', 'v']

    for i, (method, complexity, perf, desc) in enumerate(methods_data):
        ax.scatter(complexity, perf, s=450, c=colors[i], marker=markers[i],
                   edgecolors='black', linewidths=2.5, zorder=5, alpha=0.9)
        offset_y = 0.014 if i != 2 else -0.02
        ax.annotate(f'{method}\n({desc})\nAcc={perf:.4f}',
                    xy=(complexity, perf), xytext=(complexity + 0.25, perf + offset_y),
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.35),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    complexities = [m[1] for m in methods_data]
    perfs = [m[2] for m in methods_data]
    z = np.polyfit(complexities, perfs, 2)
    p = np.poly1d(z)
    x_trend = np.linspace(1, 10, 100)
    ax.plot(x_trend, p(x_trend), 'k--', linewidth=2.5, alpha=0.6,
            label='二次拟合趋势 (倒U型)')

    ax.axvspan(4, 6, alpha=0.15, color='#4A9B4A', label='最优复杂度区间')
    ax.axhline(y=0.90, color='#C0392B', linestyle=':', linewidth=2, alpha=0.7, label='90% 性能基准')

    ax.set_xlabel('方法复杂度 (主观评分: 1=简单 → 10=复杂)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('融合策略复杂度 vs 分类性能关系\n(验证"简单方法优势"现象)',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.80, 0.94)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    insight_text = ("核心发现:\n"
                   "• 最优性能不在最高复杂度\n"
                   "• Concat-SVM (复杂度5) 取得SOTA\n"
                   "• Stacking (复杂度9) 反而表现较差\n"
                   "• 小样本场景下: 简单 > 复杂")
    props = dict(boxstyle='round,pad=0.5', facecolor='#E7E6E6', edgecolor='#999', linewidth=1.5, alpha=0.95)
    ax.text(0.02, 0.98, insight_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontweight='bold')

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'fig2_complexity_vs_performance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    plt.close()
    print(f"✓ 图2已保存至: {output_path}")
    return output_path


def generate_fig3_stacking_failure_analysis():
    print("\n生成图3: Stacking失败分析")

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.axis('off')

    ax.text(6.5, 8.5, 'Stacking方法系统性欠拟合原因分析',
            ha='center', fontsize=20, fontweight='bold')

    def draw_reason_box(x, y, title, content, impact_color):
        box = mpatches.FancyBboxPatch((x, y), 3.5, 2.8,
                                       boxstyle="round,pad=0.3",
                                       facecolor='#F5F5F5',
                                       edgecolor=impact_color,
                                       linewidth=3,
                                       alpha=0.95)
        ax.add_patch(box)
        ax.text(x + 1.75, y + 2.5, title, ha='center', va='center',
                fontsize=13, fontweight='bold', color=impact_color)
        for i, line in enumerate(content):
            ax.text(x + 0.2, y + 1.9 - i*0.5, f"• {line}",
                    fontsize=11, va='top', fontweight='bold')

    draw_reason_box(0.5, 4.5, '① 元特征维度不足',
                   ['元特征仅6维 (3类×2视图)',
                    '远低于传统Stacking场景',
                    '元学习器容量受限',
                    '难以学习互补模式'],
                   '#D9730D')

    draw_reason_box(4.8, 4.5, '② 内层CV数据不足',
                   ['外层训练折仅~120样本',
                    '内层5折→每折~96样本',
                    'OOF概率估计噪声大',
                    '信号质量有限'],
                   '#D4A017')

    draw_reason_box(9.1, 4.5, '③ 超参数未优化',
                   ['使用默认超参数',
                    'Logistic C=1.0 (未调优)',
                    '未针对概率特征优化',
                    '正则化策略不匹配'],
                   '#888888')

    summary_box = mpatches.FancyBboxPatch((1, 0.4), 11, 3.2,
                                          boxstyle="round,pad=0.4",
                                          facecolor='#FFF8DC',
                                          edgecolor='#333333',
                                          linewidth=2.5,
                                          alpha=0.95)
    ax.add_patch(summary_box)

    summary_text = (
        "实证结果验证:\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "• XGB+XGB (设计为SOTA): Macro-F1=78.66% ← 全实验最低之一\n"
        "• RF+XGB-Hybrid (非主流配置): Macro-F1=85.07% ← Stacking组最佳\n"
        "• XGB+Logistic (标准配置): Macro-F1=84.79% ← 中等水平\n\n"
        "结论: Stacking对base/meta选择极为敏感，不存在普适最优配置。\n"
        "       在小样本(n≈150)场景下，增加融合层级反而引入过拟合风险。"
    )
    ax.text(6.5, 2.9, summary_text, ha='center', va='center',
            fontsize=11, fontweight='bold')

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'fig3_stacking_failure_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    plt.close()
    print(f"✓ 图3已保存至: {output_path}")
    return output_path


def main():
    print("\n" + "="*60)
    print("第8章：讨论与分析 - 图表生成（美化版）")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_files = []

    try:
        generated_files.append(generate_fig1_bias_variance_tradeoff())
        generated_files.append(generate_fig2_complexity_vs_performance())
        generated_files.append(generate_fig3_stacking_failure_analysis())
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n✓ 第8章完成! 共{len(generated_files)}个图表")
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
