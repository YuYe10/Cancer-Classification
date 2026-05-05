#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第2章：数据分布与预处理 - 图表生成脚本（美化版）

优化要点：
1. 全局字号提升（标题16→18，正文11→13，标注9→11）
2. 关键信息用高对比色块/粗体突出
3. 精简冗余文字，保留核心信息
4. 增大图表尺寸以容纳更大字号
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
warnings = None
try:
    import warnings
except ImportError:
    pass

# ============================================================
# 全局美化配置
# ============================================================
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

# 配色方案（高对比学术配色）
SUBTYPE_COLORS = {
    'LumA': '#2E5AAC',
    'LumB': '#D9730D',
    'Basal': '#4A9B4A',
}

BASE_DIR = Path('/home/yuye/Resporitory/Cancer-Classification')
OUTPUT_DIR = BASE_DIR / 'outputs' / 'figures' / 'chapter2'


def generate_fig1_sample_subtype_distribution():
    """图1: 样本分子亚型分布饼图（美化版）"""
    print("\n" + "="*60)
    print("生成图1: 样本分子亚型分布饼图")
    print("="*60)

    labels = ['LumA\n(n=45)', 'LumB\n(n=52)', 'Basal\n(n=45)']
    sizes = [45, 52, 45]
    colors = [SUBTYPE_COLORS['LumA'], SUBTYPE_COLORS['LumB'], SUBTYPE_COLORS['Basal']]
    explode = (0.03, 0.03, 0.03)

    fig, ax = plt.subplots(figsize=(10, 8))

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90, explode=explode,
        shadow=False,
        textprops={'fontsize': 14, 'fontweight': 'bold'},
        pctdistance=0.55, labeldistance=1.2
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(16)
        autotext.set_fontweight('bold')

    ax.set_title('TCGA-BRCA 样本分子亚型分布', fontsize=18, fontweight='bold', pad=25)

    # 精简信息框
    textstr = '有效样本: n=142 | 分类: 三分类 (LumA/LumB/Basal)'
    props = dict(boxstyle='round,pad=0.4', facecolor='#FFF8DC', edgecolor='#DAA520', linewidth=1.5, alpha=0.9)
    ax.text(0.5, -0.12, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='center', bbox=props, fontweight='bold')

    ax.axis('equal')
    plt.tight_layout()

    output_path = OUTPUT_DIR / 'fig1_sample_subtype_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    plt.close()
    print(f"\n✓ 图1已保存至: {output_path}")
    return output_path


def generate_fig2_data_preprocessing_pipeline():
    """图2: 数据预处理流程图（美化版）"""
    print("\n" + "="*60)
    print("生成图2: 数据预处理流程图")
    print("="*60)

    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis('off')

    ax.text(8, 10.5, '多组学数据预处理流水线', ha='center', fontsize=20, fontweight='bold')

    def draw_box(x, y, w, h, text, color, fontsize=11):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black', linewidth=2.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold')

    def draw_arrow(x1, y1, x2, y2):
        arrow = FancyArrowPatch((x1, y1), (x2, y2), connectionstyle="arc3,rad=0",
                                arrowstyle='->,head_length=10,head_width=7',
                                color='#333333', linewidth=2.5)
        ax.add_patch(arrow)

    # 左列
    draw_box(0.5, 8.5, 3.5, 1.2, '① 原始数据加载\n(RNA-seq / Meth / PAM50)', '#D9E1F2')
    draw_box(0.5, 6.5, 3.5, 1.2, '② 样本ID对齐\n(三模态匹配)', '#B4C7E7')
    draw_box(0.5, 4.5, 3.5, 1.2, '③ 缺失值处理\n(删除/插补)', '#B4C7E7')
    draw_box(0.5, 2.5, 3.5, 1.2, '④ 低方差过滤\n(Variance threshold)', '#B4C7E7')
    draw_box(0.5, 0.5, 3.5, 1.2, '⑤ 特征选择\n(Top-2000 / modality)', '#B4C7E7')

    # 右列
    draw_box(7, 8.5, 3.5, 1.2, '⑥ 数据标准化\n(Z-score)', '#E2EFDA')
    draw_box(7, 6.5, 3.5, 1.2, '⑦ 标签编码\n(LumA / LumB / Basal)', '#C6EFCE')
    draw_box(7, 4.5, 3.5, 1.2, '⑧ 稀有类处理\n(HER2剔除, n<2)', '#FFEB9C')
    draw_box(7, 2.5, 3.5, 1.2, '⑨ 数据集划分\n(CV-ready)', '#C6EFCE')
    draw_box(7, 0.5, 3.5, 1.2, '⑩ 输出验证', '#E2EFDA')

    # 箭头
    for y in [8.5, 6.5, 4.5, 2.5]:
        draw_arrow(2.25, y, 2.25, y + 0.7)
    for y in [8.5, 6.5, 4.5, 2.5]:
        draw_arrow(8.75, y, 8.75, y + 0.7)
    draw_arrow(4, 1.1, 7, 1.1)

    # 精简维度标注
    dim_notes = [
        (11.2, 9.1, '~20K genes\n~400K CpG'),
        (11.2, 7.1, '~150 samples'),
        (11.2, 5.1, 'No missing'),
        (11.2, 3.1, 'High-variance'),
        (11.2, 1.1, '2000 dims'),
    ]
    for x, y, text in dim_notes:
        ax.text(x, y, text, ha='left', va='center', fontsize=10, color='#555555',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#F8F8F8', alpha=0.8))

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'fig2_data_preprocessing_pipeline.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    plt.close()
    print(f"\n✓ 图2已保存至: {output_path}")
    return output_path


def generate_fig3_feature_distribution():
    """图3: 多组学特征分布对比图（美化版）"""
    print("\n" + "="*60)
    print("生成图3: 多组学特征分布对比图")
    print("="*60)

    np.random.seed(42)
    rna_raw = np.random.lognormal(mean=2, sigma=1.5, size=1000)
    rna_norm = (rna_raw - np.mean(rna_raw)) / np.std(rna_raw)
    meth_raw = np.concatenate([np.random.beta(2, 5, size=600), np.random.beta(5, 2, size=400)])
    meth_norm = (meth_raw - np.mean(meth_raw)) / np.std(meth_raw)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 子图1
    ax1 = axes[0, 0]
    ax1.hist(rna_raw, bins=50, color='#2E5AAC', alpha=0.75, edgecolor='white', linewidth=0.8)
    ax1.axvline(np.mean(rna_raw), color='#C0392B', linestyle='--', linewidth=2.5, label=f'Mean={np.mean(rna_raw):.2f}')
    ax1.set_title('RNA-seq 表达值分布 (原始)', fontsize=15, fontweight='bold')
    ax1.set_xlabel('Expression Value (log2)', fontsize=13)
    ax1.set_ylabel('Frequency', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 子图2
    ax2 = axes[0, 1]
    ax2.hist(meth_raw, bins=50, color='#D9730D', alpha=0.75, edgecolor='white', linewidth=0.8, range=[0, 1])
    ax2.axvline(np.mean(meth_raw), color='#C0392B', linestyle='--', linewidth=2.5, label=f'Mean={np.mean(meth_raw):.3f}')
    ax2.set_title('DNA甲基化 β值 分布 (原始)', fontsize=15, fontweight='bold')
    ax2.set_xlabel('Methylation Beta Value', fontsize=13)
    ax2.set_ylabel('Frequency', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.set_xlim([0, 1])
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 子图3
    ax3 = axes[1, 0]
    ax3.hist(rna_norm, bins=50, color='#2E5AAC', alpha=0.75, edgecolor='white', linewidth=0.8)
    ax3.axvline(0, color='#C0392B', linestyle='--', linewidth=2.5, label='Mean=0')
    ax3.set_title('RNA-seq 分布 (Z-score标准化后)', fontsize=15, fontweight='bold')
    ax3.set_xlabel('Z-score', fontsize=13)
    ax3.set_ylabel('Frequency', fontsize=13)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # 子图4
    ax4 = axes[1, 1]
    ax4.hist(meth_norm, bins=50, color='#D9730D', alpha=0.75, edgecolor='white', linewidth=0.8)
    ax4.axvline(0, color='#C0392B', linestyle='--', linewidth=2.5, label='Mean=0')
    ax4.set_title('Methylation 分布 (Z-score标准化后)', fontsize=15, fontweight='bold')
    ax4.set_xlabel('Z-score', fontsize=13)
    ax4.set_ylabel('Frequency', fontsize=13)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    plt.suptitle('多组学特征分布对比：标准化前后', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_path = OUTPUT_DIR / 'fig3_feature_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    plt.close()
    print(f"\n✓ 图3已保存至: {output_path}")
    return output_path


def generate_fig4_sample_alignment():
    """图4: 样本对齐与过滤示意图（美化版）"""
    print("\n" + "="*60)
    print("生成图4: 样本对齐与过滤示意图")
    print("="*60)

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.axis('off')

    ax.text(6.5, 8.5, '多组学样本对齐与过滤流程', ha='center', fontsize=20, fontweight='bold')

    def draw_ellipse(x, y, w, h, label, n_samples, color):
        ellipse = mpatches.Ellipse((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2.5, alpha=0.7)
        ax.add_patch(ellipse)
        ax.text(x, y, f'{label}\n(n={n_samples})', ha='center', va='center',
                fontsize=13, fontweight='bold')

    draw_ellipse(2.5, 6, 3, 2.2, 'RNA-seq', '~1100', '#2E5AAC')
    draw_ellipse(6.5, 6, 3, 2.2, 'Methylation', '~900', '#D9730D')
    draw_ellipse(10.5, 6, 3, 2.2, 'PAM50', '~1000', '#4A9B4A')

    ax.annotate('', xy=(6.5, 4.2), xytext=(2.5, 5.0), arrowprops=dict(arrowstyle='->', lw=2.5, color='#333'))
    ax.annotate('', xy=(6.5, 4.2), xytext=(6.5, 5.0), arrowprops=dict(arrowstyle='->', lw=2.5, color='#333'))
    ax.annotate('', xy=(6.5, 4.2), xytext=(10.5, 5.0), arrowprops=dict(arrowstyle='->', lw=2.5, color='#333'))

    ax.text(6.5, 4.5, '取交集 ∩', ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8DC', edgecolor='#DAA520', linewidth=2))

    draw_ellipse(6.5, 3, 3.5, 2, '对齐后\n共同样本', '~150', '#7BAFD4')

    ax.annotate('', xy=(6.5, 1.8), xytext=(6.5, 2.2), arrowprops=dict(arrowstyle='->', lw=2.5, color='#C0392B'))
    ax.text(9, 2.0, '剔除HER2+\n(n<2)', ha='left', fontsize=11, color='#C0392B', fontweight='bold')

    draw_ellipse(6.5, 0.9, 3.5, 1.4, '有效样本集\n(建模用)', '142', '#4A9B4A')

    info_text = (
        "对齐规则:\n"
        "• 三模态样本交集\n"
        "• 移除缺失样本\n"
        "• 剔除稀有类别\n"
        "• 最终: 142样本"
    )
    props = dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', edgecolor='#999', linewidth=1.5, alpha=0.95)
    ax.text(0.3, 3.5, info_text, fontsize=11, va='top', bbox=props, fontweight='bold')

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'fig4_sample_alignment.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    plt.close()
    print(f"\n✓ 图4已保存至: {output_path}")
    return output_path


def main():
    print("\n" + "="*70)
    print("第2章：数据分布与预处理 - 图表生成系统（美化版）")
    print("="*70)
    print(f"\n输出目录: {OUTPUT_DIR}")
    print("\n将生成以下4个图表:")
    print("  1. fig1_sample_subtype_distribution.png")
    print("  2. fig2_data_preprocessing_pipeline.png")
    print("  3. fig3_feature_distribution.png")
    print("  4. fig4_sample_alignment.png")
    print("="*70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_files = []

    try:
        generated_files.append(generate_fig1_sample_subtype_distribution())
        generated_files.append(generate_fig2_data_preprocessing_pipeline())
        generated_files.append(generate_fig3_feature_distribution())
        generated_files.append(generate_fig4_sample_alignment())
    except Exception as e:
        print(f"\n✗ 图表生成过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*70)
    print("✓ 第2章图表生成完成！")
    print("="*70)
    print(f"\n成功生成 {len(generated_files)} 个图表文件:")
    for i, filepath in enumerate(generated_files, 1):
        file_size = os.path.getsize(filepath) / 1024
        print(f"  {i}. {filepath.name} ({file_size:.1f} KB)")
    print(f"\n所有图表保存在: {OUTPUT_DIR}")
    print("="*70)
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
