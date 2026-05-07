#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交叉验证重复次数分析图表生成脚本

功能说明：
1. 从实验日志文件读取不同CV重复次数的实验结果
2. 分析repeat=5、10、15时的Accuracy变化趋势
3. 生成可视化图表，展示选择repeat=10的合理性
4. 输出位置：outputs/figures/cv_repeat_analysis.png
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==================== 全局配置 ====================
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Serif CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.rcParams['figure.dpi'] = 300  # 图像分辨率
plt.rcParams['savefig.dpi'] = 300  # 保存图像分辨率
plt.rcParams['font.size'] = 12  # 默认字号
plt.rcParams['axes.titlesize'] = 16  # 图表标题字号
plt.rcParams['axes.labelsize'] = 13  # 坐标轴标签字号
plt.rcParams['xtick.labelsize'] = 12  # X轴刻度字号
plt.rcParams['ytick.labelsize'] = 12  # Y轴刻度字号
plt.rcParams['legend.fontsize'] = 11  # 图例字号

# 路径配置
BASE_DIR = Path('/home/yuye/Resporitory/Cancer-Classification')
OUTPUT_DIR = BASE_DIR / 'outputs' / 'figures'


def generate_cv_repeat_analysis():
    """
    生成CV重复次数分析图表
    
    主要步骤：
    1. 读取CSV日志文件
    2. 提取repeat=5、10、15的实验数据
    3. 绘制准确率趋势图和多指标对比图
    4. 保存图表并输出数据摘要
    """
    print("\n生成CV重复次数分析图表")
    
    # ==================== 1. 数据读取 ====================
    # 定义CSV文件路径
    csv_path = BASE_DIR / 'outputs' / 'logs' / 'summary_pre.csv'
    
    # 手动解析CSV文件（避免依赖pandas）
    data = []  # 存储所有数据行
    headers = []  # 存储表头
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                headers = row  # 第一行为表头
            else:
                data.append(row)  # 其余行为数据
    
    # 创建列名到索引的映射，方便后续取值
    col_idx = {col: idx for idx, col in enumerate(headers)}
    
    # ==================== 2. 数据筛选 ====================
    # 定义需要分析的配置文件及其对应的repeat次数
    repeat_configs = {
        'exp_concat_cv_repeat5.yaml': 5,
        'exp_concat_cv_repeat10.yaml': 10,
        'exp_concat_cv_repeat15.yaml': 15
    }
    
    # 初始化数据列表
    repeats = []  # CV重复次数
    accuracies = []  # 准确率均值
    accuracy_lows = []  # 准确率95%CI下限
    accuracy_highs = []  # 准确率95%CI上限
    balanced_accs = []  # 平衡准确率
    macro_f1s = []  # Macro-F1分数
    
    # 遍历配置文件，提取对应数据
    for config_file, repeat_num in repeat_configs.items():
        target_config = f'config/{config_file}'
        for row in data:
            if row[col_idx['config']] == target_config:
                repeats.append(repeat_num)
                accuracies.append(float(row[col_idx['accuracy_mean']]) * 100)
                accuracy_lows.append(float(row[col_idx['accuracy_ci95_low']]) * 100)
                accuracy_highs.append(float(row[col_idx['accuracy_ci95_high']]) * 100)
                balanced_accs.append(float(row[col_idx['balanced_accuracy_mean']]) * 100)
                macro_f1s.append(float(row[col_idx['macro_f1_mean']]) * 100)
                break  # 找到匹配项后跳出内层循环
    
    # 按repeat次数排序数据
    sorted_indices = np.argsort(repeats)
    repeats = [repeats[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    accuracy_lows = [accuracy_lows[i] for i in sorted_indices]
    accuracy_highs = [accuracy_highs[i] for i in sorted_indices]
    balanced_accs = [balanced_accs[i] for i in sorted_indices]
    macro_f1s = [macro_f1s[i] for i in sorted_indices]
    
    # ==================== 3. 图表绘制 ====================
    # 创建双面板图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # -------------------- 子图1：准确率与置信区间 --------------------
    # 绘制准确率趋势线
    ax1.plot(repeats, accuracies, 'o-', linewidth=3, markersize=12, 
             color='#4A9B4A', label='准确率均值')
    # 绘制95%置信区间填充
    ax1.fill_between(repeats, accuracy_lows, accuracy_highs, 
                     alpha=0.2, color='#4A9B4A', label='95%置信区间')
    
    # 标记最优配置（准确率最高的点）
    optimal_idx = accuracies.index(max(accuracies))
    optimal_repeat = repeats[optimal_idx]
    optimal_acc = accuracies[optimal_idx]
    
    ax1.scatter([optimal_repeat], [optimal_acc], s=300, color='#D9730D', 
                marker='*', edgecolors='black', linewidths=2.5, zorder=5,
                label=f'最优配置 (repeat={optimal_repeat})')
    
    # 设置图表属性
    ax1.set_xlabel('CV重复次数', fontsize=13, fontweight='bold')
    ax1.set_ylabel('准确率 (%)', fontsize=13, fontweight='bold')
    ax1.set_title('准确率 vs CV重复次数 (Concat-SVM)',
                  fontsize=16, fontweight='bold', pad=15)
    ax1.set_xticks(repeats)  # 设置X轴刻度为实际repeat值
    ax1.set_ylim([85, 93])  # 设置Y轴范围
    ax1.grid(True, linestyle='--', alpha=0.5)  # 添加网格线
    ax1.legend(loc='upper right', fontsize=11)  # 添加图例
    ax1.spines['top'].set_visible(False)  # 隐藏顶部边框
    ax1.spines['right'].set_visible(False)  # 隐藏右侧边框
    
    # 在数据点上标注准确率数值
    for i, (x, y) in enumerate(zip(repeats, accuracies)):
        ax1.text(x, y + 0.2, f'{y:.2f}%', ha='center', va='bottom', 
                 fontsize=11, fontweight='bold')
    
    # -------------------- 子图2：多指标对比 --------------------
    # 绘制三个指标的趋势线
    ax2.plot(repeats, accuracies, 'o-', linewidth=3, markersize=10, 
             color='#4A9B4A', label='准确率')
    ax2.plot(repeats, balanced_accs, 's-', linewidth=3, markersize=10, 
             color='#2E5AAC', label='平衡准确率')
    ax2.plot(repeats, macro_f1s, '^-', linewidth=3, markersize=10, 
             color='#D4A017', label='Macro-F1')
    
    # 设置图表属性
    ax2.set_xlabel('CV重复次数', fontsize=13, fontweight='bold')
    ax2.set_ylabel('分数 (%)', fontsize=13, fontweight='bold')
    ax2.set_title('多指标对比分析',
                  fontsize=16, fontweight='bold', pad=15)
    ax2.set_xticks(repeats)
    ax2.set_ylim([85, 93])
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # ==================== 4. 添加结论文本框 ====================
    conclusion_text = (
        f"核心发现:\n"
        f"• 准确率最优值出现在 repeat={optimal_repeat} ({optimal_acc:.2f}%)\n"
        f"• 选择配置: repeat=10 (稳定性与性能的最佳平衡)\n"
        f"• Repeat=10: 准确率={accuracies[repeats.index(10)]:.2f}%, "
        f"平衡准确率={balanced_accs[repeats.index(10)]:.2f}%, "
        f"Macro-F1={macro_f1s[repeats.index(10)]:.2f}%"
    )
    
    # 创建文本框样式
    props = dict(boxstyle='round,pad=0.5', facecolor='#FFF8DC', 
                 edgecolor='#DAA520', linewidth=2, alpha=0.9)
    # 添加文本框到图表底部
    fig.text(0.02, 0.02, conclusion_text, transform=fig.transFigure, 
             fontsize=11, verticalalignment='bottom', bbox=props, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # 预留底部空间给文本框
    
    # ==================== 5. 保存图表 ====================
    output_path = OUTPUT_DIR / 'cv_repeat_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', pad_inches=0.3)
    plt.close()
    
    print(f"✓ 图表已保存至: {output_path}")
    
    # ==================== 6. 输出数据摘要 ====================
    print("\n数据摘要:")
    print("=" * 60)
    for r, acc, bal, f1 in zip(repeats, accuracies, balanced_accs, macro_f1s):
        print(f"Repeat={r:2d}: 准确率={acc:.4f}%, 平衡准确率={bal:.4f}%, Macro-F1={f1:.4f}%")
    print(f"\n最优配置: repeat={optimal_repeat} (准确率={optimal_acc:.4f}%)")
    print(f"选择配置: repeat=10 (准确率={accuracies[repeats.index(10)]:.4f}%)")
    
    return output_path


if __name__ == '__main__':
    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 生成分析图表
    generate_cv_repeat_analysis()
