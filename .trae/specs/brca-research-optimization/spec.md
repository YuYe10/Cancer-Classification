# BRCA分子亚型分类研究系统性改进 Spec

## Why

当前BRCA分子亚型分类研究已完成短期目标（5x10 CV框架、四种融合方法比较、统计评估体系），但在以下方面存在改进空间：
1. 论文摘要与正文存在不一致（CV配置描述为5x3但实际为5x10）
2. 外部验证集尚未开展，影响研究可信度
3. HER2类别缺失导致任务退化为三分类
4. 生物学机制解释深度不足（通路富集分析待强化）
5. LaTeX文档需多次编译验证确保无错误

## What Changes

- **论文一致性修复**：修正摘要中CV配置描述（5 folds × 3 repeats → 5 folds × 10 repeats）
- **外部验证设计**：引入METABRIC队列或TCGA不同平台数据进行跨数据集验证
- **HER2类别恢复方案**：探索数据增强或多源数据合并策略
- **通路富集深度分析**：使用gseapy对关键特征进行KEGG/GO富集分析并映射生物学机制
- **LaTeX完整编译验证**：多次编译检查确保PDF输出正确、引用无误
- **实验推进计划更新**：细化中期目标的时间节点和可量化评估标准

## Impact

- Affected specs: 研究报告完整性、实验可复现性、生物学解释深度
- Affected code: scripts/pathway_enrichment.py, docs/研究报告/main.tex, docs/实验推进计划.md

## ADDED Requirements

### Requirement: 论文一致性修复
系统应当修复摘要中CV配置描述与实际配置的不一致，确保"5 folds × 10 repeats"在全文中保持统一

#### Scenario: 论文一致性验证
- **WHEN** 用户执行LaTeX编译检查
- **THEN** 摘要、方法和实验部分对CV配置的描述必须完全一致（5x10）

### Requirement: 外部验证流程设计
系统应当提供从UCSC Xena或类似来源获取外部BRCA队列的标准化流程，支持跨平台验证

#### Scenario: 外部验证执行
- **WHEN** 研究者获取METABRIC或其他BRCA外部队列数据
- **THEN** 系统能够完成跨数据集的特征对齐、模型推断和性能评估

### Requirement: 生物学机制深度解释
系统应当对关键分类特征进行通路富集分析，生成KEGG/GO富集结果并与文献证据映射

#### Scenario: 富集分析执行
- **WHEN** 用户运行通路富集分析脚本
- **THEN** 生成包含富集通路、p值、相关基因的完整报告

## MODIFIED Requirements

### Requirement: 实验推进计划
**修改原因**：原计划中期目标描述过于笼统，需要更具体的时间节点和可量化标准
**迁移**：细化为4周内完成外部验证、2周内完成HER2类别恢复方案评估

## REMOVED Requirements

### Requirement: 旧版CV配置描述
**原因**：摘要中的"5 folds × 3 repeats"描述已过时，与实际5x10配置不符
**迁移**：全文统一更新为"5 folds × 10 repeats"