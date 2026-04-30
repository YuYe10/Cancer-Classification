# Tasks

## 1. 论文一致性修复与LaTeX编译验证

- [x] Task 1.1: 修复main.tex摘要中CV配置描述（5x3 → 5x10）
  - [x] SubTask 1.1.1: 更新main.tex摘要部分
  - [x] SubTask 1.1.2: 检查sections/04-experiments.tex中CV描述一致性
  - [x] SubTask 1.1.3: 更新sections/04-results-statistical-framework.tex中相关描述

- [x] Task 1.2: 执行LaTeX多次编译验证
  - [x] SubTask 1.2.1: 执行第一次xelatex编译
  - [x] SubTask 1.2.2: 执行第二次xelatex编译确保交叉引用正确
  - [x] SubTask 1.2.3: 检查并修复任何编译错误或警告
  - [x] SubTask 1.2.4: 验证公式编号、图表引用、文献引用正确性

## 2. 实验方案改进与模型优化

- [x] Task 2.1: 分析现有实验设计潜在缺陷
  - [x] SubTask 2.1.1: 检查样本选择标准合理性
  - [x] SubTask 2.1.2: 评估变量控制方法
  - [x] SubTask 2.1.3: 识别统计功效不足问题

- [x] Task 2.2: 优化模型架构与训练参数
  - [x] SubTask 2.2.1: 基于现有实验数据分析最优配置
  - [x] SubTask 2.2.2: 调整Stacking元模型参数
  - [x] SubTask 2.2.3: 验证优化后模型性能提升

## 3. 实验测试执行与性能验证

- [x] Task 3.1: 执行完整实验测试流程
  - [x] SubTask 3.1.1: 运行四条主线配置（RNA-only, Concat, MOFA, Stacking）
  - [x] SubTask 3.1.2: 收集各项性能指标（Accuracy, Balanced Accuracy, Macro-F1, AUC）
  - [x] SubTask 3.1.3: 验证涨点是否达到预设阈值

- [x] Task 3.2: 统计显著性验证
  - [x] SubTask 3.2.1: 执行置换检验确认组间差异
  - [x] SubTask 3.2.2: 计算95%置信区间
  - [x] SubTask 3.2.3: 验证性能提升具有统计稳定性

## 4. 论文逻辑结构与语言润色

- [x] Task 4.1: 逻辑结构系统性梳理
  - [x] SubTask 4.1.1: 检查IMRaD结构完整性
  - [x] SubTask 4.1.2: 验证研究脉络清晰度
  - [x] SubTask 4.1.3: 补充必要的过渡段落

- [x] Task 4.2: 语言表达专业精炼
  - [x] SubTask 4.2.1: 消除冗余表述
  - [x] SubTask 4.2.2: 规范专业术语使用
  - [x] SubTask 4.2.3: 统一学术格式

## 5. 数据图表设计与生成

- [x] Task 5.1: 设计高质量数据图表
  - [x] SubTask 5.1.1: 设计性能对比柱状图
  - [x] SubTask 5.1.2: 生成ROC/PR曲线
  - [x] SubTask 5.1.3: 生成混淆矩阵热力图
  - [x] SubTask 5.1.4: 生成t-SNE降维可视化

- [x] Task 5.2: 图表规范检查
  - [x] SubTask 5.2.1: 确保图表格式符合学术规范
  - [x] SubTask 5.2.2: 验证数据来源标注准确性
  - [x] SubTask 5.2.3: 检查图表与正文关联性

## 6. 实验推进计划优化

- [x] Task 6.1: 细化中期目标
  - [x] SubTask 6.1.1: 明确4周内完成外部验证
  - [x] SubTask 6.1.2: 制定2周内完成HER2类别恢复方案评估
  - [x] SubTask 6.1.3: 设定可量化的评估标准

- [x] Task 6.2: 建立进度检查机制
  - [x] SubTask 6.2.1: 设定定期检查时间节点
  - [x] SubTask 6.2.2: 建立偏差识别与调整策略

## Task Dependencies

- Task 1.2 depends on Task 1.1 (必须先修复内容才能验证编译)
- Task 3.2 depends on Task 3.1 (必须先执行实验才能进行统计验证)
- Task 4.2 depends on Task 4.1 (必须先梳理结构才能精炼语言)
- Task 6.2 depends on Task 6.1 (必须先优化计划才能建立检查机制)