# BRCA分子亚型分类研究 - 系统性改进计划

## 一、现状分析与优秀标准差距

### 优秀标准要求
1. **外部验证完成**
2. **生物学解释充分**
3. **创新性明确**
4. **代码和论文达到可发表水平**

### 当前差距

| 标准 | 当前状态 | 差距 |
|------|----------|------|
| 外部验证 | 未完成 | 缺乏独立队列验证 |
| 生物学解释 | 部分完成（SHAP） | 缺乏通路富集分析、文献证据映射 |
| 创新性 | 已声明 | 需更明确的方法学创新定位 |
| 论文可发表性 | 框架完整 | 需学术润色、图表规范化 |

---

## 二、实验设计优化（短期目标）

### 2.1 增加重采样轮次
**目标**: 将CV从5x3提升至5x10，提升统计功效

**实施步骤**:
1. 创建新配置文件 `config/exp_concat_cv_repeat10.yaml`
2. 修改 `config/exp_rna_cv.yaml` 中的 `repeats: 10`
3. 修改 `config/exp_mofa.yaml` 和 `config/exp_stacking_sota_cv.yaml`
4. 更新 `scripts/run_strict_cv.sh` 调用新配置
5. 重新运行实验并记录结果

### 2.2 完善统计分析
**目标**: 补充Balanced Accuracy、Macro-F1的成对置换检验

**实施步骤**:
1. 修改 `scripts/statistical_evaluation.py`，添加多指标置换检验
2. 在 `src/models/evaluate.py` 中添加配对统计量计算
3. 生成新的统计比较表格
4. 更新论文 `04-results-statistical-framework.tex` 中的表格

### 2.3 优化Stacking超参数
**目标**: Grid search base_model_type、meta_model_type、meta_cv_splits

**实施步骤**:
1. 创建参数搜索空间配置
2. 实现参数搜索脚本 `scripts/run_stacking_grid_search.py`
3. 运行超参数搜索并记录最优配置
4. 用最优配置重新评估

---

## 三、数据处理与统计分析完善

### 3.1 完善类别级误差分析
**目标**: 生成混淆矩阵、错误样本列表、关键特征贡献

**实施步骤**:
1. 修改 `scripts/generate_class_error_analysis.py`
2. 添加错误样本ID导出功能
3. 为每个类别计算precision/recall/F1
4. 更新论文 `13-error-analysis.tex` 章节

### 3.2 补充可视化
**目标**: 添加ROC曲线、PR曲线、t-SNE降维可视化

**实施步骤**:
1. 在 `src/visualization/` 中添加:
   - `roc_curves.py` - ROC曲线绘制
   - `pr_curves.py` - PR曲线绘制
   - `tsne_visualization.py` - t-SNE降维可视化
2. 更新 `scripts/generate_paper_artifacts.py` 集成新图表
3. 更新论文 `09-results-visualization.tex` 添加新图表

---

## 四、学术论文全面润色

### 4.1 IMRaD结构完善
**检查并补强**:
- [ ] 摘要 - 已有，需优化语言表达
- [ ] 引言 (`01-background.tex`) - 已有，需补充研究贡献声明
- [ ] 方法 (`03-method.tex`) - 已有，需补充数学公式详解
- [ ] 实验 (`04-experiments.tex`) - 已有，需补充消融实验细节
- [ ] 结果 (`04-results-statistical-framework.tex`, `09-results-visualization.tex`) - 已有，需补充外部验证说明占位
- [ ] 讨论 (`10-discussion.tex`) - 已有，需扩展局限性讨论
- [ ] 结论 (`05-conclusion.tex`) - 已有，需补充具体贡献

### 4.2 语言表达优化
**实施步骤**:
1. 检查所有章节的语法和表达
2. 统一专业术语使用（如"分子亚型"vs"亚型分类"）
3. 消除中式英语表达
4. 规范化数量词和单位使用

### 4.3 图表规范化
**实施步骤**:
1. 检查所有图表的标题、标签、图例
2. 确保图表自明性（不依赖正文也能理解）
3. 统一图表格式（字体、字号、颜色方案）
4. 添加数据来源注释

### 4.4 参考文献整理
**实施步骤**:
1. 检查 `11-references.tex` 的引用格式
2. 确保所有引用在正文中被使用
3. 补充必要的背景文献
4. 统一引用格式（Nature/Science风格）

---

## 五、外部验证准备（中期目标）

### 5.1 外部数据获取
**目标**: 获取METABRIC或TCGA其他BRCA队列

**实施步骤**:
1. 从UCSC Xena下载METABRIC BRCA数据
2. 或使用TCGA不同平台数据（RNA-seq vs Microarray）
3. 进行数据对齐和预处理

### 5.2 跨平台验证实验
**目标**: 在外部数据上验证最佳方法

**实施步骤**:
1. 使用已验证的Concat方法（当前最佳）
2. 在外部数据上进行预测
3. 报告外部验证性能

---

## 六、生物学解释强化

### 6.1 通路富集分析
**目标**: KEGG、GO富集分析关键特征

**实施步骤**:
1. 识别各方法的关键特征（Top 50）
2. 使用enrichr或gseapy进行富集分析
3. 生成通路富集图
4. 在讨论中解释生物学意义

### 6.2 文献证据映射
**目标**: 建立关键基因/位点与亚型关系的文献证据

**实施步骤**:
1. 整理SHAP分析中每个类别的Top特征
2. 检索这些基因/位点与BRCA亚型的文献关联
3. 在讨论中添加"关键特征-生物学机制"映射表

---

## 七、代码与可复现性提升

### 7.1 依赖包梳理
**当前requirements.txt内容**:
```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
umap-learn
shap
mofapy2
pyyaml
```

**需补充的包**:
- `seaborn` - 已列出但需确认版本
- `gseapy` 或 `enrichr` - 通路富集分析
- `scipy` - 统计检验
- `statsmodels` - 高级统计分析

### 7.2 代码规范化
**实施步骤**:
1. 添加docstring到所有公共函数
2. 添加类型注解（可选）
3. 统一代码风格
4. 更新README说明

---

## 八、文件组织结构

### 8.1 研究报告 (`docs/研究报告/`)
```
sections/
├── 01-background.tex          # 背景与动机
├── 02-data.tex                # 数据来源与预处理
├── 03-method.tex              # 方法
├── 04-experiments.tex         # 实验设置
├── 04-results-statistical-framework.tex  # 统计框架结果
├── 05-conclusion.tex          # 结论
├── 06-summary.tex             # 摘要（备用）
├── 07-related-work.tex        # 相关工作
├── 08-method-math.tex         # 方法数学细节
├── 09-results-visualization.tex  # 结果可视化
├── 10-discussion.tex          # 讨论
├── 11-references.tex          # 参考文献
├── 12-implementation-details.tex  # 实现细节
├── 13-error-analysis.tex      # 错误分析
├── 14-appendix-config.tex     # 附录：配置详情
├── 15-results-detailed.tex    # 附录：详细结果
└── 16-related-work-extended.tex  # 附录：扩展相关工作
```

### 8.2 源代码 (`src/`)
```
src/
├── data/
│   ├── align.py
│   ├── loader.py
│   └── preprocess.py
├── features/
│   └── mofa.py
├── models/
│   ├── evaluate.py
│   └── train.py
├── utils/
│   ├── logger.py
│   └── seed.py
├── visualization/
│   ├── advanced_visualizations.py
│   ├── feature_distribution.py
│   ├── fusion_methods.py
│   ├── generate_report_figures.py
│   ├── model_performance.py
│   ├── pr_curves.py          # 新增
│   ├── roc_curves.py         # 新增
│   └── tsne.py
├── explain/
│   └── shap_explain.py
└── pipeline.py
```

### 8.3 脚本文件 (`scripts/`)
```
scripts/
├── run_ablation.sh
├── run_strict_cv.sh
├── run_robust_evaluation.sh
├── run_stacking_sota.sh
├── generate_paper_artifacts.py
├── generate_statistical_plots.py
├── generate_class_error_analysis.py
├── statistical_evaluation.py
├── summarize_results.py
├── validate_paper_numbers.py
├── roc_pr_analysis.py         # 新增
└── pathway_enrichment.py      # 新增
```

---

## 九、实施时间线

### 第一周（短期优化）
- [ ] 增加CV重采样到5x10
- [ ] 完善多指标置换检验
- [ ] 补充ROC/PR曲线
- [ ] 完善类别级误差分析

### 第二周（论文润色）
- [ ] IMRaD结构检查与完善
- [ ] 语言表达优化
- [ ] 图表规范化
- [ ] 参考文献整理

### 第三-四周（中期提升）
- [ ] 外部验证数据获取
- [ ] 跨平台验证实验
- [ ] 通路富集分析
- [ ] 文献证据映射

---

## 十、预期成果

### 优秀标准达成检查清单
- [ ] 外部验证完成（METABRIC或其他BRCA队列）
- [ ] 生物学解释充分（通路富集+文献映射）
- [ ] 创新性明确（研究设计创新定位清晰）
- [ ] 代码可复现（完整注释+依赖说明）
- [ ] 论文可发表（结构完整+语言规范+图表专业）

---

## 十一、风险与应对

### 风险1: 外部数据获取困难
**应对**: 
- 优先使用TCGA-BRCA内部时间切分验证
- 或使用已有数据增强技术

### 风险2: 通路富集分析结果不显著
**应对**:
- 放宽富集阈值
- 使用自定义基因集

### 风险3: 时间不足
**应对**:
- 优先完成短期目标
- 并行推进多项任务