# Cancer-Classification

TCGA-BRCA 多组学分子亚型分类项目。当前默认流程聚焦 RNA-only、Concat、MOFA、Stacking 四条主线，在统一严格交叉验证口径下进行准确率比较。

## 项目目标

在单一癌种 TCGA-BRCA 内进行亚型分类，比较以下策略：

- RNA 单组学基线
- RNA + 甲基化特征拼接（Concat）
- MOFA 潜在因子融合
- Stacking 晚期融合

主结果默认使用严格 repeated CV；消融实验保留在独立脚本中，不作为默认主线入口。

## 目录结构

```text
.
├── config/
│   ├── exp_rna_cv.yaml
│   ├── exp_concat_cv.yaml
│   ├── exp_mofa.yaml
│   └── exp_stacking.yaml
├── datasets/
├── docs/
│   ├── 实验方案.md
│   └── 实验使用指南.md
├── experiments/
│   └── run.py
├── scripts/
│   ├── run_strict_cv.sh
│   ├── run_robust_evaluation.sh
│   └── run_ablation.sh
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── pipeline.py
└── run_all.sh
```

## 环境准备

```bash
pip install -r requirements.txt
```

默认使用仓库内解释器：

- `./medical/bin/python3`

## 快速开始

### 1) 运行主实验（严格 CV 主线）

```bash
bash run_all.sh
```

### 2) 运行严格 CV 主线批处理

```bash
bash scripts/run_strict_cv.sh
```

### 3) 运行稳健评估与统计比较

```bash
bash scripts/run_robust_evaluation.sh
```

### 4) 运行消融实验（可选）

```bash
bash scripts/run_ablation.sh
```

### 5) 单独运行任一配置

```bash
PYTHONPATH=. ./medical/bin/python3 experiments/run.py --config config/exp_rna_cv.yaml
PYTHONPATH=. ./medical/bin/python3 experiments/run.py --config config/exp_concat_cv.yaml
PYTHONPATH=. ./medical/bin/python3 experiments/run.py --config config/exp_mofa.yaml
PYTHONPATH=. ./medical/bin/python3 experiments/run.py --config config/exp_stacking.yaml

PYTHONPATH=. ./medical/bin/python3 experiments/run.py --config config/ablation_no_meth.yaml
PYTHONPATH=. ./medical/bin/python3 experiments/run.py --config config/ablation_no_rna.yaml
PYTHONPATH=. ./medical/bin/python3 experiments/run.py --config config/ablation_no_fs.yaml
```

## 当前评估输出

每次运行会输出：

- `ACC`
- 各类别 `precision / recall / f1-score / support`
- `macro avg` 与 `weighted avg`
- 严格 CV 下还会输出 fold 级汇总、95% CI 和分类报告

## 数据与标签要求

临床标签文件默认读取：`datasets/brca_pam50.csv`

要求：

- TSV 分隔
- 包含 `sample` 列
- 标签列为 `PAM50` 或 `label`
- 标签映射：`LumA=0, LumB=1, HER2=2, Basal=3`

## 防泄露流程说明

当前流程采用：

1. 样本对齐与标签构建
2. 训练/测试切分
3. 在训练集上拟合预处理器（标准化）
4. 用相同预处理器变换测试集

说明：

- `mofa` 分支在严格 CV 下使用训练折拟合、测试折投影的方式估计潜在因子，避免把测试样本混入训练拟合。

## 文档

- 实验方案：[docs/实验方案.md](docs/实验方案.md)
- 实验使用指南：[docs/实验使用指南.md](docs/实验使用指南.md)

## 已验证命令

以下命令已在当前仓库验证通过：

- `bash run_all.sh`
- `bash scripts/run_ablation.sh`

## License

MIT
