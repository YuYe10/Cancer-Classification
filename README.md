# Cancer-Classification

TCGA-BRCA 多组学分子亚型分类项目。当前实现支持主实验对比、消融实验批量运行，以及配置驱动的可复现实验流程。

## 项目目标

在单一癌种 TCGA-BRCA 内进行亚型分类，比较以下策略：

- RNA 单组学基线
- RNA + 甲基化特征拼接（Concat）
- MOFA 潜在因子融合

并补充三类消融实验：

- 去甲基化（仅 RNA）
- 去 RNA（仅甲基化）
- 不做特征选择

## 目录结构

```text
.
├── config/
│   ├── exp_rna.yaml
│   ├── exp_concat.yaml
│   ├── exp_mofa.yaml
│   ├── ablation_no_meth.yaml
│   ├── ablation_no_rna.yaml
│   └── ablation_no_fs.yaml
├── datasets/
├── docs/
│   ├── 实验方案.md
│   └── 实验使用指南.md
├── experiments/
│   └── run.py
├── scripts/
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

### 1) 运行主实验（RNA / Concat / MOFA）

```bash
bash run_all.sh
```

### 2) 运行消融实验（3 组）

```bash
bash scripts/run_ablation.sh
```

### 3) 单独运行任一配置

```bash
PYTHONPATH=. ./medical/bin/python3 experiments/run.py --config config/exp_rna.yaml
PYTHONPATH=. ./medical/bin/python3 experiments/run.py --config config/exp_concat.yaml
PYTHONPATH=. ./medical/bin/python3 experiments/run.py --config config/exp_mofa.yaml

PYTHONPATH=. ./medical/bin/python3 experiments/run.py --config config/ablation_no_meth.yaml
PYTHONPATH=. ./medical/bin/python3 experiments/run.py --config config/ablation_no_rna.yaml
PYTHONPATH=. ./medical/bin/python3 experiments/run.py --config config/ablation_no_fs.yaml
```

## 当前评估输出

每次运行会输出：

- `ACC`
- 各类别 `precision / recall / f1-score / support`
- `macro avg` 与 `weighted avg`

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

- 由于当前 MOFA 接口仅提供拟合+抽取潜在因子，`mofa` 分支采用“合并样本拟合一次后再切分潜在因子”的实现以保证可运行。

## 文档

- 实验方案：[docs/实验方案.md](docs/实验方案.md)
- 实验使用指南：[docs/实验使用指南.md](docs/实验使用指南.md)

## 已验证命令

以下命令已在当前仓库验证通过：

- `bash run_all.sh`
- `bash scripts/run_ablation.sh`

## License

MIT
