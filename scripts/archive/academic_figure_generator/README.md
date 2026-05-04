# Academic Figure Generator

A comprehensive Python toolkit for generating publication-quality figures for research papers and academic documents.

## Features

- **Multi-format Data Reading**: Supports CSV and Excel files with built-in data cleaning, missing value handling, and statistical analysis
- **Publication-Quality Charts**: Line plots, bar charts, scatter plots, heatmaps, and box plots with full customization
- **Multiple Output Formats**: Export figures as PDF, SVG, PNG, and other formats
- **Document Integration**: Seamless integration with LaTeX and Microsoft Word documents
- **Modular Design**: Clean separation of concerns with independent modules for data reading, chart generation, output management, and document integration
- **Academic Standards**: All charts follow academic publication conventions including proper labeling, error bars, and significance markers

## Installation

### Requirements

- Python 3.8 or higher
- Required packages (install via pip):

```bash
pip install pandas numpy matplotlib scipy python-docx
```

### Quick Start

```python
from academic_figure_generator import DataReader, ChartGenerator, OutputManager

# Load data
reader = DataReader()
reader.read_csv('experiment_results.csv')

# Generate chart
generator = ChartGenerator()
fig = generator.bar_plot(
    x=[1, 2, 3, 4],
    y=[0.85, 0.89, 0.78, 0.84],
    labels=['Methods'],
    categories=['RNA', 'Concat', 'MOFA', 'Stacking'],
    title='Model Performance Comparison',
    xlabel='Method',
    ylabel='Accuracy'
)

# Save figure
output_manager = OutputManager()
output_manager.save_figure(fig, 'performance_comparison', formats=['pdf', 'svg'])
```

## Module Overview

### DataReader

Reads and processes experimental data from CSV and Excel files.

```python
from academic_figure_generator import DataReader

reader = DataReader()

# Read CSV file
reader.read_csv('data.csv')

# Handle missing values
reader.handle_missing(strategy='mean')

# Remove outliers
reader.remove_outliers(method='iqr', threshold=1.5)

# Get statistics
stats = reader.get_statistics()
```

### ChartGenerator

Creates publication-quality charts with academic formatting.

```python
from academic_figure_generator import ChartGenerator, ChartStyle

# Custom style
style = ChartStyle(
    figsize=(8, 6),
    dpi=300,
    font_family='serif',
    color_scheme='tab10'
)

generator = ChartGenerator(style=style)

# Line plot
fig = generator.line_plot(
    x=[1, 2, 3, 4],
    y=[[10, 20, 30, 40], [15, 25, 35, 45]],
    labels=['Control', 'Treatment'],
    title='Experimental Results',
    xlabel='Time (hours)',
    ylabel='Concentration'
)

# Bar chart
fig = generator.bar_plot(
    x=[0, 1, 2, 3],
    y=[0.85, 0.89, 0.78, 0.84],
    categories=['RNA', 'Concat', 'MOFA', 'Stacking'],
    title='Model Performance',
    ylabel='Accuracy',
    errors=[0.02, 0.03, 0.04, 0.02],
    bar_labels=True
)

# Scatter plot with regression
fig = generator.scatter_plot(
    x=[1, 2, 3, 4, 5],
    y=[2, 4, 3, 5, 6],
    show_regression=True,
    regression_line=True
)

# Heatmap
fig = generator.heatmap(
    data=confusion_matrix,
    title='Confusion Matrix',
    cmap='Blues',
    annot=True
)
```

### OutputManager

Manages output operations and figure organization.

```python
from academic_figure_generator import OutputManager, OutputConfig

config = OutputConfig(
    output_dir='outputs',
    formats=['pdf', 'svg'],
    organization='type'
)

manager = OutputManager(config)

# Save figure in multiple formats
paths = manager.save_figure(
    fig,
    'performance_comparison',
    fig_type='bar',
    formats=['pdf', 'svg']
)

# Create manifest
manager.create_manifest()
```

### DocumentIntegrator

Integrates figures into LaTeX or Word documents.

```python
from academic_figure_generator import DocumentIntegrator

# For LaTeX
integrator = DocumentIntegrator('paper.tex')
integrator.add_figure(
    figure_id='fig1',
    caption='Experimental results comparison',
    path='outputs/figures/bar/performance.pdf',
    label='fig:results',
    width=0.9
)
integrator.update()

# For Word
integrator = DocumentIntegrator('paper.docx')
integrator.add_figure(
    caption='Results',
    path='outputs/figures/bar/performance.png',
    width_inches=6.0
)
integrator.update()
```

## Configuration

The package can be configured via `config.ini`:

```ini
[chart_style]
figsize_width = 8.0
figsize_height = 6.0
dpi = 300
font_family = "serif"
color_scheme = "tab10"

[output]
output_dir = "outputs"
formats = "pdf,svg"
organization = "type"
```

## Output Structure

```
outputs/
├── figures/
│   ├── bar/
│   │   └── performance_comparison.pdf
│   ├── line/
│   │   └── trend_analysis.pdf
│   ├── scatter/
│   │   └── correlation.pdf
│   ├── heatmap/
│   │   └── confusion_matrix.pdf
│   └── other/
│       └── multipanel.pdf
├── data/
│   ├── sample_data.csv
│   └── figure_snippet.tex
└── figures_manifest.json
```

## Chart Types

| Type | Method | Use Case |
|------|--------|----------|
| Line Plot | `line_plot()` | Trend analysis, time series |
| Bar Chart | `bar_plot()` | Comparison between groups |
| Scatter Plot | `scatter_plot()` | Correlation analysis |
| Heatmap | `heatmap()` | Matrix visualization |
| Box Plot | `box_plot()` | Distribution comparison |

## Academic Features

- **Error bars**: Standard deviation, standard error, confidence intervals
- **Significance markers**: *, **, ***, **** for p-values
- **Statistical annotations**: Regression lines, confidence bands
- **Academic formatting**: Proper axis labels, legends, gridlines

## File Structure

```
scripts/academic_figure_generator/
├── __init__.py                  # Package initialization
├── data_reader.py               # Data reading and processing
├── chart_generator.py            # Chart generation
├── output_manager.py            # Output management
├── document_integrator.py       # Document integration
├── config.ini                   # Configuration file
├── generate_example_figures.py  # Example usage script
└── README.md                   # This file
```

## License

MIT License

## Author

Cancer-Classification Project