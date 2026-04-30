"""
Chart Generator Module
====================

Provides comprehensive chart generation capabilities for academic publications,
supporting line plots, bar charts, scatter plots, and heatmaps with
full customization for publication standards.

Author: Cancer-Classification Project
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple, Any, Callable
from dataclasses import dataclass, field
import warnings


@dataclass
class ChartStyle:
    """
    Configuration class for chart styling parameters.

    Attributes:
        figsize: Figure size as (width, height) tuple in inches.
        dpi: Resolution in dots per inch.
        font_family: Font family name.
        font_size: Base font size.
        title_font_size: Title font size.
        label_font_size: Axis label font size.
        legend_font_size: Legend font size.
        line_width: Default line width.
        marker_size: Default marker size.
        color_scheme: Color palette name or list of colors.
        grid_alpha: Grid line transparency (0-1).
        spine_line_width: Width of axis spines.
        tick_length: Length of tick marks.
        tick_width: Width of tick marks.
    """
    figsize: Tuple[float, float] = (8, 6)
    dpi: int = 300
    font_family: str = 'serif'
    font_size: float = 12.0
    title_font_size: float = 14.0
    label_font_size: float = 12.0
    legend_font_size: float = 10.0
    line_width: float = 2.0
    marker_size: float = 8.0
    color_scheme: Union[str, List[str]] = 'tab10'
    grid_alpha: float = 0.3
    spine_line_width: float = 1.0
    tick_length: int = 5
    tick_width: float = 1.0


@dataclass
class ErrorBarConfig:
    """
    Configuration for error bars.

    Attributes:
        show: Whether to show error bars.
        method: Method for calculating errors ('std', 'sem', 'ci', 'raw').
        confidence_level: Confidence level for CI (default: 0.95).
        cap_size: Size of error bar caps.
        cap_thickness: Thickness of error bar caps.
    """
    show: bool = True
    method: str = 'std'
    confidence_level: float = 0.95
    cap_size: float = 5.0
    cap_thickness: float = 1.5


class ChartGenerator:
    """
    A comprehensive chart generator for academic publications.

    This class provides methods for creating publication-quality charts
    including line plots, bar charts, scatter plots, and heatmaps.
    All charts support customizable styling, error bars, and significance markers.

    Attributes:
        style (ChartStyle): Current style configuration.
        error_config (ErrorBarConfig): Current error bar configuration.

    Example:
        >>> generator = ChartGenerator()
        >>> fig = generator.line_plot(
        ...     x=[1, 2, 3, 4],
        ...     y=[[10, 20, 30, 40], [15, 25, 35, 45]],
        ...     labels=['Control', 'Treatment'],
        ...     title='Experimental Results'
        ... )
        >>> generator.save('figure1', format='pdf')
    """

    DEFAULT_COLORS = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    SIGNIFICANCE_MARKERS = {
        'ns': '',
        '*': '*',
        '**': '**',
        '***': '***',
        '****': '****'
    }

    def __init__(self, style: Optional[ChartStyle] = None):
        """
        Initialize the ChartGenerator.

        Args:
            style: Custom style configuration. Uses default if None.
        """
        self.style = style or ChartStyle()
        self.error_config = ErrorBarConfig()
        self._figure: Optional[plt.Figure] = None
        self._axes: Optional[plt.Axes] = None
        self._setup_matplotlib()

    def _setup_matplotlib(self) -> None:
        """Configure matplotlib for academic publication standards."""
        plt.rcParams.update({
            'font.family': self.style.font_family,
            'font.size': self.style.font_size,
            'axes.labelsize': self.style.label_font_size,
            'axes.titlesize': self.style.title_font_size,
            'axes.linewidth': self.style.spine_line_width,
            'xtick.major.size': self.style.tick_length,
            'xtick.major.width': self.style.tick_width,
            'ytick.major.size': self.style.tick_length,
            'ytick.major.width': self.style.tick_width,
            'legend.fontsize': self.style.legend_font_size,
            'figure.dpi': self.style.dpi,
            'savefig.dpi': self.style.dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
        })

    def _get_colors(self, n: int) -> List[str]:
        """
        Get n colors from the color scheme.

        Args:
            n: Number of colors needed.

        Returns:
            List of color hex strings.
        """
        if isinstance(self.style.color_scheme, list):
            colors = self.style.color_scheme * (n // len(self.style.color_scheme) + 1)
            return colors[:n]
        elif self.style.color_scheme == 'tab10':
            return plt.cm.tab10(np.linspace(0, 1, 10))[:n]
        elif self.style.color_scheme == 'tab20':
            return plt.cm.tab20(np.linspace(0, 1, 20))[:n]
        elif self.style.color_scheme == 'Set1':
            return plt.cm.Set1(np.linspace(0, 1, 9))[:n]
        elif self.style.color_scheme == 'Set2':
            return plt.cm.Set2(np.linspace(0, 1, 8))[:n]
        elif self.style.color_scheme == 'paired':
            return plt.cm.Paired(np.linspace(0, 1, 12))[:n]
        else:
            return self.DEFAULT_COLORS[:n]

    def _calculate_errors(self,
                        data: Union[List, np.ndarray],
                        method: str = 'std',
                        confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate error values for data.

        Args:
            data: Array-like data.
            method: Error calculation method ('std', 'sem', 'ci').
            confidence_level: Confidence level for CI.

        Returns:
            Tuple of (lower_errors, upper_errors).
        """
        data = np.array(data)
        n = len(data)

        if method == 'std':
            errors = np.std(data, ddof=1)
            return errors * np.ones(n), errors * np.ones(n)
        elif method == 'sem':
            errors = np.std(data, ddof=1) / np.sqrt(n)
            return errors * np.ones(n), errors * np.ones(n)
        elif method == 'ci':
            mean = np.mean(data, axis=0)
            sem = np.std(data, ddof=1, axis=0) / np.sqrt(n)
            from scipy import stats
            t_val = stats.t.ppf((1 + confidence_level) / 2, df=n-1)
            ci = t_val * sem
            return ci, ci
        else:
            return np.zeros(n), np.zeros(n)

    def _apply_academic_style(self,
                             ax: plt.Axes,
                             title: Optional[str] = None,
                             xlabel: Optional[str] = None,
                             ylabel: Optional[str] = None,
                             legend_loc: Optional[str] = 'best',
                             show_grid: bool = True) -> None:
        """
        Apply academic publication style to a chart.

        Args:
            ax: Matplotlib axes object.
            title: Chart title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            legend_loc: Legend location.
            show_grid: Whether to show grid.
        """
        if title:
            ax.set_title(title, fontweight='bold', pad=15)
        if xlabel:
            ax.set_xlabel(xlabel, fontweight='bold', labelpad=10)
        if ylabel:
            ax.set_ylabel(ylabel, fontweight='bold', labelpad=10)

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(self.style.spine_line_width)
        ax.spines['left'].set_linewidth(self.style.spine_line_width)
        ax.spines['top'].set_linewidth(self.style.spine_line_width)

        if show_grid:
            ax.grid(True, alpha=self.style.grid_alpha, linestyle='--', linewidth=0.5)
            ax.set_axisbelow(True)

        ax.tick_params(direction='in', length=self.style.tick_length,
                      width=self.style.tick_width)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    def line_plot(self,
                  x: Union[List, np.ndarray],
                  y: Union[List, np.ndarray, List[List]],
                  labels: Optional[List[str]] = None,
                  title: Optional[str] = None,
                  xlabel: Optional[str] = None,
                  ylabel: Optional[str] = None,
                  errors: Optional[Union[List, np.ndarray, List[List]]] = None,
                  markers: Optional[List[str]] = None,
                  linestyles: Optional[List[str]] = None,
                  legend_loc: str = 'best',
                  show_grid: bool = True,
                  xlim: Optional[Tuple[float, float]] = None,
                  ylim: Optional[Tuple[float, float]] = None,
                  log_scale: Optional[str] = None,
                  secondary_y: Optional[Dict[str, int]] = None) -> plt.Figure:
        """
        Create a line plot for trend analysis.

        Args:
            x: X-axis data (single array or list of arrays for multiple lines).
            y: Y-axis data (array or list of arrays for multiple lines).
            labels: Labels for each line (used in legend).
            title: Chart title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            errors: Error values for error bars (same shape as y).
            markers: Marker styles for each line.
            linestyles: Line styles for each line.
            legend_loc: Legend location.
            show_grid: Whether to show grid.
            xlim: X-axis limits as (min, max).
            ylim: Y-axis limits as (min, max).
            log_scale: Use log scale ('x', 'y', or 'both').
            secondary_y: Dict mapping line indices to secondary y-axis.

        Returns:
            Matplotlib figure object.

        Example:
            >>> fig = generator.line_plot(
            ...     x=[1, 2, 3, 4],
            ...     y=[[10, 20, 30, 40], [15, 25, 35, 45]],
            ...     labels=['Group A', 'Group B'],
            ...     title='Trend Analysis',
            ...     errors=[np.array([1, 1, 1, 1]), np.array([2, 2, 2, 2])]
            ... )
        """
        self._figure, self._axes = plt.subplots(figsize=self.style.figsize)

        if isinstance(x[0], (list, np.ndarray)) and len(x) > 1:
            x_arrays = x
        else:
            x_arrays = [np.array(x)]

        if isinstance(y[0], (list, np.ndarray)) and (len(y) > 1 or
            (isinstance(y[0], (list, np.ndarray)) and len(y[0]) > 1 and
             isinstance(y[0][0], (int, float)) and not isinstance(y[0][0], (list, np.ndarray)))):
            y_arrays = [np.array(yi) for yi in y]
        else:
            y_arrays = [np.array(y)]

        n_lines = len(y_arrays)
        colors = self._get_colors(n_lines)
        markers = markers or ['o'] * n_lines
        linestyles = linestyles or ['-'] * n_lines

        if errors is not None:
            if isinstance(errors[0], (list, np.ndarray)):
                error_arrays = [np.array(ei) for ei in errors]
            else:
                error_arrays = [np.array(errors)] * n_lines
        else:
            error_arrays = [None] * n_lines

        ax2 = None
        for i, (xi, yi, color, marker, ls) in enumerate(zip(x_arrays, y_arrays, colors, markers, linestyles)):
            if secondary_y and i in secondary_y.values():
                if ax2 is None:
                    ax2 = self._axes.twinx()
                ax = ax2
            else:
                ax = self._axes

            line, = ax.plot(xi, yi, color=color, marker=marker, markersize=self.style.marker_size,
                           linewidth=self.style.line_width, linestyle=ls, label=labels[i] if labels else None)

            if error_arrays[i] is not None and self.error_config.show:
                ax.errorbar(xi, yi, yerr=error_arrays[i], color=color,
                          capsize=self.error_config.cap_size,
                          capthick=self.error_config.cap_thickness,
                          elinewidth=1, alpha=0.7)

        self._apply_academic_style(self._axes, title, xlabel, ylabel, legend_loc, show_grid)

        if labels:
            self._axes.legend(loc=legend_loc, frameon=True, fancybox=False,
                             edgecolor='black', framealpha=0.9)

        if xlim:
            self._axes.set_xlim(xlim)
        if ylim:
            self._axes.set_ylim(ylim)

        if log_scale:
            if log_scale in ['x', 'both']:
                self._axes.set_xscale('log')
            if log_scale in ['y', 'both']:
                self._axes.set_yscale('log')

        self._figure.tight_layout()
        return self._figure

    def bar_plot(self,
                 x: Union[List, np.ndarray],
                 y: Union[List, np.ndarray, List[List]],
                 labels: Optional[List[str]] = None,
                 categories: Optional[List[str]] = None,
                 title: Optional[str] = None,
                 xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None,
                 errors: Optional[Union[List, np.ndarray]] = None,
                 colors: Optional[List[str]] = None,
                 legend_loc: str = 'best',
                 show_grid: bool = True,
                 horizontal: bool = False,
                 stacked: bool = False,
                 width: float = 0.8,
                 bar_labels: bool = False,
                 ylim: Optional[Tuple[float, float]] = None) -> plt.Figure:
        """
        Create a bar chart for comparison analysis.

        Args:
            x: X-axis categories or positions.
            y: Y-axis data (array or list of arrays for grouped bars).
            labels: Labels for grouped bars (used in legend).
            categories: Category labels for x-axis.
            title: Chart title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            errors: Error values for error bars.
            colors: Custom colors for bars.
            legend_loc: Legend location.
            show_grid: Whether to show grid.
            horizontal: Whether to create horizontal bars.
            stacked: Whether to stack bars.
            width: Bar width as fraction of spacing.
            bar_labels: Whether to show value labels on bars.
            ylim: Y-axis limits.

        Returns:
            Matplotlib figure object.
        """
        self._figure, self._axes = plt.subplots(figsize=self.style.figsize)

        x = np.array(x)
        y = np.array(y) if not isinstance(y[0], (list, np.ndarray)) else y

        if not isinstance(y[0], (list, np.ndarray)):
            y = [y]

        n_groups = len(x)
        n_bars = len(y)
        colors = colors or self._get_colors(n_bars)

        if horizontal:
            if stacked:
                bottom = np.zeros(n_groups)
                for i, (yi, color) in enumerate(zip(y, colors)):
                    yi = np.array(yi)
                    bars = self._axes.barh(x, yi, width * 0.9, left=bottom,
                                          color=color, label=labels[i] if labels else None,
                                          edgecolor='black', linewidth=0.5)
                    if bar_labels:
                        self._axes.bar_label(bars, fmt='%.2f', label_type='center')
                    bottom += yi
            else:
                bar_width = width / n_bars
                for i, (yi, color) in enumerate(zip(y, colors)):
                    offset = (i - n_bars / 2 + 0.5) * bar_width
                    bars = self._axes.barh(x + offset, yi, bar_width * 0.9,
                                          color=color, label=labels[i] if labels else None,
                                          edgecolor='black', linewidth=0.5)
                    if bar_labels and errors is not None:
                        self._axes.bar_label(bars, fmt='%.2f', label_type='edge')
        else:
            if stacked:
                bottom = np.zeros(n_groups)
                for i, (yi, color) in enumerate(zip(y, colors)):
                    yi = np.array(yi)
                    bars = self._axes.bar(x, yi, width * 0.9, bottom=bottom,
                                         color=color, label=labels[i] if labels else None,
                                         edgecolor='black', linewidth=0.5)
                    if bar_labels:
                        self._axes.bar_label(bars, fmt='%.2f', label_type='center')
                    bottom += yi
            else:
                bar_width = width / n_bars
                for i, (yi, color) in enumerate(zip(y, colors)):
                    offset = (i - n_bars / 2 + 0.5) * bar_width
                    bars = self._axes.bar(x + offset, yi, bar_width * 0.9,
                                          color=color, label=labels[i] if labels else None,
                                          edgecolor='black', linewidth=0.5)
                    if bar_labels and errors is not None:
                        self._axes.bar_label(bars, fmt='%.2f', label_type='edge')

        if categories:
            if horizontal:
                self._axes.set_yticks(x)
                self._axes.set_yticklabels(categories)
            else:
                self._axes.set_xticks(x)
                self._axes.set_xticklabels(categories, rotation=45, ha='right')

        self._apply_academic_style(self._axes, title, xlabel, ylabel, legend_loc, show_grid)

        if labels:
            self._axes.legend(loc=legend_loc, frameon=True, fancybox=False,
                             edgecolor='black', framealpha=0.9)

        if ylim:
            self._axes.set_ylim(ylim)

        if errors is not None and not stacked and not bar_labels:
            errors = np.array(errors)
            if horizontal:
                self._axes.errorbar(x + (n_bars - 1) * width / (2 * n_bars),
                                    y[-1] if stacked else y[0], yerr=errors,
                                    fmt='none', color='black', capsize=5)
            else:
                self._axes.errorbar(x + (n_bars - 1) * width / (2 * n_bars),
                                    y[-1] if stacked else y[0], yerr=errors,
                                    fmt='none', color='black', capsize=5)

        self._figure.tight_layout()
        return self._figure

    def scatter_plot(self,
                     x: Union[List, np.ndarray],
                     y: Union[List, np.ndarray],
                     labels: Optional[List[str]] = None,
                     categories: Optional[List[str]] = None,
                     title: Optional[str] = None,
                     xlabel: Optional[str] = None,
                     ylabel: Optional[str] = None,
                     colors: Optional[Union[str, List[str]]] = None,
                     sizes: Optional[Union[float, List[float]]] = None,
                     alpha: float = 0.6,
                     show_grid: bool = True,
                     show_regression: bool = False,
                     regression_line: bool = True,
                     regression_confidence: float = 0.95,
                     xlim: Optional[Tuple[float, float]] = None,
                     ylim: Optional[Tuple[float, float]] = None,
                     colorbar: bool = False,
                     marker: str = 'o') -> plt.Figure:
        """
        Create a scatter plot for correlation analysis.

        Args:
            x: X-axis data.
            y: Y-axis data.
            labels: Point labels for hover/tooltip.
            categories: Category for each point (for coloring).
            title: Chart title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            colors: Color for each point or colormap name.
            sizes: Size for each point.
            alpha: Point transparency.
            show_grid: Whether to show grid.
            show_regression: Whether to show regression statistics.
            regression_line: Whether to show regression line.
            regression_confidence: Confidence level for regression.
            xlim: X-axis limits.
            ylim: Y-axis limits.
            colorbar: Whether to show colorbar.
            marker: Marker style.

        Returns:
            Matplotlib figure object.
        """
        self._figure, self._axes = plt.subplots(figsize=self.style.figsize)

        x = np.array(x)
        y = np.array(y)

        if categories is not None:
            categories = np.array(categories)
            unique_cats = np.unique(categories)
            colors = colors or self._get_colors(len(unique_cats))
            cat_color_map = dict(zip(unique_cats, colors))

            for cat in unique_cats:
                mask = categories == cat
                self._axes.scatter(x[mask], y[mask], c=cat_color_map[cat],
                                 s=sizes if sizes else self.style.marker_size * 10,
                                 alpha=alpha, label=cat, marker=marker,
                                 edgecolors='black', linewidths=0.5)

            if labels:
                self._axes.legend(loc='best', frameon=True, fancybox=False,
                                 edgecolor='black')
        else:
            scatter_colors = colors if isinstance(colors, str) else (colors or 'steelblue')
            self._axes.scatter(x, y, c=scatter_colors, s=sizes if sizes else self.style.marker_size * 10,
                             alpha=alpha, marker=marker, edgecolors='black', linewidths=0.5)

        if show_regression or regression_line:
            from scipy import stats

            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept
            self._axes.plot(x_line, y_line, 'r--', linewidth=2,
                          label=f'Regression: R²={r_value**2:.3f}')

            if regression_confidence:
                n = len(x)
                t_val = stats.t.ppf((1 + regression_confidence) / 2, df=n - 2)
                se = std_err * np.sqrt(1/n + (x_line - x.mean())**2 / np.sum((x - x.mean())**2))
                ci = t_val * se
                self._axes.fill_between(x_line, y_line - ci, y_line + ci, alpha=0.2, color='red')

            self._axes.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')

        self._apply_academic_style(self._axes, title, xlabel, ylabel, 'best', show_grid)

        if xlim:
            self._axes.set_xlim(xlim)
        if ylim:
            self._axes.set_ylim(ylim)

        self._figure.tight_layout()
        return self._figure

    def heatmap(self,
                data: Union[np.ndarray, pd.DataFrame],
                x_labels: Optional[List[str]] = None,
                y_labels: Optional[List[str]] = None,
                title: Optional[str] = None,
                xlabel: Optional[str] = None,
                ylabel: Optional[str] = None,
                cmap: str = 'RdBu_r',
                center: Optional[float] = None,
                vmin: Optional[float] = None,
                vmax: Optional[float] = None,
                annot: bool = True,
                fmt: str = '.2f',
                annot_size: float = 10,
                linewidths: float = 0.5,
                linecolor: str = 'white',
                show_grid: bool = False,
                colorbar_label: Optional[str] = None,
                droplevel: bool = False) -> plt.Figure:
        """
        Create a heatmap for matrix data visualization.

        Args:
            data: 2D array or DataFrame for the heatmap.
            x_labels: Labels for x-axis (columns).
            y_labels: Labels for y-axis (rows).
            title: Chart title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            cmap: Colormap name.
            center: Center value for diverging colormaps.
            vmin: Minimum value for color scale.
            vmax: Maximum value for color scale.
            annot: Whether to show values in cells.
            fmt: Format string for annotations.
            annot_size: Font size for annotations.
            linewidths: Width of grid lines.
            linecolor: Color of grid lines.
            show_grid: Whether to show grid (always True for heatmap).
            colorbar_label: Label for colorbar.
            droplevel: Whether to drop the first level of multi-index.

        Returns:
            Matplotlib figure object.
        """
        self._figure, self._axes = plt.subplots(figsize=(max(8, len(data.columns) * 0.6),
                                                         max(6, len(data.index) * 0.6)))

        if isinstance(data, pd.DataFrame):
            df = data.copy()
            if droplevel and isinstance(df.index, pd.MultiIndex):
                df.index = df.index.droplevel(0)
            if droplevel and isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
            plot_data = df.values
            if x_labels is None:
                x_labels = df.columns.tolist()
            if y_labels is None:
                y_labels = df.index.tolist()
        else:
            plot_data = np.array(data)

        im = self._axes.imshow(plot_data, cmap=cmap, center=center,
                               vmin=vmin, vmax=vmax, aspect='auto')

        if x_labels:
            self._axes.set_xticks(np.arange(len(x_labels)))
            self._axes.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
        if y_labels:
            self._axes.set_yticks(np.arange(len(y_labels)))
            self._axes.set_yticklabels(y_labels, fontsize=9)

        self._axes.set_xticks(np.arange(len(x_labels) + 1) - 0.5, minor=True)
        self._axes.set_yticks(np.arange(len(y_labels) + 1) - 0.5, minor=True)
        self._axes.grid(which='minor', color=linecolor, linewidth=linewidths)

        if annot:
            self._axes.texts = []
            for i in range(len(y_labels)):
                for j in range(len(x_labels)):
                    try:
                        value = plot_data[i, j]
                        text = self._axes.text(j, i, format(value, fmt),
                                               ha='center', va='center',
                                               color='white' if abs(value) > (vmax - vmin) / 2 + vmin else 'black',
                                               fontsize=annot_size)
                    except (IndexError, TypeError):
                        pass

        cbar = self._figure.colorbar(im, ax=self._axes, shrink=0.8)
        if colorbar_label:
            cbar.set_label(colorbar_label, fontsize=self.style.label_font_size, fontweight='bold')

        if title:
            self._axes.set_title(title, fontweight='bold', pad=15)
        if xlabel:
            self._axes.set_xlabel(xlabel, fontweight='bold', labelpad=10)
        if ylabel:
            self._axes.set_ylabel(ylabel, fontweight='bold', labelpad=10)

        self._figure.tight_layout()
        return self._figure

    def box_plot(self,
                 data: Union[List, np.ndarray, List[List]],
                 labels: Optional[List[str]] = None,
                 categories: Optional[List[str]] = None,
                 title: Optional[str] = None,
                 xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None,
                 colors: Optional[List[str]] = None,
                 show_outliers: bool = True,
                 show_means: bool = False,
                 mean_marker: str = 'D',
                 whisker_type: str = 'std',
                 whisker_multiplier: float = 1.5,
                 legend_loc: str = 'best',
                 show_grid: bool = True,
                 notch: bool = False,
                 patch_artist: bool = True,
                 vert: bool = True) -> plt.Figure:
        """
        Create a box plot for distribution comparison.

        Args:
            data: Data for box plots (list of arrays).
            labels: Labels for each box.
            categories: Categories for grouped box plots.
            title: Chart title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            colors: Custom colors for boxes.
            show_outliers: Whether to show outliers.
            show_means: Whether to show mean markers.
            mean_marker: Marker style for means.
            whisker_type: 'std' or 'range' for whisker calculation.
            whisker_multiplier: Multiplier for IQR or std.
            legend_loc: Legend location.
            show_grid: Whether to show grid.
            notch: Whether to show confidence intervals.
            patch_artist: Whether to use patch artist for boxes.
            vert: Whether to create vertical boxes.

        Returns:
            Matplotlib figure object.
        """
        self._figure, self._axes = plt.subplots(figsize=self.style.figsize)

        data_arrays = [np.array(d) for d in data]
        n_boxes = len(data_arrays)
        colors = colors or self._get_colors(n_boxes)

        bp = self._axes.boxplot(data_arrays, labels=labels, patch_artist=patch_artist,
                               showfliers=show_outliers, showmeans=show_means,
                               meanprops=dict(marker=mean_marker, markerfacecolor='white',
                                            markeredgecolor='black', markersize=8),
                               notch=notch, vert=vert)

        for i, (patch, color) in enumerate(zip(bp['boxes'], colors)):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)

        for element in ['whiskers', 'caps', 'medians']:
            if element in bp:
                for line in bp[element]:
                    line.set_color('black')
                    line.set_linewidth(1.5)

        if show_outliers and 'fliers' in bp:
            for flier in bp['fliers']:
                flier.set(marker='o', markerfacecolor='gray', alpha=0.5, markersize=6)

        self._apply_academic_style(self._axes, title, xlabel, ylabel, legend_loc, show_grid)

        self._figure.tight_layout()
        return self._figure

    def add_significance_markers(self,
                                 pairs: List[Tuple[int, int]],
                                 p_values: List[float],
                                 y_positions: List[float],
                                 height: float = 0.02) -> None:
        """
        Add statistical significance markers between pairs of groups.

        Args:
            pairs: List of (group1_index, group2_index) tuples.
            p_values: Corresponding p-values for each pair.
            y_positions: Y positions for each marker line.
            height: Height increment for stacked markers.

        Example:
            >>> fig = generator.bar_plot(...)
            >>> generator.add_significance_markers(
            ...     pairs=[(0, 1), (0, 2)],
            ...     p_values=[0.001, 0.03],
            ...     y_positions=[1.1, 1.2]
            ... )
        """
        if self._axes is None:
            raise ValueError("No active axes. Create a plot first.")

        for (i, j), p_val, y_pos in zip(pairs, p_values, y_positions):
            if p_val < 0.0001:
                sig = '****'
            elif p_val < 0.001:
                sig = '***'
            elif p_val < 0.01:
                sig = '**'
            elif p_val < 0.05:
                sig = '*'
            else:
                sig = 'ns'

            x1, x2 = self._axes.get_xticks()[i], self._axes.get_xticks()[j]
            if x1 > x2:
                x1, x2 = x2, x1

            line_height = y_pos
            bar_height = height

            self._axes.plot([x1, x1, x2, x2], [y_pos, y_pos + bar_height,
                                               y_pos + bar_height, y_pos],
                          'k-', linewidth=1)

            if sig != 'ns':
                self._axes.text((x1 + x2) / 2, y_pos + bar_height * 1.5, sig,
                               ha='center', va='bottom', fontsize=12, fontweight='bold')

    def save(self,
              filename: str,
              directory: Optional[str] = None,
              format: str = 'pdf',
              dpi: Optional[int] = None,
              **kwargs) -> Path:
        """
        Save the current figure to a file.

        Args:
            filename: Output filename (without extension if format specified).
            directory: Output directory (uses current directory if None).
            format: Output format ('pdf', 'svg', 'png', 'eps', etc.).
            dpi: Output resolution (uses style default if None).
            **kwargs: Additional arguments passed to savefig.

        Returns:
            Path to the saved file.
        """
        if self._figure is None:
            raise ValueError("No figure to save. Create a plot first.")

        output_dir = Path(directory) if directory else Path.cwd() / 'outputs' / 'figures'
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{filename}.{format}"

        self._figure.savefig(output_path, dpi=dpi or self.style.dpi, format=format, **kwargs)

        return output_path

    @property
    def figure(self) -> Optional[plt.Figure]:
        """Return the current figure."""
        return self._figure

    @property
    def axes(self) -> Optional[plt.Axes]:
        """Return the current axes."""
        return self._axes

    def close(self) -> None:
        """Close the current figure and clear memory."""
        if self._figure is not None:
            plt.close(self._figure)
            self._figure = None
            self._axes = None

    def __enter__(self) -> 'ChartGenerator':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        return f"ChartGenerator(style={self.style})"