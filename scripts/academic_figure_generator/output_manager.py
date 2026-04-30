"""
Output Manager Module
===================

Manages output operations for generated figures, supporting multiple
vector formats (SVG, PDF) and figure organization.

Author: Cancer-Classification Project
"""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass
import json
import datetime


@dataclass
class OutputConfig:
    """
    Configuration for output operations.

    Attributes:
        output_dir: Base output directory.
        formats: List of output formats to generate.
        dpi: Resolution for raster formats.
        figure_dir: Subdirectory for figures.
        data_dir: Subdirectory for data exports.
        create_manifest: Whether to create a manifest file.
        timestamp: Whether to add timestamps to filenames.
        organization: How to organize figures ('type', 'chapter', 'flat').
    """
    output_dir: str = 'outputs'
    formats: List[str] = None
    dpi: int = 300
    figure_dir: str = 'figures'
    data_dir: str = 'data'
    create_manifest: bool = True
    timestamp: bool = False
    organization: str = 'type'

    def __post_init__(self):
        if self.formats is None:
            self.formats = ['pdf', 'svg']


class OutputManager:
    """
    Manages output operations for academic figures.

    This class handles saving figures in multiple formats, organizing
    output directories, and creating manifests for figure tracking.

    Attributes:
        config (OutputConfig): Current output configuration.

    Example:
        >>> manager = OutputManager()
        >>> manager.save_figure(fig, 'performance_comparison', formats=['pdf', 'svg'])
        >>> manager.create_manifest('experiment_results')
    """

    FIGURE_TYPES = [
        'line', 'bar', 'scatter', 'heatmap', 'box',
        'confusion_matrix', 'roc', 'pr', 'tsne', 'other'
    ]

    CHAPTERS = [
        'introduction', 'background', 'data', 'method',
        'experiments', 'results', 'discussion', 'conclusion', 'appendix'
    ]

    def __init__(self, config: Optional[OutputConfig] = None):
        """
        Initialize the OutputManager.

        Args:
            config: Custom output configuration. Uses default if None.
        """
        self.config = config or OutputConfig()
        self._ensure_directories()
        self._manifest: Dict[str, Any] = {'figures': [], 'generated_at': datetime.datetime.now().isoformat()}

    def _ensure_directories(self) -> None:
        """Create necessary output directories."""
        base_dir = Path(self.config.output_dir)
        figures_dir = base_dir / self.config.figure_dir
        data_dir = base_dir / self.config.data_dir

        figures_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        if self.config.organization == 'type':
            for fig_type in self.FIGURE_TYPES:
                (figures_dir / fig_type).mkdir(exist_ok=True)
        elif self.config.organization == 'chapter':
            for chapter in self.CHAPTERS:
                (figures_dir / chapter).mkdir(exist_ok=True)

    def _get_output_path(self,
                       filename: str,
                       fig_type: Optional[str] = None,
                       chapter: Optional[str] = None) -> Path:
        """
        Determine the output path for a figure.

        Args:
            filename: Base filename.
            fig_type: Type of figure for organization.
            chapter: Chapter name for organization.

        Returns:
            Path object for the output directory.
        """
        base_dir = Path(self.config.output_dir) / self.config.figure_dir

        if self.config.organization == 'type' and fig_type:
            return base_dir / fig_type
        elif self.config.organization == 'chapter' and chapter:
            return base_dir / chapter
        else:
            return base_dir

    def _add_timestamp(self, filename: str) -> str:
        """Add timestamp to filename if configured."""
        if self.config.timestamp:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            name, ext = filename.rsplit('.', 1)
            return f"{name}_{timestamp}.{ext}"
        return filename

    def save_figure(self,
                   figure: plt.Figure,
                   filename: str,
                   fig_type: Optional[str] = None,
                   chapter: Optional[str] = None,
                   formats: Optional[List[str]] = None,
                   dpi: Optional[int] = None,
                   **kwargs) -> Dict[str, Path]:
        """
        Save a figure in multiple formats.

        Args:
            figure: Matplotlib figure object to save.
            filename: Base filename (without extension).
            fig_type: Figure type for organization.
            chapter: Chapter name for organization.
            formats: List of formats to save. Uses config default if None.
            dpi: Resolution for raster formats.
            **kwargs: Additional arguments passed to savefig.

        Returns:
            Dictionary mapping format to saved file paths.

        Example:
            >>> manager = OutputManager()
            >>> paths = manager.save_figure(
            ...     fig, 'performance_bar',
            ...     fig_type='bar', chapter='results',
            ...     formats=['pdf', 'svg']
            ... )
        """
        formats = formats or self.config.formats
        dpi = dpi or self.config.dpi
        output_dir = self._get_output_path(filename, fig_type, chapter)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        for fmt in formats:
            safe_filename = self._add_timestamp(filename)
            output_path = output_dir / f"{safe_filename}.{fmt}"

            if fmt in ['pdf', 'eps']:
                figure.savefig(output_path, format=fmt, bbox_inches='tight',
                            pad_inches=0.1, **kwargs)
            elif fmt == 'svg':
                figure.savefig(output_path, format='svg', bbox_inches='tight',
                            pad_inches=0.1, **kwargs)
            elif fmt in ['png', 'jpg', 'jpeg', 'tiff']:
                figure.savefig(output_path, format=fmt, dpi=dpi,
                            bbox_inches='tight', pad_inches=0.1, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {fmt}")

            saved_paths[fmt] = output_path

            self._manifest['figures'].append({
                'filename': filename,
                'formats': list(saved_paths.keys()),
                'paths': {fmt: str(p) for fmt, p in saved_paths.items()},
                'figure_type': fig_type,
                'chapter': chapter,
                'timestamp': datetime.datetime.now().isoformat()
            })

        return saved_paths

    def save_data(self,
                  data,
                  filename: str,
                  format: str = 'csv') -> Path:
        """
        Save data alongside the figure.

        Args:
            data: Data to save (DataFrame, array, or dict).
            filename: Base filename.
            format: Data format ('csv', 'json', 'txt').

        Returns:
            Path to the saved data file.
        """
        import pandas as pd
        import numpy as np

        output_dir = Path(self.config.output_dir) / self.config.data_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_filename = self._add_timestamp(filename)
        output_path = output_dir / f"{safe_filename}.{format}"

        if format == 'csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(output_path, index=False)
            elif isinstance(data, np.ndarray):
                np.savetxt(output_path, data, delimiter=',')
            else:
                raise ValueError(f"Cannot save {type(data)} as CSV")
        elif format == 'json':
            if isinstance(data, pd.DataFrame):
                data.to_json(output_path, orient='records', indent=2)
            elif isinstance(data, dict):
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                with open(output_path, 'w') as f:
                    json.dump(list(data), f, indent=2)
        elif format == 'txt':
            if isinstance(data, np.ndarray):
                np.savetxt(output_path, data)
            elif isinstance(data, (list, tuple)):
                with open(output_path, 'w') as f:
                    f.write('\n'.join(map(str, data)))
            else:
                with open(output_path, 'w') as f:
                    f.write(str(data))
        else:
            raise ValueError(f"Unsupported data format: {format}")

        return output_path

    def create_manifest(self, filename: str = 'figure_manifest') -> Path:
        """
        Create a manifest file documenting all saved figures.

        Args:
            filename: Base filename for manifest (without extension).

        Returns:
            Path to the created manifest file.
        """
        manifest_path = Path(self.config.output_dir) / self.config.figure_dir / f"{filename}.json"

        with open(manifest_path, 'w') as f:
            json.dump(self._manifest, f, indent=2)

        return manifest_path

    def get_figure_info(self, figure: plt.Figure) -> Dict[str, Any]:
        """
        Extract information from a figure for documentation.

        Args:
            figure: Matplotlib figure object.

        Returns:
            Dictionary containing figure information.
        """
        info = {
            'size_inches': figure.get_size_inches().tolist(),
            'dpi': figure.dpi,
            'axes_count': len(figure.get_axes()),
            'has_legend': any(ax.get_legend() is not None for ax in figure.get_axes()),
            'title': figure.axes[0].get_title() if figure.axes else None,
        }

        axes = figure.get_axes()
        if axes:
            info['xlabel'] = axes[0].get_xlabel()
            info['ylabel'] = axes[0].get_ylabel()
            info['xlim'] = axes[0].get_xlim()
            info['ylim'] = axes[0].get_ylim()

        return info

    def list_figures(self,
                    fig_type: Optional[str] = None,
                    chapter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all figures in the output directory.

        Args:
            fig_type: Filter by figure type.
            chapter: Filter by chapter.

        Returns:
            List of figure information dictionaries.
        """
        figures_dir = Path(self.config.output_dir) / self.config.figure_dir

        if fig_type:
            figures_dir = figures_dir / fig_type
        elif chapter:
            figures_dir = figures_dir / chapter

        figures = []

        for fmt in self.config.formats:
            for path in figures_dir.glob(f"*.{fmt}"):
                figures.append({
                    'name': path.stem,
                    'format': fmt,
                    'path': str(path),
                    'size_kb': path.stat().st_size / 1024,
                    'modified': datetime.datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                })

        return figures

    def clean_output(self, older_than_days: Optional[int] = None) -> int:
        """
        Clean output directory, optionally keeping only recent files.

        Args:
            older_than_days: Remove files older than this many days.
                          If None, removes all generated files.

        Returns:
            Number of files removed.
        """
        import time

        figures_dir = Path(self.config.output_dir) / self.config.figure_dir
        count = 0

        if older_than_days is not None:
            cutoff_time = time.time() - (older_than_days * 86400)

            for path in figures_dir.rglob('*'):
                if path.is_file() and path.stat().st_mtime < cutoff_time:
                    path.unlink()
                    count += 1
        else:
            for path in figures_dir.rglob('*'):
                if path.is_file() and path.name not in ['.gitkeep', 'manifest.json']:
                    path.unlink()
                    count += 1

        self._manifest = {'figures': [], 'generated_at': datetime.datetime.now().isoformat()}

        return count

    def __repr__(self) -> str:
        return f"OutputManager(output_dir='{self.config.output_dir}')"