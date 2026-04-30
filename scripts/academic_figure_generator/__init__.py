"""
Academic Figure Generator Package
================================

A comprehensive toolkit for generating publication-quality figures for research papers.
Supports multiple chart types, output formats, and document integration.

Author: Cancer-Classification Project
License: MIT
"""

from .data_reader import DataReader
from .chart_generator import ChartGenerator
from .output_manager import OutputManager
from .document_integrator import DocumentIntegrator

__version__ = "1.0.0"
__all__ = [
    "DataReader",
    "ChartGenerator",
    "OutputManager",
    "DocumentIntegrator",
]