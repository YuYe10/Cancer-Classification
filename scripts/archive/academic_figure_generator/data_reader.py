"""
Data Reader Module
=================

Provides comprehensive data reading capabilities for experimental data,
supporting CSV and Excel formats with built-in data cleaning and
basic statistical analysis.

Author: Cancer-Classification Project
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Tuple, Dict, Any
import warnings


class DataReader:
    """
    A class for reading and processing experimental data from various file formats.

    This class provides unified interfaces for reading CSV and Excel files,
    with built-in data cleaning, missing value handling, and basic statistics.

    Attributes:
        data (pd.DataFrame): The loaded and processed data.
        file_path (Path): Path to the currently loaded data file.
        file_type (str): Type of the loaded file ('csv' or 'excel').

    Example:
        >>> reader = DataReader()
        >>> reader.read_csv('experiment_results.csv')
        >>> reader.handle_missing(strategy='mean')
        >>> stats = reader.get_statistics()
    """

    def __init__(self, file_path: Optional[str] = None):
        """
        Initialize the DataReader.

        Args:
            file_path: Optional path to a data file to load immediately.
        """
        self.data: Optional[pd.DataFrame] = None
        self.file_path: Optional[Path] = None
        self.file_type: Optional[str] = None
        self._original_data: Optional[pd.DataFrame] = None

        if file_path:
            self.read(file_path)

    def read(self, file_path: str, **kwargs) -> 'DataReader':
        """
        Read data from a file, automatically detecting the format.

        Args:
            file_path: Path to the data file (CSV or Excel).
            **kwargs: Additional arguments passed to the specific reader.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If the file format is not supported.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == '.csv':
            return self.read_csv(file_path, **kwargs)
        elif suffix in ['.xlsx', '.xls']:
            return self.read_excel(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. "
                           f"Supported formats: .csv, .xlsx, .xls")

    def read_csv(self, file_path: str,
                 encoding: str = 'utf-8',
                 delimiter: str = ',',
                 **kwargs) -> 'DataReader':
        """
        Read data from a CSV file.

        Args:
            file_path: Path to the CSV file.
            encoding: Character encoding of the file (default: 'utf-8').
            delimiter: Field delimiter (default: ',').
            **kwargs: Additional arguments passed to pd.read_csv.

        Returns:
            Self for method chaining.

        Example:
            >>> reader = DataReader()
            >>> reader.read_csv('data.csv', encoding='latin1')
        """
        self.file_path = Path(file_path)
        self.file_type = 'csv'

        try:
            self.data = pd.read_csv(
                self.file_path,
                encoding=encoding,
                delimiter=delimiter,
                **kwargs
            )
            self._original_data = self.data.copy()
        except Exception as e:
            raise IOError(f"Failed to read CSV file: {e}")

        return self

    def read_excel(self, file_path: str,
                   sheet_name: Union[str, int] = 0,
                   **kwargs) -> 'DataReader':
        """
        Read data from an Excel file.

        Args:
            file_path: Path to the Excel file.
            sheet_name: Name or index of the sheet to read (default: 0).
            **kwargs: Additional arguments passed to pd.read_excel.

        Returns:
            Self for method chaining.

        Example:
            >>> reader = DataReader()
            >>> reader.read_excel('data.xlsx', sheet_name='Results')
        """
        self.file_path = Path(file_path)
        self.file_type = 'excel'

        try:
            self.data = pd.read_excel(
                self.file_path,
                sheet_name=sheet_name,
                **kwargs
            )
            self._original_data = self.data.copy()
        except Exception as e:
            raise IOError(f"Failed to read Excel file: {e}")

        return self

    def handle_missing(self,
                      strategy: str = 'mean',
                      columns: Optional[List[str]] = None,
                      fill_value: Any = None) -> 'DataReader':
        """
        Handle missing values in the data.

        Args:
            strategy: Strategy for handling missing values.
                     Options: 'mean', 'median', 'mode', 'drop', 'fill'.
            columns: Specific columns to process (None = all columns).
            fill_value: Value to use when strategy='fill'.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If an invalid strategy is provided.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call read() first.")

        if columns is None:
            columns = self.data.columns.tolist()

        df = self.data.copy()

        for col in columns:
            if col not in df.columns:
                warnings.warn(f"Column '{col}' not found in data.")
                continue

            if df[col].isna().sum() == 0:
                continue

            if strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col].fillna(mode_val[0], inplace=True)
            elif strategy == 'drop':
                df.dropna(subset=[col], inplace=True)
            elif strategy == 'fill':
                if fill_value is not None:
                    df[col].fillna(fill_value, inplace=True)
                else:
                    raise ValueError("fill_value must be provided when strategy='fill'")
            else:
                raise ValueError(f"Invalid strategy: {strategy}. "
                               f"Options: mean, median, mode, drop, fill")

        self.data = df
        return self

    def remove_outliers(self,
                       columns: Optional[List[str]] = None,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> 'DataReader':
        """
        Remove outliers from the data.

        Args:
            columns: Columns to check for outliers (None = numeric columns).
            method: Method for outlier detection ('iqr' or 'zscore').
            threshold: Threshold for outlier detection.
                      For IQR: multiplier of IQR (default: 1.5)
                      For Z-score: minimum z-score to be considered outlier (default: 3.0)

        Returns:
            Self for method chaining.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call read() first.")

        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        df = self.data.copy()

        for col in columns:
            if col not in df.columns:
                continue

            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < threshold]
            else:
                raise ValueError(f"Invalid method: {method}. Options: iqr, zscore")

        self.data = df
        return self

    def normalize(self,
                  columns: Optional[List[str]] = None,
                  method: str = 'zscore') -> 'DataReader':
        """
        Normalize specified columns.

        Args:
            columns: Columns to normalize (None = all numeric columns).
            method: Normalization method ('zscore' or 'minmax').

        Returns:
            Self for method chaining.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call read() first.")

        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        df = self.data.copy()

        for col in columns:
            if col not in df.columns:
                continue

            if method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std
            elif method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                raise ValueError(f"Invalid method: {method}. Options: zscore, minmax")

        self.data = df
        return self

    def get_statistics(self,
                      columns: Optional[List[str]] = None,
                      include_missing: bool = True) -> pd.DataFrame:
        """
        Calculate basic statistics for the data.

        Args:
            columns: Columns to include (None = all columns).
            include_missing: Whether to include missing value statistics.

        Returns:
            DataFrame containing statistics for each column.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call read() first.")

        if columns is None:
            columns = self.data.columns.tolist()

        stats_list = []

        for col in columns:
            if col not in self.data.columns:
                continue

            col_data = self.data[col]
            is_numeric = pd.api.types.is_numeric_dtype(col_data)

            stats = {'column': col, 'dtype': str(col_data.dtype)}

            if is_numeric:
                stats.update({
                    'count': len(col_data),
                    'missing': col_data.isna().sum(),
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'q25': col_data.quantile(0.25),
                    'median': col_data.median(),
                    'q75': col_data.quantile(0.75),
                    'max': col_data.max(),
                })
            else:
                stats.update({
                    'count': len(col_data),
                    'missing': col_data.isna().sum(),
                    'unique': col_data.nunique(),
                    'mode': col_data.mode().iloc[0] if len(col_data.mode()) > 0 else None,
                })

            stats_list.append(stats)

        return pd.DataFrame(stats_list)

    def filter(self,
               conditions: Dict[str, Any],
               operator: str = '&') -> 'DataReader':
        """
        Filter rows based on conditions.

        Args:
            conditions: Dictionary of column-value conditions.
            operator: Logical operator ('&' for AND, '|' for OR).

        Returns:
            Self for method chaining.

        Example:
            >>> reader.filter({'method': 'Concat', 'accuracy': lambda x: x > 0.8})
        """
        if self.data is None:
            raise ValueError("No data loaded. Call read() first.")

        df = self.data.copy()
        masks = []

        for col, condition in conditions.items():
            if col not in df.columns:
                warnings.warn(f"Column '{col}' not found, skipping filter.")
                continue

            if callable(condition):
                masks.append(df[col].apply(condition))
            else:
                masks.append(df[col] == condition)

        if masks:
            if operator == '&':
                combined_mask = masks[0]
                for mask in masks[1:]:
                    combined_mask = combined_mask & mask
            else:
                combined_mask = masks[0]
                for mask in masks[1:]:
                    combined_mask = combined_mask | mask

            df = df[combined_mask]

        self.data = df
        return self

    def group_by(self,
                 column: str,
                 agg_funcs: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Group data by a column and apply aggregation functions.

        Args:
            column: Column to group by.
            agg_funcs: Dictionary of {column: function} aggregations.
                      If None, applies mean to all numeric columns.

        Returns:
            Grouped DataFrame.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call read() first.")

        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found.")

        if agg_funcs is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            agg_funcs = {col: 'mean' for col in numeric_cols}

        return self.data.groupby(column).agg(agg_funcs)

    def pivot(self,
              index: str,
              columns: str,
              values: str,
              aggfunc: str = 'mean') -> pd.DataFrame:
        """
        Create a pivot table from the data.

        Args:
            index: Column to use as index.
            columns: Column to use as columns.
            values: Column to use as values.
            aggfunc: Aggregation function (default: 'mean').

        Returns:
            Pivot table as DataFrame.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call read() first.")

        return pd.pivot_table(
            self.data,
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc
        )

    def to_numpy(self, columns: Optional[List[str]] = None) -> np.ndarray:
        """
        Convert data to numpy array.

        Args:
            columns: Columns to include (None = all numeric columns).

        Returns:
            Numpy array of the data.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call read() first.")

        if columns is None:
            data = self.data.select_dtypes(include=[np.number])
        else:
            data = self.data[columns]

        return data.values

    def get_matrix(self,
                   index_col: str,
                   value_col: str,
                   columns_col: str) -> pd.DataFrame:
        """
        Create a matrix (pivot table) from long-format data.

        Args:
            index_col: Column to use as matrix index.
            value_col: Column containing matrix values.
            columns_col: Column to use as matrix columns.

        Returns:
            Matrix as DataFrame with index_col as index and columns_col as columns.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call read() first.")

        matrix = self.data.pivot_table(
            index=index_col,
            columns=columns_col,
            values=value_col,
            aggfunc='mean'
        )

        return matrix

    def reset(self) -> 'DataReader':
        """
        Reset data to original state (undo all transformations).

        Returns:
            Self for method chaining.
        """
        if self._original_data is not None:
            self.data = self._original_data.copy()
        return self

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the current data."""
        if self.data is None:
            return (0, 0)
        return self.data.shape

    @property
    def columns(self) -> List[str]:
        """Return the column names of the current data."""
        if self.data is None:
            return []
        return self.data.columns.tolist()

    def __repr__(self) -> str:
        if self.data is None:
            return "DataReader(empty)"
        return f"DataReader(shape={self.shape}, file={self.file_path})"