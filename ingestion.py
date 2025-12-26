"""
Data ingestion module supporting CSV, Parquet, S3, and SQL sources.
"""

from typing import Tuple, Optional, Dict, Any, List
import os
import pandas as pd
import warnings

# Optional imports with flags
try:
    import s3fs  # noqa: F401
    _HAS_S3 = True
except ImportError:
    _HAS_S3 = False

try:
    import sqlalchemy
    from sqlalchemy import inspect
    _HAS_SQLA = True
except ImportError:
    _HAS_SQLA = False


_SQL_PREFIXES = (
    "sqlite://", "postgresql://", "postgres://", "mysql://",
    "mssql://", "oracle://", "redshift://"
)


def _is_sql_conn(path: str) -> bool:
    """Check if path is a SQL connection string."""
    return any(path.lower().startswith(pref) for pref in _SQL_PREFIXES)


def _read_sql(path: str, table: Optional[str] = None) -> pd.DataFrame:
    """Read data from SQL database."""
    if not _HAS_SQLA:
        raise ImportError(
            "Reading from SQL requires `sqlalchemy`. "
            "Install it with `pip install sqlalchemy`."
        )
    
    engine = sqlalchemy.create_engine(path)
    insp = inspect(engine)

    if not table:
        table_names = insp.get_table_names()
        if not table_names:
            raise ValueError(
                "No tables found in the SQL database. "
                "Provide `sql_table` explicitly."
            )
        table = table_names[0]
        warnings.warn(f"No table specified, using first table: '{table}'")
    
    return pd.read_sql_table(table, con=engine)


def _infer_file_type(path: str) -> str:
    """Infer file type from extension."""
    path_lower = path.lower()
    if path_lower.endswith('.parquet') or path_lower.endswith('.pq'):
        return 'parquet'
    elif path_lower.endswith('.csv'):
        return 'csv'
    elif path_lower.endswith('.tsv'):
        return 'tsv'
    elif path_lower.endswith('.json'):
        return 'json'
    elif path_lower.endswith(('.xls', '.xlsx')):
        return 'excel'
    elif path_lower.endswith('.feather'):
        return 'feather'
    else:
        return 'unknown'


def load_data(
    path: str,
    sample_frac: Optional[float] = None,
    sql_table: Optional[str] = None,
    infer_schema: bool = True,
    dtype_hints: Optional[Dict[str, Any]] = None,
    **read_kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load data from various sources.
    
    Parameters
    ----------
    path : str
        Path to data file, S3 URI, or SQL connection string.
    sample_frac : float, optional
        Fraction of data to sample (0 < sample_frac <= 1).
    sql_table : str, optional
        Table name for SQL sources.
    infer_schema : bool, default True
        Whether to detect and return schema information.
    dtype_hints : dict, optional
        Dtype hints to pass to pandas read functions.
    **read_kwargs : dict
        Additional arguments passed to pandas read functions.
    
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Loaded DataFrame and inferred schema dictionary.
    
    Raises
    ------
    FileNotFoundError
        If local file doesn't exist.
    ImportError
        If required optional dependency is missing.
    ValueError
        If file format is unknown or unsupported.
    """
    path = str(path).strip()
    df = None

    # S3 path
    if path.startswith('s3://'):
        if not _HAS_S3:
            raise ImportError(
                "To read from s3:// paths you need `s3fs`. "
                "Install with `pip install s3fs`."
            )
        file_type = _infer_file_type(path)
        if file_type == 'parquet':
            df = pd.read_parquet(path, engine='pyarrow', **read_kwargs)
        elif file_type in ('csv', 'tsv'):
            sep = '\t' if file_type == 'tsv' else ','
            df = pd.read_csv(path, dtype=dtype_hints, sep=sep, **read_kwargs)
        else:
            df = pd.read_csv(path, dtype=dtype_hints, **read_kwargs)

    # SQL connection
    elif _is_sql_conn(path):
        df = _read_sql(path, table=sql_table)

    # Local file
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        
        file_type = _infer_file_type(path)
        
        if file_type == 'parquet':
            df = pd.read_parquet(path, **read_kwargs)
        elif file_type == 'csv':
            df = pd.read_csv(path, dtype=dtype_hints, **read_kwargs)
        elif file_type == 'tsv':
            df = pd.read_csv(path, dtype=dtype_hints, sep='\t', **read_kwargs)
        elif file_type == 'json':
            df = pd.read_json(path, dtype=dtype_hints, **read_kwargs)
        elif file_type == 'excel':
            df = pd.read_excel(path, dtype=dtype_hints, **read_kwargs)
        elif file_type == 'feather':
            df = pd.read_feather(path, **read_kwargs)
        else:
            # Fallback: try CSV
            try:
                df = pd.read_csv(path, dtype=dtype_hints, **read_kwargs)
                warnings.warn(
                    f"Unknown file extension for '{path}', attempted CSV parsing."
                )
            except Exception as e:
                raise ValueError(
                    f"Unknown file extension and CSV parsing failed: {e}. "
                    "Use .csv, .parquet, .json, .tsv, or .xlsx"
                )

    if df is None:
        raise RuntimeError(f"Failed to load data from path: {path}")

    # Validate DataFrame
    if df.empty:
        warnings.warn("Loaded DataFrame is empty.")

    # Optional sampling
    if sample_frac is not None:
        if not (0 < sample_frac <= 1.0):
            raise ValueError("sample_frac must be between 0 (exclusive) and 1 (inclusive)")
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    # Detect schema
    schema = detect_schema(df) if infer_schema else {}
    
    return df, schema


def detect_schema(
    df: pd.DataFrame, 
    max_unique_for_cardinality: int = 50,
    datetime_sample_size: int = 20,
    datetime_threshold: float = 0.35
) -> Dict[str, Any]:
    """
    Detect schema information for each column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    max_unique_for_cardinality : int, default 50
        Maximum unique values to consider as categorical vs text.
    datetime_sample_size : int, default 20
        Number of samples to use for datetime detection.
    datetime_threshold : float, default 0.35
        Minimum fraction of parseable datetimes to classify as datetime.
    
    Returns
    -------
    Dict[str, Any]
        Schema dictionary with column metadata.
    """
    schema = {}
    
    for col in df.columns:
        ser = df[col]
        missing_count = int(ser.isna().sum())
        missing_pct = float(ser.isna().mean())
        nunique = int(ser.nunique(dropna=True))
        total_count = len(ser)
        
        # Sample values (non-null)
        non_null = ser.dropna()
        sample_vals = non_null.head(7).tolist() if len(non_null) > 0 else []

        # Type detection
        if pd.api.types.is_datetime64_any_dtype(ser):
            guessed = 'datetime'
        elif pd.api.types.is_bool_dtype(ser):
            guessed = 'bool'
        elif pd.api.types.is_numeric_dtype(ser):
            # Check if likely integer-based categorical
            if pd.api.types.is_integer_dtype(ser) and nunique <= max_unique_for_cardinality:
                guessed = 'numeric_categorical'
            else:
                guessed = 'numeric'
        elif pd.api.types.is_categorical_dtype(ser):
            guessed = 'categorical'
        elif pd.api.types.is_object_dtype(ser):
            # Try to detect datetime strings
            if len(non_null) > 0:
                sample_size = min(datetime_sample_size, len(non_null))
                sample_data = non_null.iloc[:sample_size]
                
                try:
                    parsed = pd.to_datetime(sample_data, errors='coerce', infer_datetime_format=True)
                    parse_ratio = parsed.notna().sum() / len(sample_data)
                    
                    if parse_ratio >= datetime_threshold:
                        guessed = 'datetime'
                    elif nunique > max_unique_for_cardinality:
                        guessed = 'text'
                    else:
                        guessed = 'categorical'
                except Exception:
                    guessed = 'categorical' if nunique <= max_unique_for_cardinality else 'text'
            else:
                guessed = 'categorical'
        else:
            guessed = str(ser.dtype)

        schema[col] = {
            'dtype': guessed,
            'pandas_dtype': str(ser.dtype),
            'missing_count': missing_count,
            'missing_pct': round(missing_pct, 4),
            'nunique': nunique,
            'total_count': total_count,
            'sample_values': sample_vals
        }
    
    return schema


def validate_target_column(df: pd.DataFrame, target: str) -> None:
    """Validate that target column exists and has no issues."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame. Available: {list(df.columns)}")
    
    if df[target].isna().all():
        raise ValueError(f"Target column '{target}' contains all missing values.")
    
    missing_pct = df[target].isna().mean()
    if missing_pct > 0.5:
        warnings.warn(f"Target column '{target}' has {missing_pct:.1%} missing values.")