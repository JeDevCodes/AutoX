from typing import Tuple, Optional, Dict, Any
import os
import pandas as pd
import warnings

# Try to import optional libs
try:
    import s3fs  # noqa: F401
    _HAS_S3 = True
except Exception:
    _HAS_S3 = False

try:
    import sqlalchemy  #
    from sqlalchemy import inspect
    _HAS_SQLA = True
except Exception:
    _HAS_SQLA = False


_SQL_PREFIXES = (
    "sqlite://", "postgresql://", "mysql://",
    "mssql://", "oracle://", "redshift://"
)

def _is_sql_conn(path: str) -> bool:
    return any(path.startswith(pref) for pref in _SQL_PREFIXES)


def _read_sql(path: str, table: Optional[str] = None) -> pd.DataFrame:
    if not _HAS_SQLA:
        raise ImportError("Reading from SQL requires `sqlalchemy`. Install it with `pip install sqlalchemy`.")
    engine = sqlalchemy.create_engine(path)
    insp = inspect(engine)

    if not table:
        table_names = insp.get_table_names()
        if not table_names:
            raise ValueError("No tables found in the SQL database. Provide `sql_table` explicitly.")
        table = table_names[0]
    return pd.read_sql_table(table, con=engine)



def load_data(path: str,
              sample_frac: Optional[float] = None,
              sql_table: Optional[str] = None,
              infer_schema: bool = True,
              dtype_hints: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    path = str(path)
    df = None

    if path.startswith('s3://'):
        if not _HAS_S3:
            raise ImportError("To read from s3:// paths you need `s3fs`. Install with `pip install s3fs`.")
        # pandas can read s3:// paths if s3fs is installed
        if path.endswith('.parquet'):
            df = pd.read_parquet(path, engine='pyarrow')``
        else:
            df = pd.read_csv(path, dtype=dtype_hints)   

    elif _is_sql_conn(path):
        df = _read_sql(path, table=sql_table)

    else:
        # Local file
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        
        if path.endswith('.parquet') or path.endswith('.pq'):
            df = pd.read_parquet(path)
        elif path.endswith('.csv') or path.endswith('.txt'):
            df = pd.read_csv(path, dtype=dtype_hints)
        else:
            # fallback: try CSV then parquet
            try:
                df = pd.read_csv(path)
            except Exception:
                raise ValueError("Unknown file extension. Use .csv or .parquet")

    if df is None:
        raise RuntimeError("Failed to load data from path: " + path)

    # Optionally sample
    if sample_frac and 0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    # detect schema
    schema = detect_schema(df) if infer_schema else {}
    return df, schema


def detect_schema(df: pd.DataFrame, max_unique_for_cardinality: int = 50) -> Dict[str, Any]:

    schema = {}
    for col in df.columns:
        ser = df[col]
        missing_pct = float(ser.isna().mean())
        nunique = int(ser.nunique(dropna=True))
        sample_vals = ser.dropna().head(7).tolist()

        # guess types
        if pd.api.types.is_datetime64_any_dtype(ser):
            guessed = 'datetime'
        elif pd.api.types.is_bool_dtype(ser):
            guessed = 'bool'
        elif pd.api.types.is_numeric_dtype(ser):
            guessed = 'numeric'
        elif pd.api.types.is_categorical_dtype(ser):
            guessed = 'categorical'
        elif pd.api.types.is_object_dtype(ser):
            # try parse datetime strings heuristically
            try:
                parsed = pd.to_datetime(ser.dropna().iloc[:20], errors='coerce')
                if parsed.notna().sum() >= 7:
                    guessed = 'datetime'
                else:
                    # simple check for high cardinality text
                    if nunique > max_unique_for_cardinality:
                        guessed = 'text'
                    else:
                        guessed = 'categorical'
            except Exception:
                guessed = 'categorical'
        else:
            guessed = str(ser.dtype)

        schema[col] = {
            'dtype': guessed,
            'missing_pct': missing_pct,
            'nunique': nunique,
            'sample_values': sample_vals
        }
    return schema
