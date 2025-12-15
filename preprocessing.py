from typing import Tuple, List, Dict, Any
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
import numpy as np
import joblib
import os

DEFAULT_MAX_OHE_CARDINALITY = 20


def _extract_datetime_features(df: pd.DataFrame, cols: List[str], drop_original: bool = True) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        ser = pd.to_datetime(df[c], errors='coerce')
        df[f"{c}__year"] = ser.dt.year
        df[f"{c}__month"] = ser.dt.month
        df[f"{c}__day"] = ser.dt.day
        df[f"{c}__hour"] = ser.dt.hour
        df[f"{c}__weekday"] = ser.dt.weekday
    if drop_original:
        df = df.drop(columns=cols)
    return df


def _make_datetime_transformer(cols: List[str], drop_original=True):
    # sklearn FunctionTransformer expects numpy arrays, but we can wrap to operate on pandas DF slices
    def fn(X):
        if isinstance(X, np.ndarray):
            # convert to DataFrame with cols names
            X_df = pd.DataFrame(X, columns=cols)
        else:
            X_df = X
        return _extract_datetime_features(X_df, cols, drop_original)
    return FunctionTransformer(fn, validate=False)


def build_preprocessor(df: pd.DataFrame,
                       max_ohe_cardinality: int = DEFAULT_MAX_OHE_CARDINALITY,
                       include_datetime: bool = True,
                       drop_datetime: bool = True) -> Tuple[ColumnTransformer, Dict[str, Any]]:
    
    # detect types using pandas
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()

    # object-like cols may be categorical or datetime or text
    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    datetime_cols = []
    categorical_cols = []
    text_cols = []

    for col in object_cols:
        ser = df[col].dropna()
        if ser.empty:
            categorical_cols.append(col)
            continue
        # heuristic: try parse a subset as datetime
        try:
            parsed = pd.to_datetime(ser.iloc[:50], errors='coerce')
            if parsed.notna().sum() >= max(5, int(0.4 * min(50, len(ser)))):
                datetime_cols.append(col)
                continue
        except Exception:
            pass
        # cardinality heuristic to decide text vs categorical
        if df[col].nunique(dropna=True) > max_ohe_cardinality:
            text_cols.append(col)
        else:
            categorical_cols.append(col)

    # For boolean columns, treat as categorical (0/1)
    categorical_cols += bool_cols

    # Build pipelines
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # For categorical: small cardinality => OneHot, otherwise Ordinal
    low_card_cats = [c for c in categorical_cols if df[c].nunique(dropna=True) <= max_ohe_cardinality]
    high_card_cats = [c for c in categorical_cols if df[c].nunique(dropna=True) > max_ohe_cardinality]

    transformers = []

    if numeric_cols:
        transformers.append(('num', numeric_pipeline, numeric_cols))

    if low_card_cats:
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
        transformers.append(('cat_ohe', cat_pipe, low_card_cats))

    if high_card_cats:
        # ordinal encoder will map categories to integers, preserving unknowns as -1 via SimpleImputer + OrdinalEncoder
        high_cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='__MISSING__')),
            ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        transformers.append(('cat_ord', high_cat_pipe, high_card_cats))

    if include_datetime and datetime_cols:
        # Add a datetime function transformer that expands datetime cols
        dt_transformer = _make_datetime_transformer(datetime_cols, drop_original=drop_datetime)
        transformers.append(('dt', dt_transformer, datetime_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0.3)
    metadata = {
        'numeric_columns': numeric_cols,
        'categorical_low_card': low_card_cats,
        'categorical_high_card': high_card_cats,
        'datetime_columns': datetime_cols,
        'text_columns': text_cols
    }
    return preprocessor, metadata


def fit_preprocessor(preprocessor: ColumnTransformer, X: pd.DataFrame) -> ColumnTransformer:
    """
    Fit the ColumnTransformer on X and return the fitted object.
    """
    preprocessor.fit(X)
    return preprocessor


def save_preprocessor(preprocessor: ColumnTransformer, path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path: str) -> ColumnTransformer:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)
