"""
Preprocessing module with sklearn-compatible transformers.
"""

from typing import Tuple, List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import os
import joblib
import warnings

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, 
    OneHotEncoder, 
    OrdinalEncoder, 
    FunctionTransformer,
    LabelEncoder,
    MinMaxScaler
)
from sklearn.base import BaseEstimator, TransformerMixin


DEFAULT_MAX_OHE_CARDINALITY = 20


class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract features from datetime columns.
    sklearn-compatible with fit/transform interface.
    """
    
    def __init__(self, datetime_cols: List[str], drop_original: bool = True):
        self.datetime_cols = datetime_cols
        self.drop_original = drop_original
        self.feature_names_out_ = None
    
    def fit(self, X, y=None):
        # Generate feature names
        self.feature_names_out_ = []
        for col in self.datetime_cols:
            self.feature_names_out_.extend([
                f"{col}__year",
                f"{col}__month",
                f"{col}__day",
                f"{col}__hour",
                f"{col}__weekday",
                f"{col}__dayofyear"
            ])
        return self
    
    def transform(self, X):
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.datetime_cols)
        else:
            X_df = X.copy()
        
        result_arrays = []
        
        for col in self.datetime_cols:
            if col in X_df.columns:
                ser = pd.to_datetime(X_df[col], errors='coerce')
            else:
                # Column might be passed as single column array
                ser = pd.to_datetime(X_df.iloc[:, self.datetime_cols.index(col)], errors='coerce')
            
            features = np.column_stack([
                ser.dt.year.fillna(0).astype(float),
                ser.dt.month.fillna(0).astype(float),
                ser.dt.day.fillna(0).astype(float),
                ser.dt.hour.fillna(0).astype(float),
                ser.dt.weekday.fillna(0).astype(float),
                ser.dt.dayofyear.fillna(0).astype(float)
            ])
            result_arrays.append(features)
        
        if result_arrays:
            return np.hstack(result_arrays)
        return np.zeros((len(X_df), 0))
    
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


class TextVectorizer(BaseEstimator, TransformerMixin):
    """
    Simple text vectorizer wrapper for high-cardinality text columns.
    Uses TF-IDF by default with dimensionality reduction.
    """
    
    def __init__(self, max_features: int = 100):
        self.max_features = max_features
        self.vectorizers_ = {}
        self.columns_ = None
    
    def fit(self, X, y=None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X
        
        self.columns_ = list(X_df.columns)
        
        for col in self.columns_:
            vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english',
                lowercase=True
            )
            # Fill NA with empty string for text processing
            text_data = X_df[col].fillna('').astype(str)
            vectorizer.fit(text_data)
            self.vectorizers_[col] = vectorizer
        
        return self
    
    def transform(self, X):
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.columns_)
        else:
            X_df = X
        
        result_arrays = []
        
        for col in self.columns_:
            text_data = X_df[col].fillna('').astype(str)
            transformed = self.vectorizers_[col].transform(text_data).toarray()
            result_arrays.append(transformed)
        
        if result_arrays:
            return np.hstack(result_arrays)
        return np.zeros((len(X_df), 0))
    
    def get_feature_names_out(self, input_features=None):
        names = []
        for col in self.columns_:
            vocab = self.vectorizers_[col].get_feature_names_out()
            names.extend([f"{col}__{name}" for name in vocab])
        return np.array(names)


def detect_column_types(
    df: pd.DataFrame,
    max_ohe_cardinality: int = DEFAULT_MAX_OHE_CARDINALITY,
    datetime_parse_sample: int = 50
) -> Dict[str, List[str]]:
    """
    Detect and categorize column types.
    
    Returns
    -------
    Dict with keys: numeric, bool, categorical_low, categorical_high, datetime, text
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    
    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    datetime_cols = []
    categorical_low = []
    categorical_high = []
    text_cols = []
    
    for col in object_cols:
        ser = df[col].dropna()
        
        if ser.empty:
            categorical_low.append(col)
            continue
        
        # Heuristic: try parsing as datetime
        try:
            sample_size = min(datetime_parse_sample, len(ser))
            parsed = pd.to_datetime(ser.iloc[:sample_size], errors='coerce')
            parse_ratio = parsed.notna().sum() / sample_size
            
            if parse_ratio >= 0.4:
                datetime_cols.append(col)
                continue
        except Exception:
            pass
        
        # Cardinality-based classification
        nunique = df[col].nunique(dropna=True)
        
        if nunique > max_ohe_cardinality * 5:  # Very high cardinality = text
            text_cols.append(col)
        elif nunique > max_ohe_cardinality:
            categorical_high.append(col)
        else:
            categorical_low.append(col)
    
    # Include bool columns in categorical_low
    categorical_low.extend(bool_cols)
    
    return {
        'numeric': numeric_cols,
        'bool': bool_cols,
        'categorical_low': categorical_low,
        'categorical_high': categorical_high,
        'datetime': datetime_cols,
        'text': text_cols
    }


def build_preprocessor(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    max_ohe_cardinality: int = DEFAULT_MAX_OHE_CARDINALITY,
    include_datetime: bool = True,
    include_text: bool = False,
    text_max_features: int = 100,
    numeric_strategy: str = 'median',
    scale_numeric: bool = True,
    schema: Optional[Dict[str, Any]] = None
) -> Tuple[ColumnTransformer, Dict[str, Any]]:
    """
    Build a preprocessing ColumnTransformer based on detected column types.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (features only, or will exclude target_col).
    target_col : str, optional
        Target column to exclude from preprocessing.
    max_ohe_cardinality : int, default 20
        Max unique values for one-hot encoding.
    include_datetime : bool, default True
        Whether to extract datetime features.
    include_text : bool, default False
        Whether to vectorize text columns.
    text_max_features : int, default 100
        Max features per text column for TF-IDF.
    numeric_strategy : str, default 'median'
        Imputation strategy for numeric columns.
    scale_numeric : bool, default True
        Whether to standardize numeric features.
    schema : dict, optional
        Pre-computed schema from detect_schema (can inform type detection).
    
    Returns
    -------
    Tuple[ColumnTransformer, Dict[str, Any]]
        Fitted preprocessor and metadata about column types.
    """
    # Exclude target column if specified
    if target_col and target_col in df.columns:
        df = df.drop(columns=[target_col])
    
    # Detect column types
    col_types = detect_column_types(df, max_ohe_cardinality)
    
    numeric_cols = col_types['numeric']
    categorical_low = col_types['categorical_low']
    categorical_high = col_types['categorical_high']
    datetime_cols = col_types['datetime']
    text_cols = col_types['text']
    
    transformers = []
    
    # Numeric pipeline
    if numeric_cols:
        numeric_steps = [('imputer', SimpleImputer(strategy=numeric_strategy))]
        if scale_numeric:
            numeric_steps.append(('scaler', StandardScaler()))
        numeric_pipeline = Pipeline(numeric_steps)
        transformers.append(('num', numeric_pipeline, numeric_cols))
    
    # Low cardinality categorical (one-hot encoding)
    if categorical_low:
        cat_low_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False,  # Updated from sparse=False
                drop='if_binary'  # Avoid redundant columns for binary
            ))
        ])
        transformers.append(('cat_ohe', cat_low_pipeline, categorical_low))
    
    # High cardinality categorical (ordinal encoding)
    if categorical_high:
        cat_high_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='__MISSING__')),
            ('ordinal', OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ))
        ])
        transformers.append(('cat_ord', cat_high_pipeline, categorical_high))
    
    # Datetime features
    if include_datetime and datetime_cols:
        dt_transformer = DatetimeFeatureExtractor(datetime_cols, drop_original=True)
        transformers.append(('dt', dt_transformer, datetime_cols))
    
    # Text features
    if include_text and text_cols:
        text_transformer = TextVectorizer(max_features=text_max_features)
        transformers.append(('text', text_transformer, text_cols))
    
    if not transformers:
        warnings.warn("No columns to preprocess. Returning passthrough transformer.")
        preprocessor = ColumnTransformer(
            transformers=[],
            remainder='passthrough'
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',
            sparse_threshold=0.0  # Force dense output
        )
    
    metadata = {
        'numeric_columns': numeric_cols,
        'categorical_low_card': categorical_low,
        'categorical_high_card': categorical_high,
        'datetime_columns': datetime_cols,
        'text_columns': text_cols,
        'all_feature_columns': (
            numeric_cols + categorical_low + categorical_high + 
            datetime_cols + (text_cols if include_text else [])
        )
    }
    
    return preprocessor, metadata


def fit_preprocessor(
    preprocessor: ColumnTransformer, 
    X: pd.DataFrame
) -> ColumnTransformer:
    """Fit the ColumnTransformer on X and return the fitted object."""
    preprocessor.fit(X)
    return preprocessor


def save_preprocessor(preprocessor: ColumnTransformer, path: str) -> str:
    """Save preprocessor to disk."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    joblib.dump(preprocessor, path)
    return path


def load_preprocessor(path: str) -> ColumnTransformer:
    """Load preprocessor from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessor file not found: {path}")
    return joblib.load(path)


def encode_target(
    y: pd.Series,
    task: str = 'classification'
) -> Tuple[np.ndarray, Optional[LabelEncoder]]:
    """
    Encode target variable.
    
    Returns
    -------
    Tuple[np.ndarray, Optional[LabelEncoder]]
        Encoded target and encoder (for classification) or None.
    """
    if task == 'classification':
        if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y.fillna('__MISSING__'))
            return y_encoded, encoder
        else:
            return y.values, None
    else:
        # Regression - just return as float array
        return y.astype(float).values, None