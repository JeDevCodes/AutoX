"""
Feature engineering utilities for automated feature generation.
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import warnings


def add_polynomial_interactions(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    max_pairs: Optional[int] = None,
    drop_original: bool = False,
    include_ratios: bool = False
) -> pd.DataFrame:
    """
    Add pairwise interaction (product) features for numeric columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    numeric_cols : list, optional
        Columns to use. If None, auto-detect numeric columns.
    max_pairs : int, optional
        Maximum number of pairs to create.
    drop_original : bool, default False
        Whether to drop original columns.
    include_ratios : bool, default False
        Whether to include ratio features (a/b).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added interaction features.
    """
    df = df.copy()
    
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) < 2:
        warnings.warn("Need at least 2 numeric columns for interactions.")
        return df
    
    pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            pairs.append((numeric_cols[i], numeric_cols[j]))
    
    if max_pairs is not None and max_pairs > 0:
        pairs = pairs[:max_pairs]
    
    for a, b in pairs:
        # Product interaction
        new_col = f"{a}__x__{b}"
        df[new_col] = df[a].astype(float) * df[b].astype(float)
        
        # Ratio features (with zero handling)
        if include_ratios:
            ratio_col = f"{a}__div__{b}"
            denominator = df[b].astype(float).replace(0, np.nan)
            df[ratio_col] = df[a].astype(float) / denominator
    
    if drop_original:
        df = df.drop(columns=numeric_cols)
    
    return df


def add_polynomial_features(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    degree: int = 2,
    include_bias: bool = False,
    interaction_only: bool = False,
    drop_original: bool = False
) -> pd.DataFrame:
    """
    Add polynomial features using sklearn's PolynomialFeatures.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    numeric_cols : list, optional
        Columns to transform. If None, auto-detect.
    degree : int, default 2
        Polynomial degree.
    include_bias : bool, default False
        Include bias (all 1s) column.
    interaction_only : bool, default False
        Only produce interaction features.
    drop_original : bool, default False
        Drop original columns.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with polynomial features added.
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    df = df.copy()
    
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        warnings.warn("No numeric columns found for polynomial features.")
        return df
    
    poly = PolynomialFeatures(
        degree=degree,
        include_bias=include_bias,
        interaction_only=interaction_only
    )
    
    # Handle missing values
    X_numeric = df[numeric_cols].fillna(0).astype(float).values
    X_poly = poly.fit_transform(X_numeric)
    feature_names = poly.get_feature_names_out(numeric_cols)
    
    poly_df = pd.DataFrame(X_poly, columns=feature_names, index=df.index)
    
    # Remove original columns from poly_df to avoid duplication
    cols_to_keep = [c for c in poly_df.columns if c not in df.columns]
    poly_df = poly_df[cols_to_keep]
    
    # Prefix new columns
    poly_df.columns = [f"poly__{c}" for c in poly_df.columns]
    
    if drop_original:
        base = df.drop(columns=numeric_cols)
    else:
        base = df
    
    return pd.concat([base, poly_df], axis=1)


def add_lag_features(
    df: pd.DataFrame,
    value_cols: Optional[List[str]] = None,
    groupby_cols: Optional[List[str]] = None,
    time_col: Optional[str] = None,
    lags: Optional[List[int]] = None,
    roll_windows: Optional[List[int]] = None,
    roll_agg: str = 'mean'
) -> pd.DataFrame:
    """
    Add lag and rolling window features for time series data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    value_cols : list, optional
        Columns to create lag features for. If None, use all numeric.
    groupby_cols : list, optional
        Columns to group by (e.g., entity ID for panel data).
    time_col : str, optional
        Time column for sorting.
    lags : list, optional
        List of lag periods. Default is [1].
    roll_windows : list, optional
        List of rolling window sizes.
    roll_agg : str, default 'mean'
        Aggregation function for rolling windows.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with lag features added.
    """
    df = df.copy()
    
    # Defaults - use copy to avoid mutable default argument issue
    if lags is None:
        lags = [1]
    else:
        lags = list(lags)
    
    if roll_windows is None:
        roll_windows = []
    else:
        roll_windows = list(roll_windows)
    
    if value_cols is None:
        value_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if groupby_cols is None:
        groupby_cols = []
    
    # Sort data
    sort_cols = groupby_cols + ([time_col] if time_col else [])
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    
    def add_features_to_group(group_df: pd.DataFrame) -> pd.DataFrame:
        result = group_df.copy()
        
        for col in value_cols:
            if col not in result.columns:
                continue
                
            # Lag features
            for lag in lags:
                result[f"{col}__lag_{lag}"] = result[col].shift(lag)
            
            # Rolling features
            for window in roll_windows:
                shifted = result[col].shift(1)  # Shift to avoid lookahead
                
                if roll_agg == 'mean':
                    result[f"{col}__roll_mean_{window}"] = shifted.rolling(
                        window=window, min_periods=1
                    ).mean()
                elif roll_agg == 'std':
                    result[f"{col}__roll_std_{window}"] = shifted.rolling(
                        window=window, min_periods=1
                    ).std()
                elif roll_agg == 'sum':
                    result[f"{col}__roll_sum_{window}"] = shifted.rolling(
                        window=window, min_periods=1
                    ).sum()
        
        return result
    
    if groupby_cols:
        # Apply per group
        df = df.groupby(groupby_cols, group_keys=False).apply(
            add_features_to_group
        ).reset_index(drop=True)
    else:
        df = add_features_to_group(df)
    
    return df


def add_date_features(
    df: pd.DataFrame,
    date_cols: Optional[List[str]] = None,
    features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Add date-based features from datetime columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    date_cols : list, optional
        Datetime columns. If None, auto-detect.
    features : list, optional
        Features to extract. Options: year, month, day, hour, weekday, 
        quarter, dayofyear, weekofyear, is_weekend, is_month_start, is_month_end.
        Default extracts all.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with date features added.
    """
    df = df.copy()
    
    default_features = [
        'year', 'month', 'day', 'weekday', 'quarter', 
        'dayofyear', 'is_weekend'
    ]
    
    if features is None:
        features = default_features
    
    if date_cols is None:
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    for col in date_cols:
        dt = pd.to_datetime(df[col], errors='coerce')
        
        if 'year' in features:
            df[f"{col}__year"] = dt.dt.year
        if 'month' in features:
            df[f"{col}__month"] = dt.dt.month
        if 'day' in features:
            df[f"{col}__day"] = dt.dt.day
        if 'hour' in features:
            df[f"{col}__hour"] = dt.dt.hour
        if 'weekday' in features:
            df[f"{col}__weekday"] = dt.dt.weekday
        if 'quarter' in features:
            df[f"{col}__quarter"] = dt.dt.quarter
        if 'dayofyear' in features:
            df[f"{col}__dayofyear"] = dt.dt.dayofyear
        if 'weekofyear' in features:
            df[f"{col}__weekofyear"] = dt.dt.isocalendar().week.astype(int)
        if 'is_weekend' in features:
            df[f"{col}__is_weekend"] = (dt.dt.weekday >= 5).astype(int)
        if 'is_month_start' in features:
            df[f"{col}__is_month_start"] = dt.dt.is_month_start.astype(int)
        if 'is_month_end' in features:
            df[f"{col}__is_month_end"] = dt.dt.is_month_end.astype(int)
    
    return df


def auto_feature_engineering(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    add_interactions: bool = True,
    max_interaction_pairs: int = 10,
    add_polynomial: bool = False,
    polynomial_degree: int = 2
) -> pd.DataFrame:
    """
    Automatically apply feature engineering based on data characteristics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_col : str, optional
        Target column to exclude.
    add_interactions : bool, default True
        Add interaction features.
    max_interaction_pairs : int, default 10
        Max interaction pairs.
    add_polynomial : bool, default False
        Add polynomial features.
    polynomial_degree : int, default 2
        Polynomial degree.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features.
    """
    df = df.copy()
    
    if target_col and target_col in df.columns:
        target = df[target_col]
        df = df.drop(columns=[target_col])
    else:
        target = None
    
    # Add date features for any detected datetime columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if datetime_cols:
        df = add_date_features(df, datetime_cols)
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if add_interactions and len(numeric_cols) >= 2:
        df = add_polynomial_interactions(
            df, 
            numeric_cols=numeric_cols[:min(5, len(numeric_cols))],  # Limit columns
            max_pairs=max_interaction_pairs
        )
    
    if add_polynomial and numeric_cols:
        df = add_polynomial_features(
            df,
            numeric_cols=numeric_cols[:min(3, len(numeric_cols))],  # Limit columns
            degree=polynomial_degree
        )
    
    if target is not None:
        df[target_col] = target
    
    return df