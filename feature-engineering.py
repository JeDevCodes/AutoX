
from typing import List, Optional, Union
import pandas as pd
import numpy as np

def add_polynomial_interactions(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None,
                                max_pairs: Optional[int] = None, drop_original: bool = False) -> pd.DataFrame:
    """
    Add pairwise product interactions between numeric_cols.
    numeric_cols: list of numeric column names; if None, infer numeric columns.
    max_pairs: if set, limit to the first max_pairs interactions (deterministic order)
    drop_original: if True, drop original numeric cols after creating interactions
    """
    df = df.copy()
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            pairs.append((numeric_cols[i], numeric_cols[j]))
    if max_pairs:
        pairs = pairs[:max_pairs]
    for a, b in pairs:
        new_col = f"{a}__x__{b}"
        # element-wise product; preserve dtype
        df[new_col] = df[a].astype(float) * df[b].astype(float)
    if drop_original:
        df = df.drop(columns=numeric_cols)
    return df


def add_polynomial_features(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None, degree: int = 2,
                            include_bias: bool = False, drop_original: bool = False) -> pd.DataFrame:
    """
    Add polynomial features for numeric columns using sklearn's PolynomialFeatures
    """
    from sklearn.preprocessing import PolynomialFeatures
    df = df.copy()
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        return df
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias, interaction_only=False)
    arr = poly.fit_transform(df[numeric_cols].astype(float).values)
    names = poly.get_feature_names_out(numeric_cols)
    poly_df = pd.DataFrame(arr, columns=names, index=df.index)
    # remove original columns if desired
    if drop_original:
        base = df.drop(columns=numeric_cols)
    else:
        base = df
    # avoid duplicating same column names; suffix poly_
    poly_df = poly_df.loc[:, ~poly_df.columns.isin(base.columns)]
    # prefix or keep names; to keep safe, prefix with poly__
    poly_df.columns = [f"poly__{c}" for c in poly_df.columns]
    df_out = pd.concat([base, poly_df], axis=1)
    return df_out


# def expand_datetime_features(df: pd.DataFrame, ts_cols: Optional[List[str]] = None,
#                              drop_original: bool = True) -> pd.DataFrame:
#     df = df.copy()
#     if ts_cols is None:
#         ts_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or df[c].dtype == object]
#         # filter heuristically by parsing first few non-null values
#         candidates = []
#         for c in ts_cols:
#             ser = df[c].dropna()
#             if ser.empty:
#                 continue
#             try:
#                 parsed = pd.to_datetime(ser.iloc[:20], errors='coerce')
#                 if parsed.notna().sum() >= 5:
#                     candidates.append(c)
#             except Exception:
#                 continue
#         ts_cols = candidates
#     for c in ts_cols:
#         ser = pd.to_datetime(df[c], errors='coerce')
#         df[f"{c}__year"] = ser.dt.year
#         df[f"{c}__month"] = ser.dt.month
#         df[f"{c}__day"] = ser.dt.day
#         df[f"{c}__hour"] = ser.dt.hour
#         df[f"{c}__weekday"] = ser.dt.weekday
#         df[f"{c}__is_weekend"] = ser.dt.weekday.isin([5,6]).astype(int)
#     if drop_original:
#         df = df.drop(columns=ts_cols, errors='ignore')
#     return df


def add_lag_features(df: pd.DataFrame, groupby_cols: Optional[List[str]] = None,
                     value_cols: Optional[List[str]] = None, lags: Optional[List[int]] = [1],
                     roll_windows: Optional[List[int]] = None, time_col: Optional[str] = None) -> pd.DataFrame:
    
    df = df.copy()
    if value_cols is None:
        value_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if groupby_cols is None:
        groupby_cols = []
        
    if groupby_cols and time_col:
        df = df.sort_values(groupby_cols + [time_col])
    elif time_col:
        df = df.sort_values(time_col)

    if groupby_cols:
        gb = df.groupby(groupby_cols, group_keys=False)
    # else:
    #     gb = [(None, df)]

    out = []
    if groupby_cols:
        for name, sub in gb:
            s = sub.copy()
            for col in value_cols:
                for lag in (lags or []):
                    s[f"{col}__lag_{lag}"] = s[col].shift(lag)
                for w in (roll_windows or []):
                    s[f"{col}__roll_mean_{w}"] = s[col].shift(1).rolling(window=w, min_periods=1).mean()
            out.append(s)
        df_out = pd.concat(out, axis=0).sort_index()
    else:
        s = df.copy()
        for col in value_cols:
            for lag in (lags or []):
                s[f"{col}__lag_{lag}"] = s[col].shift(lag)
            for w in (roll_windows or []):
                s[f"{col}__roll_mean_{w}"] = s[col].shift(1).rolling(window=w, min_periods=1).mean()
        df_out = s
    return df_out
