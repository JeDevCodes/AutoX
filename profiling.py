from typing import Tuple, Dict, Any, Optional
import os
import json
import io
import warnings
import pandas as pd
import numpy as np
try:
    import ydata_profiling  # type: ignore
    from ydata_profiling import ProfileReport  # type: ignore
    _HAS_YDATA = True
except Exception:
    _HAS_YDATA = False

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt


def _basic_schema(df: pd.DataFrame, max_top: int = 5) -> Dict[str, Any]:
    schema = {}
    for col in df.columns:
        ser = df[col]
        missing_pct = float(ser.isna().mean())
        nunique = int(ser.nunique(dropna=True))
        try:
            top_vals = ser.dropna().value_counts().head(max_top).to_dict()
        except Exception:
            top_vals = {}
        mem = int(ser.memory_usage(deep=True))
        
        if pd.api.types.is_datetime64_any_dtype(ser):
            guessed = 'datetime'
        elif pd.api.types.is_bool_dtype(ser):
            guessed = 'bool'
        elif pd.api.types.is_numeric_dtype(ser):
            guessed = 'numeric'
        elif pd.api.types.is_categorical_dtype(ser):
            guessed = 'categorical'
        elif pd.api.types.is_object_dtype(ser):
            try:
                parsed = pd.to_datetime(ser.dropna().iloc[:20], errors='coerce')
                if parsed.notna().sum() >= 8:
                    guessed = 'datetime'
                else:
                    guessed = 'text' if nunique > 50 else 'categorical'
            except Exception:
                guessed = 'categorical'
        else:
            guessed = str(ser.dtype)

        schema[col] = {
            'dtype': guessed,
            'missing_pct': missing_pct,
            'nunique': nunique,
            'top_values': top_vals,
            'memory_bytes': mem
        }
    return schema


def _plot_missingness(df: pd.DataFrame, out_path: str, max_cols: int = 80):
    mask = (~df.isna()).astype(int)
    if mask.shape[1] > max_cols:
        # pick top max_cols by non-missing ratio
        col_present = mask.mean(axis=0).sort_values(ascending=False)
        selected = col_present.index[:max_cols].tolist()
        mask = mask[selected]

    fig, ax = plt.subplots(figsize=(min(20, 0.2 * mask.shape[1] + 2), 6))
    ax.imshow(mask.T, aspect='auto', interpolation='nearest')
    ax.set_yticks(range(len(mask.columns)))
    ax.set_yticklabels(mask.columns)
    ax.set_xlabel("rows")
    ax.set_title("Missingness matrix (white=present, dark=missing)")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def _plot_corr_heatmap(df: pd.DataFrame, out_path: str, max_vars: int = 40):
    """
    Compute correlations for numeric columns and save a heatmap PNG.
    Limits to max_vars numeric variables (highest variance).
    """
    numeric = df.select_dtypes(include=['number'])
    if numeric.shape[1] == 0:
        # nothing to plot
        return None
    # limit variables
    if numeric.shape[1] > max_vars:
        # select top vars by variance
        variances = numeric.var().sort_values(ascending=False)
        selected = variances.index[:max_vars].tolist()
        numeric = numeric[selected]
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(min(14, 0.3 * corr.shape[1] + 2), min(14, 0.3 * corr.shape[0] + 2)))
    cax = ax.matshow(corr, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, ha='left', fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title("Numeric correlation matrix")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def _simple_html_report(schema: Dict[str, Any], df: pd.DataFrame, out_dir: str, html_name: str = "profile.html") -> str:
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, html_name)
    rows = df.shape[0]
    cols = df.shape[1]
    mem = int(df.memory_usage(deep=True).sum())
    
    schema_rows = []
    for col, info in schema.items():
        top = ", ".join([f"{k} ({v})" for k, v in info.get('top_values', {}).items()]) if info.get('top_values') else ''
        schema_rows.append((col, info.get('dtype'), f"{info.get('missing_pct'):.3f}", info.get('nunique'), top))


    missing_fn = os.path.join(out_dir, "missingness.png")
    try:
        _plot_missingness(df, missing_fn)
    except Exception as e:
        warnings.warn(f"Failed to create missingness plot: {e}")
        missing_fn = None

    corr_fn = os.path.join(out_dir, "corr_heatmap.png")
    try:
        corr_res = _plot_corr_heatmap(df, corr_fn)
        if corr_res is None:
            corr_fn = None
    except Exception as e:
        warnings.warn(f"Failed to create correlation heatmap: {e}")
        corr_fn = None


    html = io.StringIO()
    html.write("<html><head><meta charset='utf-8'><title>AutoX Profiling Report</title></head><body>")
    html.write(f"<h1>AutoX Profiling Report</h1>")
    html.write(f"<p><b>Rows:</b> {rows} &nbsp; <b>Columns:</b> {cols} &nbsp; <b>Total memory (bytes):</b> {mem}</p>")
    html.write("<h2>Schema</h2>")
    html.write("<table border='1' cellpadding='4' cellspacing='0'>")
    html.write("<tr><th>column</th><th>dtype_guess</th><th>missing_pct</th><th>nunique</th><th>top_values</th></tr>")
    for r in schema_rows:
        html.write("<tr>")
        html.write(f"<td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{r[4]}</td>")
        html.write("</tr>")
    html.write("</table>")

    if missing_fn:
        html.write("<h2>Missingness</h2>")
        html.write(f"<img src='{os.path.basename(missing_fn)}' style='max-width:100%;height:auto;'>")

    if corr_fn:
        html.write("<h2>Numeric Correlation</h2>")
        html.write(f"<img src='{os.path.basename(corr_fn)}' style='max-width:100%;height:auto;'>")


    html.write("<h2>Data preview (top 10 rows)</h2>")
    html.write(df.head(10).to_html(index=False, escape=False))

    html.write("</body></html>")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html.getvalue())

    return html_path


def profile_dataframe(df: pd.DataFrame,
                      out_dir: Optional[str] = "artifacts/profile",
                      sample_frac: Optional[float] = None,
                      use_ydata: bool = True,
                      html_name: str = "profile.html") -> Tuple[Dict[str, Any], Dict[str, str]]:
    
    os.makedirs(out_dir or ".", exist_ok=True)

    if sample_frac is not None and 0 < sample_frac < 1.0:
        df_sample = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
    else:
        df_sample = df.copy()

    # canonical schema (compact)
    schema = _basic_schema(df_sample)

    artifacts = {}

    # Save schema JSON
    schema_path = os.path.join(out_dir, "schema.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    artifacts['schema_json'] = schema_path

    # If ydata_profiling available and requested -> produce full html
    if use_ydata and _HAS_YDATA:
        try:
            pr = ProfileReport(df_sample, title="AutoX Profiling Report", minimal=False)
            html_path = os.path.join(out_dir, html_name)
            pr.to_file(html_path)
            artifacts['html'] = html_path
            return schema, artifacts
        except Exception as e:
            warnings.warn(f"ydata_profiling failed, falling back to simple report: {e}")

    # Fallback: simple html + png plots
    html_path = _simple_html_report(schema, df_sample, out_dir, html_name=html_name)
    artifacts['html'] = html_path
    # include any pngs we created (missingness, corr)
    missing_fn = os.path.join(out_dir, "missingness.png")
    if os.path.exists(missing_fn):
        artifacts['missingness_png'] = missing_fn
    corr_fn = os.path.join(out_dir, "corr_heatmap.png")
    if os.path.exists(corr_fn):
        artifacts['corr_png'] = corr_fn

    return schema, artifacts


def save_schema_json(schema: Dict[str, Any], path: str):
    """
    Save the schema dict to path (JSON).
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    return path
