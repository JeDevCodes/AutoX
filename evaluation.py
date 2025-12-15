from typing import Dict, Any, Optional
import pandas as pd
import os
import numpy as np

def scoring_name(task: str) -> str:
    if task == 'classification':
        return 'f1_macro'
    return 'r2'

def build_leaderboard(candidate_summary: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for k, v in candidate_summary.items():
        rows.append({
            'model': k,
            'cv_score': float(v.get('cv_score', np.nan)),
            'best_params': v.get('best_params')
        })
    df = pd.DataFrame(rows).sort_values('cv_score', ascending=False).reset_index(drop=True)
    return df

def save_leaderboard_html(df: pd.DataFrame, out_path: str, title: str = "AutoX Leaderboard") -> str:
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    html = f"<html><head><meta charset='utf-8'><title>{title}</title></head><body>"
    html += f"<h1>{title}</h1>"
    html += df.to_html(index=False, escape=False)
    html += "</body></html>"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
