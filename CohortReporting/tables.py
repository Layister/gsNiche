from __future__ import annotations

import math

import numpy as np
import pandas as pd


TABLE_NOTES = {
    "object_overview": (
        "Atlas note: leading anchor is a readability grouping key only for object heatmaps; "
        "it does not define an object class or unique identity."
    ),
}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def encode_full_axis_strip(values: np.ndarray, axis_order: list[str]) -> str:
    if values.size == 0 or not axis_order:
        return ""
    return ";".join(f"{axis_id}={float(value):.3f}" for axis_id, value in zip(axis_order, values.tolist()))


def encode_supported_axes(values: np.ndarray, axis_order: list[str], threshold: float) -> str:
    if values.size == 0 or not axis_order:
        return ""
    kept = [axis_id for axis_id, value in zip(axis_order, values.tolist()) if float(value) >= float(threshold)]
    return ",".join(kept)


def derive_support_spread(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    clipped = np.clip(values.astype(float), 0.0, 1.0)
    total = float(clipped.sum())
    if total <= 0.0:
        return 0.0
    norm = clipped / total
    nonzero = norm[norm > 0]
    if nonzero.size == 0:
        return 0.0
    entropy = float(-np.sum(nonzero * np.log(nonzero)) / np.log(max(len(values), 2)))
    return float(np.clip(entropy, 0.0, 1.0))


def dataframe_to_markdown(df: pd.DataFrame, note: str | None = None) -> str:
    header = ""
    if note:
        header = f"> {note}\n\n"
    if df.empty:
        return header + "| empty |\n| --- |\n| no rows |\n"
    cols = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value).replace("\n", " ").replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return header + "\n".join(lines) + "\n"
