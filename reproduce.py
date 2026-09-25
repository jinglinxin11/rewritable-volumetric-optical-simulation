"""Numerical summary utilities retained for source-snapshot compatibility."""
import json
import numpy as np


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def mean_std(rows, key):
    values = [r[key] for r in rows]
    return {"mean": float(np.mean(values)), "std_population": float(np.std(values)),
            "std_sample": float(np.std(values, ddof=1)) if len(values) > 1 else None}
