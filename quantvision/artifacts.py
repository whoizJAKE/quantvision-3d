from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch

from quantvision.models import MarketTransformer


def artifact_dir(root: Path, ticker: str, interval: str) -> Path:
    safe = ticker.replace("/", "-").replace(" ", "")
    path = root / f"{safe}_{interval}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_training_run(
    root: Path,
    ticker: str,
    interval: str,
    scaler,
    logistic,
    boosting,
    transformer: MarketTransformer,
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
    meta: dict[str, Any],
) -> Path:
    folder = artifact_dir(root, ticker, interval)
    joblib.dump(scaler, folder / "scaler.joblib")
    joblib.dump(logistic, folder / "logistic.joblib")
    joblib.dump(boosting, folder / "boosting.joblib")
    torch.save(transformer.state_dict(), folder / "transformer.pt")
    (folder / "transformer_meta.json").write_text(
        json.dumps({"n_features": int(transformer.project.in_features)}, indent=2)
    )
    predictions.to_csv(folder / "predictions.csv")
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    return folder


def load_latest_ensemble(root: Path, ticker: str) -> dict[str, Any] | None:
    if not root.exists():
        return None
    safe = ticker.replace("/", "-").replace(" ", "")
    candidates = sorted(root.glob(f"{safe}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for folder in candidates:
        pred_path = folder / "predictions.csv"
        metrics_path = folder / "metrics.json"
        if not pred_path.exists():
            continue
        preds = pd.read_csv(pred_path, index_col=0)
        if preds.empty or "ensemble" not in preds.columns:
            continue
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        latest = float(preds["ensemble"].iloc[-1])
        return {
            "ticker": ticker,
            "path": str(folder),
            "latest_ensemble": latest,
            "predictions": preds,
            "metrics": metrics,
        }
    return None
