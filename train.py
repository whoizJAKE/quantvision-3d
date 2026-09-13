from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from quantvision.artifacts import save_training_run
from quantvision.data import load_ohlcv
from quantvision.features import FEATURES, build_features, make_sequences
from quantvision.models import fit_classical, fit_transformer, transformer_proba


def _safe_auc(y_true, y_prob) -> float:
    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def run_training(
    ticker: str = "BTC-USD",
    period: str = "730d",
    interval: str = "1h",
    horizon: int = 12,
    epochs: int = 20,
    lookback: int = 64,
    artifacts_dir: str | Path = "artifacts",
):
    raw = load_ohlcv(ticker, period, interval)
    frame = build_features(raw, horizon)
    sequences, labels, index = make_sequences(frame, FEATURES, lookback)
    if len(labels) < 300:
        raise ValueError("Not enough usable rows; choose a longer period or coarser interval.")

    n = len(labels)
    train_end, valid_end = int(n * 0.6), int(n * 0.8)
    scaler = StandardScaler().fit(sequences[:train_end].reshape(-1, len(FEATURES)))
    scaled = scaler.transform(sequences.reshape(-1, len(FEATURES))).reshape(sequences.shape)
    last_step = scaled[:, -1, :]

    logistic, boosting = fit_classical(last_step[:train_end], labels[:train_end])
    transformer = fit_transformer(
        scaled[:train_end],
        labels[:train_end],
        scaled[train_end:valid_end],
        labels[train_end:valid_end],
        epochs=epochs,
    )

    test = slice(valid_end, None)
    predictions = pd.DataFrame(index=index[test])
    predictions["logistic"] = logistic.predict_proba(last_step[test])[:, 1]
    predictions["boosting"] = boosting.predict_proba(last_step[test])[:, 1]
    predictions["transformer"] = transformer_proba(transformer, scaled[test])
    predictions["ensemble"] = predictions[["logistic", "boosting", "transformer"]].mean(axis=1)

    truth = labels[test]
    metrics = {}
    for name in predictions.columns:
        prob = predictions[name].to_numpy()
        metrics[name] = {
            "accuracy": float(accuracy_score(truth, (prob >= 0.5).astype(int))),
            "auc": _safe_auc(truth, prob),
            "brier": float(brier_score_loss(truth, prob)),
        }

    meta = {
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "horizon": horizon,
        "lookback": lookback,
        "rows": int(n),
        "test_rows": int(len(predictions)),
        "latest_ensemble": float(predictions["ensemble"].iloc[-1]),
    }
    folder = save_training_run(
        Path(artifacts_dir),
        ticker,
        interval,
        scaler,
        logistic,
        boosting,
        transformer,
        predictions,
        metrics,
        meta,
    )
    return {
        "predictions": predictions,
        "metrics": metrics,
        "data": frame,
        "raw": raw,
        "artifact_dir": str(folder),
        "meta": meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QuantVision 3D models")
    parser.add_argument("--ticker", default="BTC-USD")
    parser.add_argument("--period", default="730d")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lookback", type=int, default=64)
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    result = run_training(
        ticker=args.ticker,
        period=args.period,
        interval=args.interval,
        horizon=args.horizon,
        epochs=args.epochs,
        lookback=args.lookback,
        artifacts_dir=args.artifacts,
    )
    print(json.dumps({"metrics": result["metrics"], "meta": result["meta"]}, indent=2))


if __name__ == "__main__":
    main()
