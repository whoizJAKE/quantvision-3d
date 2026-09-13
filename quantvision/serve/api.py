from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


from quantvision.config import load_settings
from quantvision.etl.db import fetchall, insert_many, latest_books, session
from quantvision.etl.normalize import utc_now
from quantvision.monitor.alerts import run_monitor
from quantvision.etl.pipeline import run_etl

WEB_DIR = Path(__file__).resolve().parents[2] / "quantvisionUI"


def create_app() -> FastAPI:
    settings = load_settings()
    app = FastAPI(title="QuantVision 3D Live", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins + ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        exists = settings.db_path.exists()
        return {
            "ok": True,
            "db": str(settings.db_path),
            "db_exists": exists,
            "generated_at": utc_now(),
        }

    @app.get("/api/markets")
    def markets(venue: str | None = None, major: bool = False, limit: int = 80):
        sql = "SELECT * FROM markets"
        clauses = []
        params: list = []
        if venue:
            clauses.append("venue=?")
            params.append(venue)
        if major:
            clauses.append("is_major=1")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY COALESCE(volume_24h, volume, liquidity, 0) DESC LIMIT ?"
        params.append(limit)
        with session(settings.db_path) as conn:
            rows = fetchall(conn, sql, tuple(params))
            books = latest_books(conn)
            latest_pred = fetchall(
                conn,
                """
                SELECT p.* FROM predictions p
                JOIN (
                    SELECT market_id, MAX(prediction_id) AS max_id
                    FROM predictions GROUP BY market_id
                ) t ON p.prediction_id = t.max_id
                """,
            )
        preds = {row["market_id"]: row for row in latest_pred}
        for row in rows:
            book = books.get(row["market_id"])
            pred = preds.get(row["market_id"])
            row["orderbook"] = book
            row["prediction"] = pred
            if pred and pred.get("market_prob") is not None:
                row["deviation"] = pred.get("deviation")
        return {"markets": rows, "generated_at": utc_now()}

    @app.get("/api/trades")
    def trades(market_id: str | None = None, limit: int = 120):
        limit = max(1, min(limit, 500))
        with session(settings.db_path) as conn:
            if market_id:
                rows = fetchall(
                    conn,
                    "SELECT * FROM trades WHERE market_id=? ORDER BY traded_at DESC LIMIT ?",
                    (market_id, limit),
                )
            else:
                rows = fetchall(conn, "SELECT * FROM trades ORDER BY traded_at DESC LIMIT ?", (limit,))
        return {"trades": rows}

    @app.get("/api/orderbook")
    def orderbook(market_id: str = Query(...)):
        with session(settings.db_path) as conn:
            rows = fetchall(
                conn,
                "SELECT * FROM orderbook_snapshots WHERE market_id=? ORDER BY snapshot_id DESC LIMIT 1",
                (market_id,),
            )
        return {"orderbook": rows[0] if rows else None}

    @app.get("/api/alerts")
    def alerts(limit: int = 50):
        with session(settings.db_path) as conn:
            rows = fetchall(conn, "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,))
        return {"alerts": rows}

    @app.get("/api/performance")
    def performance(limit: int = 200):
        with session(settings.db_path) as conn:
            rows = fetchall(
                conn,
                "SELECT * FROM performance_log ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            summary = fetchall(
                conn,
                """
                SELECT
                    COUNT(*) AS n,
                    AVG(abs_deviation) AS mean_abs_dev,
                    AVG(deviation) AS mean_dev,
                    AVG(brier) AS mean_brier
                FROM performance_log
                """,
            )
        return {"rows": rows, "summary": summary[0] if summary else {}}

    @app.post("/api/predictions")
    def add_prediction(payload: dict):
        row = {
            "market_id": payload["market_id"],
            "model_name": payload.get("model_name", "manual"),
            "predicted_prob": float(payload["predicted_prob"]),
            "market_prob": payload.get("market_prob"),
            "deviation": payload.get("deviation"),
            "created_at": utc_now(),
        }
        with session(settings.db_path) as conn:
            insert_many(conn, "predictions", [row])
        return {"ok": True, "prediction": row}

    @app.post("/api/etl")
    def trigger_etl():
        return run_etl(settings)

    @app.post("/api/monitor")
    def trigger_monitor():
        return run_monitor(settings)

    @app.get("/api/snapshot")
    def snapshot():
        with session(settings.db_path) as conn:
            from quantvision.etl.pipeline import export_snapshot

            path = export_snapshot(settings, conn)
        return FileResponse(path, media_type="application/json")

    if WEB_DIR.exists():
        @app.get("/")
        def index():
            return FileResponse(WEB_DIR / "index.html")

        @app.get("/styles.css")
        def styles():
            return FileResponse(WEB_DIR / "styles.css", media_type="text/css")

        @app.get("/app.js")
        def script():
            return FileResponse(WEB_DIR / "app.js", media_type="application/javascript")

        @app.get("/snapshot.json")
        def static_snapshot():
            snap = WEB_DIR / "snapshot.json"
            if snap.exists():
                return FileResponse(snap, media_type="application/json")
            return {"generated_at": utc_now(), "markets": [], "trades": [], "alerts": [], "performance": []}

    return app


def main() -> None:
    import uvicorn

    settings = load_settings()
    uvicorn.run(
        "quantvision.serve.api:create_app",
        factory=True,
        host=settings.web_host,
        port=settings.web_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
