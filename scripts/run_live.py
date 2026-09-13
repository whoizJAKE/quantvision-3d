#!/usr/bin/env python3
"""ETL + monitor loop, then serve the live dashboard."""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantvision.config import load_settings
from quantvision.etl.pipeline import run_etl
from quantvision.monitor.alerts import run_monitor


def _loop(settings, interval: int, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            etl = run_etl(settings)
            print("etl", json.dumps(etl))
            mon = run_monitor(settings)
            print("monitor", json.dumps(mon))
        except Exception as exc:
            print(f"live loop error: {exc}", file=sys.stderr)
        stop.wait(max(5, interval))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ETL/monitor loop and the live web API")
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-server", action="store_true")
    parser.add_argument("--interval", type=int, default=None)
    args = parser.parse_args()
    settings = load_settings(args.config)
    interval = args.interval or settings.poll_seconds
    stop = threading.Event()
    worker = threading.Thread(target=_loop, args=(settings, interval, stop), daemon=True)
    worker.start()
    if args.no_server:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            stop.set()
        return
    import uvicorn

    from quantvision.serve.api import create_app

    try:
        uvicorn.run(create_app(), host=settings.web_host, port=settings.web_port, log_level="info")
    finally:
        stop.set()


if __name__ == "__main__":
    main()
