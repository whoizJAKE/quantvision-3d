#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantvision.config import load_settings
from quantvision.monitor.alerts import run_monitor


def main() -> None:
    parser = argparse.ArgumentParser(description="Alert when model probability diverges from market consensus")
    parser.add_argument("--config", default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval", type=int, default=None)
    args = parser.parse_args()
    settings = load_settings(args.config)
    interval = args.interval or settings.poll_seconds
    while True:
        result = run_monitor(settings)
        print(json.dumps(result, indent=2))
        if not args.loop:
            break
        time.sleep(max(5, interval))


if __name__ == "__main__":
    main()
