from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class HttpError(RuntimeError):
    def __init__(self, status: int, url: str, body: str):
        super().__init__(f"HTTP {status} for {url}: {body[:300]}")
        self.status = status
        self.url = url
        self.body = body


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 20.0,
    retries: int = 3,
) -> Any:
    if params:
        filtered = {k: v for k, v in params.items() if v is not None and v != ""}
        query = urllib.parse.urlencode(filtered, doseq=True)
        url = f"{url}?{query}" if query else url
    request_headers = {"User-Agent": "QuantVision-3D/0.2", "Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=request_headers, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read().decode("utf-8")
                if not payload:
                    return {}
                return json.loads(payload)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code in {429, 500, 502, 503, 504} and attempt < retries - 1:
                time.sleep(0.6 * (attempt + 1))
                last_error = HttpError(exc.code, url, body)
                continue
            raise HttpError(exc.code, url, body) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(0.6 * (attempt + 1))
                continue
            raise
    raise last_error or RuntimeError(f"Failed GET {url}")
