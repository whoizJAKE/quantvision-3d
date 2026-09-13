# QuantVision UI

Standalone live dashboard for QuantVision 3D. It is a document in this repo, not part of the personal homepage.

Intended public URL, when you choose to publish it:

`https://jacobhmaiddout.com/quantvisionUI`

## Run locally

From the QuantVision-3D repo root:

```bash
python scripts/run_live.py
```

Open [http://127.0.0.1:8787](http://127.0.0.1:8787). That serves this folder.

Without the API, open `index.html` in a browser. It will try public Polymarket data, then `snapshot.json` if present.

## Publish later

Do **not** add this to the homepage nav. Copy the folder as a site path:

```bash
cp -R quantvisionUI /path/to/whoizjake.github.io/quantvisionUI
```

GitHub Pages will then serve:

- `https://jacobhmaiddout.com/quantvisionUI`
- `https://jacobhmaiddout.com/quantvisionUI/index.html`

Keep `index.html`, `app.js`, `styles.css`, and optionally `snapshot.json` together. Relative asset paths are already set for that layout.

## What the page shows

- Normalized Kalshi + Polymarket markets (Kalshi only through the local API)
- Order book and trade tape
- App probability vs market consensus
- Deviation alerts (5% on major events, 10% otherwise)

Research / paper trading only. Not financial advice.
