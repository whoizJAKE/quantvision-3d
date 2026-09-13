const state = {
  apiBase: localStorage.getItem("qv_api") || "http://127.0.0.1:8787",
  mode: "connecting",
  markets: [],
  trades: [],
  alerts: [],
  performance: [],
  selected: null,
};

const $ = (id) => document.getElementById(id);

function fmtPct(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "—";
  return `${(Number(v) * 100).toFixed(1)}%`;
}
function fmtNum(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "—";
  const n = Number(v);
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return n.toFixed(2);
}
function clsDev(v) {
  if (v === null || v === undefined) return "";
  return Number(v) >= 0 ? "pos" : "neg";
}

async function fetchJson(url, opts, timeoutMs = 12000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...opts, signal: ctrl.signal });
    if (!res.ok) throw new Error(`${res.status} ${url}`);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

function wantLocalApi() {
  const host = location.hostname;
  if (host === "127.0.0.1" || host === "localhost") return true;
  return localStorage.getItem("qv_force_api") === "1";
}

function parseJsonish(value) {
  if (typeof value !== "string") return value;
  try { return JSON.parse(value); } catch { return value; }
}

function setStatus(mode, label) {
  state.mode = mode;
  const el = $("sourcePill");
  el.textContent = label;
  el.className = "pill " + (mode === "api" ? "ok" : mode === "public" ? "warn" : "err");
}

async function loadFromApi() {
  const base = state.apiBase.replace(/\/$/, "");
  const timeout = wantLocalApi() ? 12000 : 800;
  const [markets, trades, alerts, perf] = await Promise.all([
    fetchJson(`${base}/api/markets?limit=80`, undefined, timeout),
    fetchJson(`${base}/api/trades?limit=120`, undefined, timeout),
    fetchJson(`${base}/api/alerts?limit=40`, undefined, timeout),
    fetchJson(`${base}/api/performance?limit=120`, undefined, timeout),
  ]);
  state.markets = markets.markets || [];
  state.trades = trades.trades || [];
  state.alerts = alerts.alerts || [];
  state.performance = perf.rows || [];
  $("updatedPill").textContent = `updated ${new Date().toLocaleTimeString()}`;
  setStatus("api", `local API ${base}`);
}

function polyToMarket(raw) {
  const tokens = parseJsonish(raw.clobTokenIds) || [];
  const prices = parseJsonish(raw.outcomePrices) || [];
  const last = prices.length ? Number(prices[0]) : null;
  return {
    market_id: `polymarket:${raw.conditionId}`,
    venue: "polymarket",
    venue_market_id: raw.conditionId,
    title: raw.question || raw.slug,
    question: raw.question,
    ticker: raw.slug,
    last_price: last,
    mid_price: last,
    volume_24h: Number(raw.volume24hr || raw.volumeNum || raw.volume || 0),
    liquidity: Number(raw.liquidityNum || raw.liquidity || 0),
    yes_token_id: tokens[0] || null,
    no_token_id: tokens[1] || null,
    is_major: Number(raw.volume24hr || 0) > 100000 ? 1 : 0,
    status: raw.closed ? "closed" : "open",
  };
}

async function loadPublicPolymarket() {
  const events = await fetchJson("https://gamma-api.polymarket.com/markets?closed=false&active=true&limit=30&order=volume24hr&ascending=false");
  const raw = Array.isArray(events) ? events : (events.markets || []);
  state.markets = raw.map(polyToMarket);
  const first = state.markets[0];
  if (first) {
    const trades = await fetchJson(`https://data-api.polymarket.com/trades?limit=80&takerOnly=true&market=${encodeURIComponent(first.venue_market_id)}`);
    state.trades = (Array.isArray(trades) ? trades : []).map((t) => ({
      market_id: first.market_id,
      venue: "polymarket",
      traded_at: new Date((t.timestamp > 1e12 ? t.timestamp : t.timestamp * 1000)).toISOString(),
      price: t.outcome && String(t.outcome).toLowerCase() === "no" ? 1 - Number(t.price) : Number(t.price),
      size: Number(t.size),
      side: t.side,
      outcome: t.outcome,
      title: t.title,
    }));
    if (first.yes_token_id) {
      try {
        const book = await fetchJson(`https://clob.polymarket.com/book?token_id=${encodeURIComponent(first.yes_token_id)}`);
        first.orderbook = {
          bids_json: JSON.stringify((book.bids || []).map((l) => [Number(l.price), Number(l.size)])),
          asks_json: JSON.stringify((book.asks || []).map((l) => [Number(l.price), Number(l.size)])),
          mid: book.last_trade_price ? Number(book.last_trade_price) : first.mid_price,
        };
      } catch {}
    }
  }
  state.alerts = [];
  state.performance = [];
  $("updatedPill").textContent = `updated ${new Date().toLocaleTimeString()}`;
  setStatus("public", "public Polymarket APIs");
}

async function loadSnapshot() {
  const urls = ["./snapshot.json", "./quantvision-snapshot.json", "/quantvision-snapshot.json"];
  for (const url of urls) {
    try {
      const snap = await fetchJson(url);
      state.markets = snap.markets || [];
      state.trades = snap.trades || [];
      state.alerts = snap.alerts || [];
      state.performance = snap.performance || [];
      $("updatedPill").textContent = `snapshot ${snap.generated_at || ""}`;
      setStatus("public", "static snapshot");
      return true;
    } catch {}
  }
  return false;
}

async function refresh() {
  try {
    if (wantLocalApi() || localStorage.getItem("qv_api")) {
      await loadFromApi();
    } else {
      throw new Error("skip local api");
    }
  } catch {
    try {
      await loadPublicPolymarket();
    } catch {
      const ok = await loadSnapshot();
      if (!ok) setStatus("err", "no live source");
    }
  }
  if (!state.selected && state.markets.length) state.selected = state.markets[0].market_id;
  render();
}

function filteredMarkets() {
  const q = ($("q")?.value || "").toLowerCase();
  const venue = $("venue")?.value || "all";
  return state.markets.filter((m) => {
    if (venue !== "all" && m.venue !== venue) return false;
    if (!q) return true;
    return `${m.title} ${m.question} ${m.ticker} ${m.venue}`.toLowerCase().includes(q);
  });
}

function renderKpis() {
  $("kpiMarkets").textContent = String(state.markets.length);
  $("kpiTrades").textContent = String(state.trades.length);
  $("kpiAlerts").textContent = String(state.alerts.length);
  const abs = state.performance.map((r) => Number(r.abs_deviation)).filter((n) => !Number.isNaN(n));
  $("kpiDev").textContent = abs.length ? fmtPct(abs.reduce((a, b) => a + b, 0) / abs.length) : "—";
}

function renderMarkets() {
  const body = $("marketBody");
  body.innerHTML = "";
  for (const m of filteredMarkets()) {
    const pred = m.prediction || {};
    const dev = pred.deviation ?? m.deviation;
    const tr = document.createElement("tr");
    if (m.market_id === state.selected) tr.className = "active";
    tr.innerHTML = `
      <td>${m.venue}</td>
      <td>${m.title || m.question || m.market_id}${m.is_major ? " · major" : ""}</td>
      <td>${fmtPct(m.mid_price ?? m.last_price)}</td>
      <td>${fmtPct(pred.predicted_prob)}</td>
      <td class="${clsDev(dev)}">${dev === undefined || dev === null ? "—" : fmtPct(dev)}</td>
      <td>${fmtNum(m.volume_24h || m.volume)}</td>`;
    tr.onclick = () => { state.selected = m.market_id; render(); };
    body.appendChild(tr);
  }
}

function renderBookAndTape() {
  const market = state.markets.find((m) => m.market_id === state.selected) || state.markets[0];
  $("selectedTitle").textContent = market ? (market.title || market.market_id) : "No market selected";
  const book = market?.orderbook;
  const bids = parseJsonish(book?.bids_json) || [];
  const asks = parseJsonish(book?.asks_json) || [];
  $("bids").innerHTML = bids.slice(0, 12).map((l) => `<div class="lvl pos"><span>${fmtPct(l[0])}</span><span>${fmtNum(l[1])}</span></div>`).join("") || '<div class="note">No bid levels</div>';
  $("asks").innerHTML = asks.slice(0, 12).map((l) => `<div class="lvl neg"><span>${fmtPct(l[0])}</span><span>${fmtNum(l[1])}</span></div>`).join("") || '<div class="note">No ask levels</div>';
  const tape = state.trades.filter((t) => !market || t.market_id === market.market_id).slice(0, 30);
  $("tape").innerHTML = tape.map((t) => (
    `<tr><td>${(t.traded_at || "").replace("T", " ").replace("Z", "")}</td><td>${fmtPct(t.price)}</td><td>${fmtNum(t.size)}</td><td>${t.side || t.outcome || ""}</td></tr>`
  )).join("") || '<tr><td colspan="4" class="note">No trades yet</td></tr>';
}

function renderAlerts() {
  $("alertBody").innerHTML = (state.alerts || []).slice(0, 20).map((a) => (
    `<tr><td>${a.created_at || ""}</td><td>${a.severity || ""}</td><td>${a.title || a.market_id}</td><td class="${clsDev(a.deviation)}">${fmtPct(a.deviation)}</td></tr>`
  )).join("") || '<tr><td colspan="4" class="note">No alerts. Deviations under threshold are logged silently for backtests.</td></tr>';
}

function render() {
  renderKpis();
  renderMarkets();
  renderBookAndTape();
  renderAlerts();
}

function bind() {
  $("q").addEventListener("input", render);
  $("venue").addEventListener("change", render);
  $("refreshBtn").addEventListener("click", refresh);
  $("apiBase").value = state.apiBase;
  $("saveApi").addEventListener("click", () => {
    state.apiBase = $("apiBase").value.trim() || "http://127.0.0.1:8787";
    localStorage.setItem("qv_api", state.apiBase);
    localStorage.setItem("qv_force_api", "1");
    refresh();
  });
}

bind();
refresh();
setInterval(refresh, 8000);
