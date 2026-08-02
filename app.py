"""
dashboard_app.py
----------------
Page 2 — Analysis Dashboard: Overview, Results, Plots, Privacy Analysis.
Reads existing logs/ and plots/ produced by your training scripts —
zero changes required to any training code.

Install:  pip install flask
Run:      python dashboard_app.py
Open:     http://127.0.0.1:5050
"""

import os
import json
import glob
import base64
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

LOG_DIR   = "logs"
PLOTS_DIR = "plots"


# ── helpers ──────────────────────────────────────────────────────────────────

def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def img_b64(path):
    """Return a base64 data-URI for a PNG so the browser can embed it."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/api/results")
def api_results():
    records = []
    for fpath in sorted(glob.glob(os.path.join(LOG_DIR, "*.json"))):
        if os.path.basename(fpath) in ("privacy_report.json",):
            continue
        rec = read_json(fpath)
        if rec and "method" in rec:
            records.append(rec)
    return jsonify(records)


@app.route("/api/privacy")
def api_privacy():
    p = read_json(os.path.join(LOG_DIR, "privacy_report.json"))
    return jsonify(p or {})


@app.route("/api/plots")
def api_plots():
    plot_names = [
        "summary_grid", "accuracy", "f1", "time", "comm",
        "radar", "per_client_accuracy", "per_client_f1",
        "ier_bar", "reconstruction_loss", "gradient_risk",
        "privacy_summary_table",
    ]
    out = {}
    for name in plot_names:
        uri = img_b64(os.path.join(PLOTS_DIR, f"{name}.png"))
        if uri:
            out[name] = uri
    return jsonify(out)


@app.route("/api/status")
def api_status():
    methods  = ["normal_split", "u_shape"]
    machines = ["M01", "M02", "M03"]
    status   = {}
    for m in methods:
        done = [mid for mid in machines
                if os.path.exists(os.path.join(LOG_DIR, f"{m}_{mid}.json"))]
        status[m] = {"done": done, "total": len(machines)}
    return jsonify(status)


# ── main page ─────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Split Learning Dashboard</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.30.0/dist/tabler-icons.min.css">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:      #f8f7f4;
    --surface: #ffffff;
    --border:  rgba(0,0,0,0.09);
    --text:    #1a1a1a;
    --muted:   #666;
    --accent:  #2563eb;
    --radius:  10px;
    --blue:    #2E86AB;
    --red:     #E84855;
    --green:   #2a9d5c;
    --amber:   #d97706;
  }

  body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.6; }

  /* ── sidebar ── */
  .layout { display: flex; min-height: 100vh; }
  .sidebar {
    width: 220px; min-width: 220px;
    background: var(--surface); border-right: 0.5px solid var(--border);
    display: flex; flex-direction: column;
    padding: 1.5rem 0; position: sticky; top: 0; height: 100vh; overflow-y: auto;
  }
  .sidebar-logo {
    padding: 0 1.25rem 1.5rem; font-size: 15px; font-weight: 600; letter-spacing: -0.3px;
    display: flex; align-items: center; gap: 8px; color: var(--text);
    border-bottom: 0.5px solid var(--border); margin-bottom: 0.75rem;
  }
  .sidebar-logo .dot {
    width: 26px; height: 26px; border-radius: 6px; background: var(--accent);
    display: flex; align-items: center; justify-content: center; color: white; font-size: 13px;
  }
  .nav-item {
    display: flex; align-items: center; gap: 10px; padding: 0.55rem 1.25rem;
    cursor: pointer; color: var(--muted); font-size: 13.5px;
    border-left: 3px solid transparent; transition: background 0.12s, color 0.12s;
  }
  .nav-item:hover  { background: var(--bg); color: var(--text); }
  .nav-item.active { color: var(--accent); background: #eff6ff; border-left-color: var(--accent); }
  .nav-item i { font-size: 17px; }
  .sidebar-link {
    margin-top: auto; padding: 0.75rem 1.25rem;
    border-top: 0.5px solid var(--border);
    font-size: 12px; color: var(--muted);
  }
  .sidebar-link a { color: var(--accent); text-decoration: none; font-weight: 500; }

  /* ── main ── */
  .main { flex: 1; padding: 2rem 2.5rem; overflow: auto; max-width: 1100px; }
  .page { display: none; }
  .page.active { display: block; }

  /* ── cards ── */
  .card { background: var(--surface); border: 0.5px solid var(--border); border-radius: var(--radius); padding: 1.25rem 1.5rem; margin-bottom: 1.25rem; }
  .card-title { font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin-bottom: 1rem; }

  /* ── metric row ── */
  .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 1.25rem; }
  .metric { background: var(--surface); border: 0.5px solid var(--border); border-radius: var(--radius); padding: 1rem 1.25rem; }
  .metric-label { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
  .metric-value { font-size: 22px; font-weight: 600; }
  .metric-sub   { font-size: 11px; color: var(--muted); margin-top: 2px; }

  /* ── status badges ── */
  .badge { display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 20px; font-size: 11.5px; font-weight: 500; }
  .badge-done { background: #d1fae5; color: #065f46; }
  .badge-pend { background: #fef3c7; color: #92400e; }
  .badge-none { background: #f3f4f6; color: #6b7280; }

  /* ── method tabs ── */
  .tabs { display: flex; gap: 4px; margin-bottom: 1.25rem; }
  .tab { padding: 6px 18px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; border: 0.5px solid var(--border); background: var(--surface); color: var(--muted); transition: all 0.12s; }
  .tab.active { background: var(--accent); color: white; border-color: var(--accent); }

  /* ── comparison table ── */
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { background: var(--bg); font-weight: 600; padding: 8px 12px; text-align: left; border-bottom: 0.5px solid var(--border); }
  td { padding: 8px 12px; border-bottom: 0.5px solid var(--border); }
  tr:last-child td { border-bottom: none; }
  .better { color: var(--green); font-weight: 600; }
  .worse  { color: var(--red);   font-weight: 500; }

  /* ── chart wrapper ── */
  .chart-wrap { position: relative; height: 280px; }
  .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; }

  /* ── plot images ── */
  .plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.25rem; }
  .plot-card { background: var(--surface); border: 0.5px solid var(--border); border-radius: var(--radius); overflow: hidden; }
  .plot-card img { width: 100%; display: block; }
  .plot-label { padding: 8px 12px; font-size: 12px; color: var(--muted); font-weight: 500; }

  /* ── lightbox ── */
  .lightbox-overlay {
    display: none; position: fixed; inset: 0; z-index: 1000;
    background: rgba(0,0,0,0.78); backdrop-filter: blur(4px);
    align-items: center; justify-content: center;
  }
  .lightbox-overlay.open { display: flex; }
  .lightbox-inner {
    position: relative; max-width: 92vw; max-height: 90vh;
    background: var(--surface); border-radius: 12px; overflow: hidden;
    box-shadow: 0 24px 64px rgba(0,0,0,0.45);
    display: flex; flex-direction: column;
  }
  .lightbox-inner img {
    display: block; max-width: 92vw; max-height: calc(90vh - 44px);
    object-fit: contain;
  }
  .lightbox-footer {
    padding: 8px 14px; font-size: 12px; color: var(--muted);
    font-weight: 500; border-top: 0.5px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    flex-shrink: 0;
  }
  .lightbox-close {
    width: 30px; height: 30px; border-radius: 50%;
    border: 0.5px solid var(--border); background: var(--surface);
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; font-size: 16px; color: var(--text);
    transition: background 0.12s, color 0.12s; flex-shrink: 0;
  }
  .lightbox-close:hover { background: var(--red); color: white; border-color: var(--red); }
  .plot-card img, .priv-plot-card img { cursor: zoom-in; }

  /* ── privacy section ── */
  .privacy-hero { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin-bottom: 1.25rem; }
  .priv-card { background: var(--surface); border: 0.5px solid var(--border); border-radius: var(--radius); padding: 1.25rem 1.5rem; }
  .priv-title { font-size: 14px; font-weight: 600; margin-bottom: 0.75rem; display: flex; align-items: center; gap: 8px; }
  .priv-title .dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .priv-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 0.5px solid var(--border); font-size: 13px; }
  .priv-row:last-child { border-bottom: none; }
  .priv-key { color: var(--muted); display: flex; align-items: center; gap: 5px; }
  .priv-val { font-weight: 600; }

  /* IER gauge bar */
  .ier-gauge-wrap { margin: 0.9rem 0 0.5rem; }
  .ier-gauge-label { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); margin-bottom: 4px; }
  .ier-track { background: var(--bg); border-radius: 6px; height: 10px; overflow: hidden; border: 0.5px solid var(--border); }
  .ier-fill { height: 10px; border-radius: 6px; transition: width 0.5s ease; }

  /* metric trio inside priv-card */
  .metric-trio { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 0.9rem; }
  .mtrio { background: var(--bg); border-radius: 8px; padding: 0.65rem 0.75rem; text-align: center; }
  .mtrio-val  { font-size: 18px; font-weight: 700; }
  .mtrio-lbl  { font-size: 10.5px; color: var(--muted); margin-top: 2px; line-height: 1.3; }

  /* privacy plots strip */
  .priv-plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.25rem; margin-bottom: 1.25rem; }
  .priv-plot-card { background: var(--surface); border: 0.5px solid var(--border); border-radius: var(--radius); overflow: hidden; }
  .priv-plot-card img { width: 100%; display: block; }
  .priv-plot-label { padding: 7px 12px; font-size: 11.5px; color: var(--muted); font-weight: 500; border-top: 0.5px solid var(--border); }
  .priv-plot-caption { padding: 0 12px 10px; font-size: 11px; color: var(--muted); line-height: 1.45; }

  /* comparison table inside privacy */
  .priv-table-wrap { overflow-x: auto; }
  .priv-compare-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .priv-compare-table th { background: #f1f5f9; font-weight: 600; padding: 9px 14px; text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap; }
  .priv-compare-table td { padding: 9px 14px; border-bottom: 0.5px solid var(--border); }
  .priv-compare-table tr:last-child td { border-bottom: none; }
  .priv-compare-table .metric-name { color: var(--muted); font-size: 12px; display: block; margin-top: 1px; }

  /* verdict chip */
  .verdict { display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px; border-radius: 20px; font-size: 11.5px; font-weight: 600; }
  .verdict-good { background: #d1fae5; color: #065f46; }
  .verdict-bad  { background: #fee2e2; color: #991b1b; }
  .verdict-na   { background: #f3f4f6; color: #6b7280; }

  /* ── pill ── */
  .pill { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 11.5px; font-weight: 600; }
  .pill-blue  { background: #dbeafe; color: #1e40af; }
  .pill-red   { background: #fee2e2; color: #991b1b; }
  .pill-green { background: #d1fae5; color: #065f46; }
  .pill-amber { background: #fef3c7; color: #92400e; }

  /* ── empty state ── */
  .empty { text-align: center; padding: 3rem; color: var(--muted); }
  .empty i { font-size: 40px; display: block; margin-bottom: 0.75rem; opacity: 0.4; }
  .empty p { font-size: 13px; }

  h2 { font-size: 18px; font-weight: 600; margin-bottom: 0.25rem; }
  .page-sub { color: var(--muted); font-size: 13px; margin-bottom: 1.75rem; }

  .refresh-btn { display: flex; align-items: center; gap: 6px; padding: 6px 14px; border: 0.5px solid var(--border); border-radius: 6px; background: var(--surface); font-size: 13px; cursor: pointer; color: var(--text); margin-left: auto; }
  .refresh-btn:hover { background: var(--bg); }
  .page-header { display: flex; align-items: center; margin-bottom: 0.25rem; }

  .progress-bar-wrap { background: var(--bg); border-radius: 4px; height: 6px; margin-top: 4px; overflow: hidden; }
  .progress-bar { height: 6px; border-radius: 4px; transition: width 0.4s; }

  .conclude-box {
    background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
    border: 0.5px solid #bfdbfe; border-radius: var(--radius); padding: 1.25rem 1.5rem;
  }
  .conclude-box h3 { color: #1e40af; }
  .conclude-row { display: flex; align-items: flex-start; gap: 10px; padding: 5px 0; font-size: 13.5px; }
  .conclude-row i { color: var(--green); margin-top: 2px; flex-shrink: 0; }
</style>
</head>
<body>
<div class="layout">

<!-- ── Sidebar ── -->
<nav class="sidebar" aria-label="Main navigation">
  <div class="sidebar-logo">
    <div class="dot"><i class="ti ti-brand-python" aria-hidden="true"></i></div>
    SplitLearn
  </div>
  <div class="nav-item active" onclick="showPage('overview')" data-page="overview">
    <i class="ti ti-layout-dashboard" aria-hidden="true"></i> Overview
  </div>
  <div class="nav-item" onclick="showPage('results')" data-page="results">
    <i class="ti ti-chart-bar" aria-hidden="true"></i> Results
  </div>
  <div class="nav-item" onclick="showPage('plots')" data-page="plots">
    <i class="ti ti-photo" aria-hidden="true"></i> Plots
  </div>
  <div class="nav-item" onclick="showPage('privacy')" data-page="privacy">
    <i class="ti ti-shield" aria-hidden="true"></i> Privacy analysis
  </div>

</nav>

<!-- ── Main ── -->
<main class="main">

<!-- ─────────── OVERVIEW ─────────── -->
<section class="page active" id="page-overview">
  <div class="page-header">
    <div>
      <h2>Split learning comparison</h2>
      <p class="page-sub">U-shape vs normal split — federated vibration fault detection</p>
    </div>
    <button class="refresh-btn" onclick="loadAll()">
      <i class="ti ti-refresh" aria-hidden="true"></i> Refresh
    </button>
  </div>

  <div id="overview-metrics" class="metric-grid">
    <div class="metric">
      <div class="metric-label">Experiments run</div>
      <div class="metric-value" id="ov-total">—</div>
      <div class="metric-sub">client log files</div>
    </div>
    <div class="metric">
      <div class="metric-label">Best accuracy</div>
      <div class="metric-value" id="ov-acc">—</div>
      <div class="metric-sub" id="ov-acc-method">—</div>
    </div>
    <div class="metric">
      <div class="metric-label">Best F1</div>
      <div class="metric-value" id="ov-f1">—</div>
      <div class="metric-sub" id="ov-f1-method">—</div>
    </div>
    <div class="metric">
      <div class="metric-label">Privacy winner</div>
      <div class="metric-value" style="font-size:16px; color: var(--blue);">U-shape</div>
      <div class="metric-sub">labels never exposed</div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Training status</div>
    <div id="status-table">
      <div class="empty"><i class="ti ti-loader" aria-hidden="true"></i><p>Loading...</p></div>
    </div>
  </div>

  <div class="chart-grid">
    <div class="card">
      <div class="card-title">Accuracy by method</div>
      <div class="chart-wrap"><canvas id="acc-chart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">F1 score by method</div>
      <div class="chart-wrap"><canvas id="f1-chart"></canvas></div>
    </div>
  </div>
  <div class="chart-grid">
    <div class="card">
      <div class="card-title">Training time (s)</div>
      <div class="chart-wrap"><canvas id="time-chart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Total communication (MB)</div>
      <div class="chart-wrap"><canvas id="comm-chart"></canvas></div>
    </div>
  </div>
</section>

<!-- ─────────── RESULTS ─────────── -->
<section class="page" id="page-results">
  <div class="page-header">
    <div>
      <h2>Results</h2>
      <p class="page-sub">Per-client metrics loaded from <code>logs/</code></p>
    </div>
    <button class="refresh-btn" onclick="loadAll()"><i class="ti ti-refresh" aria-hidden="true"></i> Refresh</button>
  </div>

  <div class="tabs" id="results-tabs">
    <div class="tab active" onclick="filterResults('all')"          data-filter="all">All</div>
    <div class="tab"        onclick="filterResults('u_shape')"      data-filter="u_shape">U-shape</div>
    <div class="tab"        onclick="filterResults('normal_split')" data-filter="normal_split">Normal split</div>
  </div>

  <div id="results-table-wrap">
    <div class="empty"><i class="ti ti-loader" aria-hidden="true"></i><p>Loading...</p></div>
  </div>
</section>

<!-- ─────────── PLOTS ─────────── -->
<section class="page" id="page-plots">
  <div class="page-header">
    <div>
      <h2>Plots</h2>
      <p class="page-sub">Generated by <code>auto_plot.py</code> and <code>privacy_analysis.py</code></p>
    </div>
    <button class="refresh-btn" onclick="loadPlots()"><i class="ti ti-refresh" aria-hidden="true"></i> Refresh</button>
  </div>
  <div id="plot-grid" class="plot-grid">
    <div class="empty"><i class="ti ti-photo-off" aria-hidden="true"></i><p>No plots yet. Run <code>auto_plot.py</code> first.</p></div>
  </div>
</section>

<!-- ─────────── PRIVACY ─────────── -->
<section class="page" id="page-privacy">
  <div class="page-header">
    <div>
      <h2>Privacy analysis</h2>
      <p class="page-sub">Three complementary metrics — all grounded in whether the server has access to labels</p>
    </div>
    <button class="refresh-btn" onclick="loadPrivacy()"><i class="ti ti-refresh" aria-hidden="true"></i> Refresh</button>
  </div>

  <!-- metric cards side-by-side -->
  <div id="privacy-hero" class="privacy-hero">
    <div class="empty"><i class="ti ti-loader" aria-hidden="true"></i><p>Loading…</p></div>
  </div>

  <!-- comparison table -->
  <div class="card" id="privacy-table-card" style="display:none">
    <div class="card-title">Side-by-side metric comparison</div>
    <div class="priv-table-wrap" id="privacy-table-wrap"></div>
  </div>

  <!-- privacy plots -->
  <div class="card" id="privacy-plots-card" style="display:none">
    <div class="card-title">Privacy plots</div>
    <div class="priv-plot-grid" id="privacy-plot-grid"></div>
  </div>

</section>

</main>
</div>

<!-- ── Lightbox ── -->
<div class="lightbox-overlay" id="lightbox" role="dialog" aria-modal="true" aria-label="Image preview" onclick="closeLightbox(event)">
  <div class="lightbox-inner">
    <img id="lightbox-img" src="" alt="">
    <div class="lightbox-footer">
      <span id="lightbox-label"></span>
      <button class="lightbox-close" onclick="closeLightbox()" title="Close (Esc)">
        <i class="ti ti-x" aria-hidden="true"></i>
      </button>
    </div>
  </div>
</div>

<script>
const BLUE  = '#2E86AB', RED = '#E84855', GREEN = '#2a9d5c';
const METHOD_COLOR = { u_shape: BLUE, normal_split: RED };
const METHOD_LABEL = { u_shape: 'U-shape', normal_split: 'Normal split' };

// ── Lightbox ─────────────────────────────────────────────────
function openLightbox(src, label) {
  document.getElementById('lightbox-img').src = src;
  document.getElementById('lightbox-label').textContent = label || '';
  document.getElementById('lightbox').classList.add('open');
  document.body.style.overflow = 'hidden';
}
function closeLightbox(e) {
  // close on overlay click (not on inner content) or on button click
  if (e && e.target !== document.getElementById('lightbox')) return;
  document.getElementById('lightbox').classList.remove('open');
  document.body.style.overflow = '';
}
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    document.getElementById('lightbox').classList.remove('open');
    document.body.style.overflow = '';
  }
});
// delegate clicks on any plot image anywhere on the page
document.addEventListener('click', e => {
  const img = e.target.closest('.plot-card img, .priv-plot-card img');
  if (!img) return;
  const label = img.closest('.plot-card, .priv-plot-card')
    ?.querySelector('.plot-label, .priv-plot-label')?.textContent?.trim() || '';
  openLightbox(img.src, label);
});

let allResults = [], charts = {};

function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  document.querySelector(`[data-page="${name}"]`).classList.add('active');
  if (name === 'plots')   loadPlots();
  if (name === 'privacy') loadPrivacy();
}

async function loadAll() {
  try {
    const [res, status] = await Promise.all([
      fetch('/api/results').then(r => r.json()),
      fetch('/api/status').then(r => r.json()),
    ]);
    allResults = res;
    renderOverview(res, status);
    renderResults(res);
  } catch(e) { console.error(e); }
}

function renderOverview(data, status) {
  document.getElementById('ov-total').textContent = data.length;
  if (data.length) {
    const best_acc = data.reduce((a,b) => a.accuracy > b.accuracy ? a : b);
    const best_f1  = data.reduce((a,b) => a.f1 > b.f1 ? a : b);
    document.getElementById('ov-acc').textContent = (best_acc.accuracy * 100).toFixed(1) + '%';
    document.getElementById('ov-acc-method').textContent = METHOD_LABEL[best_acc.method] + ' · ' + best_acc.client;
    document.getElementById('ov-f1').textContent = (best_f1.f1 * 100).toFixed(1) + '%';
    document.getElementById('ov-f1-method').textContent = METHOD_LABEL[best_f1.method] + ' · ' + best_f1.client;
  }

  const wrap = document.getElementById('status-table');
  if (!Object.keys(status).length) { wrap.innerHTML = '<div class="empty"><p>No training runs found</p></div>'; return; }
  let html = '<table><thead><tr><th>Method</th><th>Progress</th><th>Clients done</th></tr></thead><tbody>';
  for (const [method, info] of Object.entries(status)) {
    const pct   = (info.done.length / info.total * 100).toFixed(0);
    const color = METHOD_COLOR[method] || BLUE;
    html += `<tr>
      <td><span class="pill" style="background:${color}22; color:${color}">${METHOD_LABEL[method] || method}</span></td>
      <td style="width:220px">
        <div style="font-size:12px; color:var(--muted); margin-bottom:3px">${info.done.length} / ${info.total} complete</div>
        <div class="progress-bar-wrap"><div class="progress-bar" style="width:${pct}%; background:${color}"></div></div>
      </td>
      <td>${info.done.length ? info.done.map(d => `<span class="badge badge-done">${d}</span>`).join(' ') : '<span class="badge badge-none">none yet</span>'}</td>
    </tr>`;
  }
  html += '</tbody></table>';
  wrap.innerHTML = html;

  // Charts
  const methods = ['u_shape', 'normal_split'];
  const labels  = methods.map(m => METHOD_LABEL[m]);

  function avgBy(metric) {
    return methods.map(m => {
      const rows = data.filter(d => d.method === m);
      if (!rows.length) return 0;
      return rows.reduce((s,r) => s + r[metric], 0) / rows.length;
    });
  }

  const chartDefs = [
    { id: 'acc-chart',  data: avgBy('accuracy'),     label: 'Avg accuracy',  fmt: v => (v*100).toFixed(1)+'%' },
    { id: 'f1-chart',   data: avgBy('f1'),            label: 'Avg F1',        fmt: v => (v*100).toFixed(1)+'%' },
    { id: 'time-chart', data: avgBy('time'),          label: 'Avg time (s)',  fmt: v => v.toFixed(1)+'s' },
    { id: 'comm-chart', data: avgBy('total_comm_MB'), label: 'Avg comm (MB)', fmt: v => v.toFixed(2)+'MB' },
  ];

  chartDefs.forEach(({ id, data: vals, label, fmt }) => {
    const ctx = document.getElementById(id).getContext('2d');
    if (charts[id]) charts[id].destroy();
    charts[id] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label,
          data: vals,
          backgroundColor: methods.map(m => METHOD_COLOR[m] + 'cc'),
          borderColor: methods.map(m => METHOD_COLOR[m]),
          borderWidth: 1.5,
          borderRadius: 6,
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: ctx => fmt(ctx.raw) } },
        },
        scales: {
          y: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { font: { size: 11 } } },
          x: { grid: { display: false }, ticks: { font: { size: 12 } } },
        }
      }
    });
  });
}

let currentFilter = 'all';
function filterResults(f) {
  currentFilter = f;
  document.querySelectorAll('#results-tabs .tab').forEach(t => t.classList.remove('active'));
  document.querySelector(`[data-filter="${f}"]`).classList.add('active');
  renderResults(allResults);
}

function renderResults(data) {
  const filtered = currentFilter === 'all' ? data : data.filter(d => d.method === currentFilter);
  const wrap = document.getElementById('results-table-wrap');
  if (!filtered.length) {
    wrap.innerHTML = '<div class="empty"><i class="ti ti-database-off" aria-hidden="true"></i><p>No log files found yet. Complete a training run first.</p></div>';
    return;
  }
  let html = `<table>
    <thead><tr>
      <th>Method</th><th>Client</th><th>Accuracy</th><th>F1</th>
      <th>Time (s)</th><th>Sent (MB)</th><th>Recv (MB)</th><th>Total comm (MB)</th>
    </tr></thead><tbody>`;
  for (const r of filtered) {
    const c = METHOD_COLOR[r.method] || BLUE;
    html += `<tr>
      <td><span class="pill" style="background:${c}22; color:${c}">${METHOD_LABEL[r.method] || r.method}</span></td>
      <td><strong>${r.client}</strong></td>
      <td>${(r.accuracy*100).toFixed(2)}%</td>
      <td>${(r.f1*100).toFixed(2)}%</td>
      <td>${r.time ? r.time.toFixed(1) : '—'}</td>
      <td>${r.sent_MB != null ? r.sent_MB.toFixed(2) : '—'}</td>
      <td>${r.recv_MB != null ? r.recv_MB.toFixed(2) : '—'}</td>
      <td>${r.total_comm_MB != null ? r.total_comm_MB.toFixed(2) : '—'}</td>
    </tr>`;
  }
  html += '</tbody></table>';
  wrap.innerHTML = html;
}

async function loadPlots() {
  const data = await fetch('/api/plots').then(r => r.json());
  const grid = document.getElementById('plot-grid');
  const LABELS = {
    summary_grid: 'Summary grid (all metrics)',
    accuracy: 'Accuracy comparison', f1: 'F1 comparison',
    time: 'Training time', comm: 'Communication overhead',
    radar: 'Radar chart (normalised)',
    per_client_accuracy: 'Per-client accuracy', per_client_f1: 'Per-client F1',
    ier_bar: 'Privacy: Information Exposure Ratio',
    reconstruction_loss: 'Privacy: Input reconstruction MSE',
    gradient_risk: 'Privacy: Gradient inversion risk (GIRS)',
    privacy_summary_table: 'Privacy: summary table',
  };
  if (!Object.keys(data).length) {
    grid.innerHTML = '<div class="empty"><i class="ti ti-photo-off" aria-hidden="true"></i><p>No plots found. Run <code>auto_plot.py</code> and <code>privacy_analysis.py</code> first.</p></div>';
    return;
  }
  grid.innerHTML = Object.entries(data).map(([key, uri]) =>
    `<div class="plot-card">
       <img src="${uri}" alt="${LABELS[key] || key}" loading="lazy">
       <div class="plot-label">${LABELS[key] || key}</div>
     </div>`
  ).join('');
}

async function loadPrivacy() {
  const [data, plots] = await Promise.all([
    fetch('/api/privacy').then(r => r.json()),
    fetch('/api/plots').then(r => r.json()),
  ]);

  const hero = document.getElementById('privacy-hero');

  if (!data || !Object.keys(data).length) {
    hero.innerHTML = '<div class="empty" style="grid-column:1/-1"><i class="ti ti-shield-off" aria-hidden="true"></i><p>No privacy report found. Run <code>privacy_analysis.py</code> first.</p></div>';
    return;
  }

  const ier   = data.ier            || {};
  const recon = data.reconstruction || {};
  const girs  = data.girs           || {};

  // ── helper: build one method card ──────────────────────────────
  function methodCard(method) {
    const ir = ier[method]   || {};
    const rr = recon[method] || {};
    const gr = girs[method]  || {};
    const color = METHOD_COLOR[method];

    const ierVal  = ir.ier != null ? ir.ier.toFixed(1) : '—';
    const ierPct  = (ir.ier || 0) * 100;
    const fillCol = method === 'u_shape' ? 'var(--green)' : 'var(--red)';

    const mse     = rr.test_mse    != null ? rr.test_mse.toFixed(4)    : '—';
    const girsVal = gr.girs        != null ? gr.girs.toFixed(4)        : '—';
    const girsApplicable = gr.applicable === true;

    const privPill = method === 'u_shape'
      ? '<span class="verdict verdict-good"><i class="ti ti-lock" aria-hidden="true"></i> Strong privacy</span>'
      : '<span class="verdict verdict-bad"><i class="ti ti-lock-open" aria-hidden="true"></i> Weak privacy</span>';

    // IER interpretation text
    const ierDesc = method === 'u_shape'
      ? 'Labels never transmitted — protocol guarantee'
      : 'Every label sent to server each batch';

    // GIRS row text
    const girsDisplay = girsApplicable
      ? `${girsVal} <span style="color:var(--muted); font-weight:400; font-size:11px">(mean ‖∂L/∂A‖)</span>`
      : `0 <span class="verdict verdict-good" style="font-size:10.5px">N/A by protocol</span>`;

    return `
    <div class="priv-card">
      <div class="priv-title">
        <div class="dot" style="background:${color}"></div>
        ${METHOD_LABEL[method]}
        ${privPill}
      </div>

      <!-- IER gauge -->
      <div class="priv-row">
        <span class="priv-key"><i class="ti ti-percentage" aria-hidden="true"></i> IER (label exposure)</span>
        <span class="priv-val" style="color:${fillCol}">${ierVal}</span>
      </div>
      <div class="ier-gauge-wrap">
        <div class="ier-gauge-label"><span>0 — no exposure</span><span>1 — full exposure</span></div>
        <div class="ier-track">
          <div class="ier-fill" style="width:${ierPct}%; background:${fillCol}"></div>
        </div>
      </div>
      <div style="font-size:11.5px; color:var(--muted); margin-bottom:0.75rem">${ierDesc}</div>

      <!-- Recon + GIRS rows -->
      <div class="priv-row">
        <span class="priv-key"><i class="ti ti-eye-off" aria-hidden="true"></i> Reconstruction MSE ↑</span>
        <span class="priv-val">${mse}</span>
      </div>
      <div class="priv-row">
        <span class="priv-key"><i class="ti ti-wave-sine" aria-hidden="true"></i> Gradient inv. risk (GIRS) ↓</span>
        <span class="priv-val">${girsDisplay}</span>
      </div>

      <!-- mini trio -->
      <div class="metric-trio">
        <div class="mtrio">
          <div class="mtrio-val" style="color:${fillCol}">${ierVal}</div>
          <div class="mtrio-lbl">IER<br>(lower = better)</div>
        </div>
        <div class="mtrio">
          <div class="mtrio-val">${mse}</div>
          <div class="mtrio-lbl">Recon. MSE<br>(higher = better)</div>
        </div>
        <div class="mtrio">
          <div class="mtrio-val">${girsApplicable ? girsVal : '0'}</div>
          <div class="mtrio-lbl">GIRS<br>(lower = better)</div>
        </div>
      </div>

      <div style="margin-top:0.75rem; font-size:11px; color:var(--muted); line-height:1.5">
        ${ir.explanation || ''}
      </div>
    </div>`;
  }

  hero.innerHTML = methodCard('u_shape') + methodCard('normal_split');

  // ── comparison table ──────────────────────────────────────────
  const tableCard = document.getElementById('privacy-table-card');
  const tableWrap = document.getElementById('privacy-table-wrap');
  tableCard.style.display = 'block';

  const METRICS = [
    {
      key: 'IER', arrow: '↓ lower is better',
      icon: 'ti-percentage',
      desc: 'Fraction of label entropy structurally exposed to server',
      ref:  'Vepakomma et al., arXiv 1812.03288',
      u:  { val: (ier.u_shape?.ier ?? '—').toString(),  verdict: 'good', label: 'Zero exposure' },
      ns: { val: (ier.normal_split?.ier ?? '—').toString(), verdict: 'bad',  label: 'Full exposure' },
    },
    {
      key: 'Reconstruction MSE', arrow: '↑ higher is better',
      icon: 'ti-eye-off',
      desc: 'Adversarial decoder test MSE — higher = harder to recover inputs',
      ref:  'Server-side reconstruction attack (self-supervised)',
      u:  { val: recon.u_shape?.test_mse?.toFixed(4)    ?? '—', verdict: 'good', label: 'Harder to reconstruct' },
      ns: { val: recon.normal_split?.test_mse?.toFixed(4) ?? '—', verdict: 'bad',  label: 'Easier to reconstruct' },
    },
    {
      key: 'GIRS', arrow: '↓ lower is better',
      icon: 'ti-wave-sine',
      desc: 'Mean ‖∂L/∂A‖ — gradient inversion attack surface',
      ref:  'Zhu et al., NeurIPS 2019',
      u:  { val: '0', verdict: 'good', label: 'N/A — no loss on server' },
      ns: { val: girs.normal_split?.girs?.toFixed(4) ?? '—', verdict: 'bad',  label: 'DLG attack surface' },
    },
  ];

  let tHtml = `<table class="priv-compare-table">
    <thead><tr>
      <th>Metric</th>
      <th style="color:var(--blue)">U-Shape</th>
      <th style="color:var(--red)">Normal Split</th>
    </tr></thead><tbody>`;
  for (const m of METRICS) {
    tHtml += `<tr>
      <td>
        <strong><i class="ti ${m.icon}" aria-hidden="true"></i> ${m.key}</strong>
        <span class="metric-name">${m.desc} &nbsp; <em>${m.arrow}</em></span>
      </td>
      <td>
        <span class="priv-val">${m.u.val}</span><br>
        <span class="verdict verdict-${m.u.verdict}" style="margin-top:3px; display:inline-flex">${m.u.label}</span>
      </td>
      <td>
        <span class="priv-val">${m.ns.val}</span><br>
        <span class="verdict verdict-${m.ns.verdict}" style="margin-top:3px; display:inline-flex">${m.ns.label}</span>
      </td>
    </tr>`;
  }
  tHtml += '</tbody></table>';
  tableWrap.innerHTML = tHtml;

  // ── privacy plots ─────────────────────────────────────────────
  const plotCard = document.getElementById('privacy-plots-card');
  const plotGrid = document.getElementById('privacy-plot-grid');
  const PRIV_PLOTS = [
    { key: 'ier_bar',             label: 'Information Exposure Ratio',          caption: 'U-shape IER = 0 (no labels sent); Normal split IER = 1 (all labels sent every batch).' },
    { key: 'reconstruction_loss', label: 'Input Reconstruction Attack — MSE',   caption: 'Higher MSE means the server\'s decoder cannot recover the original vibration signal.' },
    { key: 'gradient_risk',       label: 'Gradient Inversion Risk Score (GIRS)', caption: 'U-shape GIRS = 0 by protocol. Normal split exposes ∂L/∂A, enabling DLG-style attacks.' },
    { key: 'privacy_summary_table', label: 'Privacy summary table',             caption: 'One-page overview of all three metrics across both methods.' },
  ];
  const availPlots = PRIV_PLOTS.filter(p => plots[p.key]);
  if (availPlots.length) {
    plotCard.style.display = 'block';
    plotGrid.innerHTML = availPlots.map(p => `
      <div class="priv-plot-card">
        <img src="${plots[p.key]}" alt="${p.label}" loading="lazy">
        <div class="priv-plot-label">${p.label}</div>
        <div class="priv-plot-caption">${p.caption}</div>
      </div>`).join('');
  } else {
    plotCard.style.display = 'none';
  }
}

loadAll();
setInterval(loadAll, 15000);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


if __name__ == "__main__":
    print("=" * 55)
    print("  Split Learning Dashboard")
    print("  http://127.0.0.1:5050")
    print("=" * 55)
    app.run(host="127.0.0.1", port=5050, debug=False)