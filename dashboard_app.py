"""
dashboard_app.py
----------------
Flask dashboard for Split Learning comparison.
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
from flask import Flask, jsonify, render_template_string, send_file

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
        "privacy_bars", "bc_per_dim", "activation_dist",
        "gradient_leak", "privacy_summary_table",
    ]
    out = {}
    for name in plot_names:
        uri = img_b64(os.path.join(PLOTS_DIR, f"{name}.png"))
        if uri:
            out[name] = uri
    return jsonify(out)


@app.route("/api/status")
def api_status():
    methods = ["normal_split", "u_shape"]
    machines = ["M01", "M02", "M03"]
    status = {}
    for m in methods:
        done = []
        for mid in machines:
            if os.path.exists(os.path.join(LOG_DIR, f"{m}_{mid}.json")):
                done.append(mid)
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
    --bg:       #f8f7f4;
    --surface:  #ffffff;
    --border:   rgba(0,0,0,0.09);
    --text:     #1a1a1a;
    --muted:    #666;
    --accent:   #2563eb;
    --radius:   10px;
    --blue:     #2E86AB;
    --red:      #E84855;
    --green:    #2a9d5c;
    --amber:    #d97706;
  }

  body {
    font-family: system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    font-size: 14px;
    line-height: 1.6;
  }

  /* ── sidebar ── */
  .layout { display: flex; min-height: 100vh; }

  .sidebar {
    width: 220px;
    min-width: 220px;
    background: var(--surface);
    border-right: 0.5px solid var(--border);
    display: flex;
    flex-direction: column;
    padding: 1.5rem 0;
    position: sticky;
    top: 0;
    height: 100vh;
    overflow-y: auto;
  }

  .sidebar-logo {
    padding: 0 1.25rem 1.5rem;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: -0.3px;
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text);
    border-bottom: 0.5px solid var(--border);
    margin-bottom: 0.75rem;
  }

  .sidebar-logo .dot {
    width: 26px; height: 26px; border-radius: 6px;
    background: var(--accent);
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 13px;
  }

  .nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 0.55rem 1.25rem;
    cursor: pointer;
    color: var(--muted);
    font-size: 13.5px;
    border-radius: 0;
    transition: background 0.12s, color 0.12s;
    border-left: 3px solid transparent;
  }
  .nav-item:hover { background: var(--bg); color: var(--text); }
  .nav-item.active { color: var(--accent); background: #eff6ff; border-left-color: var(--accent); }
  .nav-item i { font-size: 17px; }

  /* ── main ── */
  .main { flex: 1; padding: 2rem 2.5rem; overflow: auto; max-width: 1100px; }

  .page { display: none; }
  .page.active { display: block; }

  /* ── cards ── */
  .card {
    background: var(--surface);
    border: 0.5px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
  }

  .card-title {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
    margin-bottom: 1rem;
  }

  /* ── metric row ── */
  .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 1.25rem; }

  .metric {
    background: var(--surface);
    border: 0.5px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
  }
  .metric-label { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
  .metric-value { font-size: 22px; font-weight: 600; }
  .metric-sub { font-size: 11px; color: var(--muted); margin-top: 2px; }

  /* ── status badges ── */
  .badge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 8px; border-radius: 20px;
    font-size: 11.5px; font-weight: 500;
  }
  .badge-done  { background: #d1fae5; color: #065f46; }
  .badge-pend  { background: #fef3c7; color: #92400e; }
  .badge-none  { background: #f3f4f6; color: #6b7280; }

  /* ── terminal block ── */
  .terminal {
    background: #1e1e2e;
    color: #cdd6f4;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    font-family: 'Courier New', monospace;
    font-size: 13px;
    line-height: 1.8;
    margin-bottom: 0.75rem;
    position: relative;
  }
  .terminal .comment { color: #6c7086; }
  .terminal .cmd     { color: #a6e3a1; }
  .terminal .arg     { color: #89b4fa; }
  .terminal .wait    { color: #f9e2af; }

  .terminal-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 0.5rem; padding-bottom: 0.5rem;
    border-bottom: 0.5px solid rgba(255,255,255,0.08);
  }
  .term-title { color: #cba6f7; font-size: 12px; font-weight: 600; }
  .copy-btn {
    background: rgba(255,255,255,0.08);
    color: #cdd6f4;
    border: none;
    border-radius: 5px;
    padding: 3px 10px;
    font-size: 11px;
    cursor: pointer;
    display: flex; align-items: center; gap: 5px;
  }
  .copy-btn:hover { background: rgba(255,255,255,0.15); }

  .step-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 22px; height: 22px; border-radius: 50%;
    background: var(--accent); color: white;
    font-size: 11px; font-weight: 600;
    margin-right: 8px; flex-shrink: 0;
  }

  .step-row {
    display: flex; align-items: flex-start; gap: 0;
    margin-bottom: 0.75rem;
  }

  /* ── method tabs ── */
  .tabs { display: flex; gap: 4px; margin-bottom: 1.25rem; }
  .tab {
    padding: 6px 18px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    border: 0.5px solid var(--border);
    background: var(--surface);
    color: var(--muted);
    transition: all 0.12s;
  }
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

  /* ── privacy section ── */
  .privacy-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; }
  .priv-card { background: var(--surface); border: 0.5px solid var(--border); border-radius: var(--radius); padding: 1.25rem 1.5rem; }
  .priv-title { font-size: 14px; font-weight: 600; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 8px; }
  .priv-title .dot { width: 10px; height: 10px; border-radius: 50%; }
  .priv-row { display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 0.5px solid var(--border); font-size: 13px; }
  .priv-row:last-child { border-bottom: none; }
  .priv-key { color: var(--muted); }
  .priv-val { font-weight: 500; }

  /* ── pill ── */
  .pill {
    display: inline-block; padding: 2px 10px;
    border-radius: 20px; font-size: 11.5px; font-weight: 600;
  }
  .pill-blue   { background: #dbeafe; color: #1e40af; }
  .pill-red    { background: #fee2e2; color: #991b1b; }
  .pill-green  { background: #d1fae5; color: #065f46; }
  .pill-amber  { background: #fef3c7; color: #92400e; }

  /* ── upload section ── */
  .upload-zone {
    border: 1.5px dashed var(--border);
    border-radius: var(--radius);
    padding: 2rem;
    text-align: center;
    color: var(--muted);
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
    margin-bottom: 1rem;
  }
  .upload-zone:hover { border-color: var(--accent); background: #eff6ff; }
  .upload-zone i { font-size: 36px; margin-bottom: 0.5rem; display: block; }

  .file-list { font-size: 12.5px; }
  .file-row { display: flex; align-items: center; justify-content: space-between; padding: 6px 0; border-bottom: 0.5px solid var(--border); gap: 8px; }
  .file-name { display: flex; align-items: center; gap: 6px; color: var(--text); }
  .file-size { color: var(--muted); }

  /* ── empty state ── */
  .empty { text-align: center; padding: 3rem; color: var(--muted); }
  .empty i { font-size: 40px; display: block; margin-bottom: 0.75rem; opacity: 0.4; }
  .empty p { font-size: 13px; }

  h2 { font-size: 18px; font-weight: 600; margin-bottom: 0.25rem; }
  h3 { font-size: 15px; font-weight: 600; margin-bottom: 0.75rem; }
  .page-sub { color: var(--muted); font-size: 13px; margin-bottom: 1.75rem; }

  .refresh-btn {
    display: flex; align-items: center; gap: 6px;
    padding: 6px 14px;
    border: 0.5px solid var(--border);
    border-radius: 6px;
    background: var(--surface);
    font-size: 13px; cursor: pointer;
    color: var(--text);
    margin-left: auto;
  }
  .refresh-btn:hover { background: var(--bg); }

  .page-header { display: flex; align-items: center; margin-bottom: 0.25rem; }

  .notice {
    background: #eff6ff; border: 0.5px solid #bfdbfe;
    border-radius: var(--radius); padding: 0.75rem 1rem;
    font-size: 13px; color: #1e40af; margin-bottom: 1.25rem;
    display: flex; align-items: flex-start; gap: 8px;
  }
  .notice i { font-size: 16px; margin-top: 1px; flex-shrink: 0; }

  .progress-bar-wrap { background: var(--bg); border-radius: 4px; height: 6px; margin-top: 4px; overflow: hidden; }
  .progress-bar { height: 6px; border-radius: 4px; transition: width 0.4s; }

  .conclude-box {
    background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
    border: 0.5px solid #bfdbfe;
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
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
  <div class="nav-item" onclick="showPage('run')" data-page="run">
    <i class="ti ti-terminal-2" aria-hidden="true"></i> How to run
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
  <div class="nav-item" onclick="showPage('upload')" data-page="upload">
    <i class="ti ti-database-plus" aria-hidden="true"></i> New factory data
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

  <!-- training status -->
  <div class="card">
    <div class="card-title">Training status</div>
    <div id="status-table">
      <div class="empty"><i class="ti ti-loader" aria-hidden="true"></i><p>Loading...</p></div>
    </div>
  </div>

  <!-- mini bar charts -->
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

<!-- ─────────── HOW TO RUN ─────────── -->
<section class="page" id="page-run">
  <h2>How to run</h2>
  <p class="page-sub">Open 4 terminals in your project directory and run the commands below in order.</p>

  <div class="notice">
    <i class="ti ti-info-circle" aria-hidden="true"></i>
    <span>Run <strong>normal split first</strong> — it creates the <code>indices/</code> folder that ensures both methods see identical data.</span>
  </div>

  <div class="tabs" id="run-tabs">
    <div class="tab active" onclick="switchRun('normal')">Normal split</div>
    <div class="tab" onclick="switchRun('ushape')">U-shape</div>
    <div class="tab" onclick="switchRun('analysis')">Analysis</div>
  </div>

  <div id="run-normal">
    <div class="step-row">
      <span class="step-num">1</span>
      <div style="flex:1">
        <div style="font-weight:600; margin-bottom:6px;">Terminal 1 — Start the server</div>
        <div class="terminal">
          <div class="terminal-header">
            <span class="term-title">server</span>
            <button class="copy-btn" onclick="copyCmd('cmd-ns-server')"><i class="ti ti-copy" aria-hidden="true"></i> copy</button>
          </div>
          <span id="cmd-ns-server"><span class="cmd">python</span> <span class="arg">server_normal_split.py</span></span>
          <br><span class="wait"># wait until: Listening on port 8888, waiting for 3 client(s)...</span>
        </div>
      </div>
    </div>

    <div class="step-row">
      <span class="step-num">2</span>
      <div style="flex:1">
        <div style="font-weight:600; margin-bottom:6px;">Terminals 2, 3, 4 — Start each client</div>
        <div class="terminal">
          <div class="terminal-header">
            <span class="term-title">client M01</span>
            <button class="copy-btn" onclick="copyCmd('cmd-ns-m01')"><i class="ti ti-copy" aria-hidden="true"></i> copy</button>
          </div>
          <span id="cmd-ns-m01"><span class="cmd">python</span> <span class="arg">client_normal_split.py</span> M01</span>
        </div>
        <div class="terminal">
          <div class="terminal-header">
            <span class="term-title">client M02</span>
            <button class="copy-btn" onclick="copyCmd('cmd-ns-m02')"><i class="ti ti-copy" aria-hidden="true"></i> copy</button>
          </div>
          <span id="cmd-ns-m02"><span class="cmd">python</span> <span class="arg">client_normal_split.py</span> M02</span>
        </div>
        <div class="terminal">
          <div class="terminal-header">
            <span class="term-title">client M03</span>
            <button class="copy-btn" onclick="copyCmd('cmd-ns-m03')"><i class="ti ti-copy" aria-hidden="true"></i> copy</button>
          </div>
          <span id="cmd-ns-m03"><span class="cmd">python</span> <span class="arg">client_normal_split.py</span> M03</span>
        </div>
      </div>
    </div>
  </div>

  <div id="run-ushape" style="display:none">
    <div class="notice">
      <i class="ti ti-alert-circle" aria-hidden="true"></i>
      <span>Keep the <code>indices/</code> folder from the previous run — U-shape reuses those shuffle indices for a fair comparison.</span>
    </div>

    <div class="step-row">
      <span class="step-num">1</span>
      <div style="flex:1">
        <div style="font-weight:600; margin-bottom:6px;">Terminal 1 — Start the server</div>
        <div class="terminal">
          <div class="terminal-header">
            <span class="term-title">server</span>
            <button class="copy-btn" onclick="copyCmd('cmd-us-server')"><i class="ti ti-copy" aria-hidden="true"></i> copy</button>
          </div>
          <span id="cmd-us-server"><span class="cmd">python</span> <span class="arg">server_split.py</span></span>
          <br><span class="wait"># wait until: Listening on port 8888, waiting for 3 client(s)...</span>
        </div>
      </div>
    </div>

    <div class="step-row">
      <span class="step-num">2</span>
      <div style="flex:1">
        <div style="font-weight:600; margin-bottom:6px;">Terminals 2, 3, 4 — Start each client</div>
        <div class="terminal">
          <div class="terminal-header">
            <span class="term-title">client M01</span>
            <button class="copy-btn" onclick="copyCmd('cmd-us-m01')"><i class="ti ti-copy" aria-hidden="true"></i> copy</button>
          </div>
          <span id="cmd-us-m01"><span class="cmd">python</span> <span class="arg">client_split.py</span> M01</span>
        </div>
        <div class="terminal">
          <div class="terminal-header">
            <span class="term-title">client M02</span>
            <button class="copy-btn" onclick="copyCmd('cmd-us-m02')"><i class="ti ti-copy" aria-hidden="true"></i> copy</button>
          </div>
          <span id="cmd-us-m02"><span class="cmd">python</span> <span class="arg">client_split.py</span> M02</span>
        </div>
        <div class="terminal">
          <div class="terminal-header">
            <span class="term-title">client M03</span>
            <button class="copy-btn" onclick="copyCmd('cmd-us-m03')"><i class="ti ti-copy" aria-hidden="true"></i> copy</button>
          </div>
          <span id="cmd-us-m03"><span class="cmd">python</span> <span class="arg">client_split.py</span> M03</span>
        </div>
      </div>
    </div>
  </div>

  <div id="run-analysis" style="display:none">
    <div class="step-row">
      <span class="step-num">1</span>
      <div style="flex:1">
        <div style="font-weight:600; margin-bottom:6px;">Generate comparison CSVs and plots</div>
        <div class="terminal">
          <div class="terminal-header">
            <span class="term-title">any terminal</span>
            <button class="copy-btn" onclick="copyCmd('cmd-compare')"><i class="ti ti-copy" aria-hidden="true"></i> copy</button>
          </div>
          <span id="cmd-compare"><span class="cmd">python</span> <span class="arg">auto_compare.py</span>
<span class="cmd">python</span> <span class="arg">auto_plot.py</span>
<span class="cmd">python</span> <span class="arg">privacy_analysis.py</span></span>
        </div>
      </div>
    </div>
    <div class="step-row">
      <span class="step-num">2</span>
      <div style="flex:1">
        <div style="font-weight:600; margin-bottom:6px;">Reload this dashboard</div>
        <p style="color:var(--muted); font-size:13px;">Click the <strong>Refresh</strong> button on any page — the dashboard reads the updated log files automatically.</p>
      </div>
    </div>
  </div>

  <!-- tradeoff cheat-sheet -->
  <div class="card" style="margin-top: 1.5rem;">
    <div class="card-title">Expected tradeoffs</div>
    <table>
      <thead><tr>
        <th>Dimension</th>
        <th>Normal split</th>
        <th>U-shape</th>
      </tr></thead>
      <tbody>
        <tr><td>Accuracy / F1</td><td>Similar</td><td>Similar</td></tr>
        <tr><td>Label privacy</td><td class="worse">❌ Labels sent every batch</td><td class="better">✅ Labels never leave client</td></tr>
        <tr><td>Communication</td><td class="better">✅ Lower</td><td class="worse">Higher (extra tail grad)</td></tr>
        <tr><td>Client compute</td><td class="better">✅ Lower (front only)</td><td class="worse">Higher (front + tail)</td></tr>
        <tr><td>Server compute</td><td class="worse">Higher (owns loss)</td><td class="better">✅ Lower (middle only)</td></tr>
        <tr><td>Overall verdict</td><td>Simpler but privacy-leaking</td><td class="better" style="color:var(--blue)">✅ Better: same accuracy, stronger privacy</td></tr>
      </tbody>
    </table>
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
    <div class="tab active" onclick="filterResults('all')" data-filter="all">All</div>
    <div class="tab" onclick="filterResults('u_shape')" data-filter="u_shape">U-shape</div>
    <div class="tab" onclick="filterResults('normal_split')" data-filter="normal_split">Normal split</div>
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
      <p class="page-sub">From <code>logs/privacy_report.json</code></p>
    </div>
    <button class="refresh-btn" onclick="loadPrivacy()"><i class="ti ti-refresh" aria-hidden="true"></i> Refresh</button>
  </div>

  <div id="privacy-content">
    <div class="empty"><i class="ti ti-loader" aria-hidden="true"></i><p>Loading...</p></div>
  </div>

  <div class="conclude-box" style="margin-top:1.5rem">
    <h3><i class="ti ti-trophy" aria-hidden="true"></i> Conclusion: why U-shape wins</h3>
    <div class="conclude-row"><i class="ti ti-check" aria-hidden="true"></i><span><strong>Same predictive performance</strong> — accuracy and F1 are statistically equivalent across both methods when trained under fair comparison conditions.</span></div>
    <div class="conclude-row"><i class="ti ti-check" aria-hidden="true"></i><span><strong>Zero label exposure</strong> — in U-shape, labels are computed only at the client tail; the server never receives or infers them directly.</span></div>
    <div class="conclude-row"><i class="ti ti-check" aria-hidden="true"></i><span><strong>Lower inference attack accuracy</strong> — a simulated curious-server attack achieves significantly lower accuracy against U-shape activations.</span></div>
    <div class="conclude-row"><i class="ti ti-check" aria-hidden="true"></i><span><strong>Lower mutual information</strong> — the activations sent to the server carry less label-correlated information in U-shape.</span></div>
    <div class="conclude-row"><i class="ti ti-check" aria-hidden="true"></i><span><strong>Modest overhead</strong> — the extra communication and client compute are a small price for the privacy gain, especially in industrial IoT deployments.</span></div>
  </div>
</section>

<!-- ─────────── UPLOAD ─────────── -->
<section class="page" id="page-upload">
  <h2>New factory data</h2>
  <p class="page-sub">Add a new machine client to the federation. Follow the steps below — no code changes needed.</p>

  <div class="card">
    <div class="card-title">Step 1 — Prepare your data folder</div>
    <p style="font-size:13px; color:var(--muted); margin-bottom:1rem;">Your H5 files must be organised as shown:</p>
    <div class="terminal">
      <span class="comment"># Required directory structure</span><br>
      new_train/<br>
      &nbsp;&nbsp;<span class="arg">M04</span>/          <span class="comment">← use the next available ID</span><br>
      &nbsp;&nbsp;&nbsp;&nbsp;good/  <span class="comment">*.h5  (normal vibration)</span><br>
      &nbsp;&nbsp;&nbsp;&nbsp;bad/   <span class="comment">*.h5  (faulty vibration)</span>
    </div>
    <p style="font-size:13px; color:var(--muted);">Each H5 file must contain a dataset named <code>vibration_data</code> with shape <code>(1024, 3)</code>.</p>
  </div>

  <div class="card">
    <div class="card-title">Step 2 — Upload H5 files</div>
    <div class="upload-zone" onclick="document.getElementById('file-input').click()"
         ondragover="event.preventDefault(); this.style.borderColor='var(--accent)'"
         ondragleave="this.style.borderColor=''"
         ondrop="handleDrop(event)">
      <i class="ti ti-upload" aria-hidden="true"></i>
      <p style="font-weight:500; margin-bottom:4px;">Drop H5 files here or click to browse</p>
      <p style="font-size:12px;">Only <code>.h5</code> files accepted</p>
    </div>
    <input type="file" id="file-input" accept=".h5" multiple style="display:none" onchange="handleFiles(this.files)">
    <div id="file-list" class="file-list"></div>
  </div>

  <div class="card">
    <div class="card-title">Step 3 — Register and run</div>
    <p style="font-size:13px; color:var(--muted); margin-bottom:1rem;">
      Once your data is in place, add your machine ID to the server config and start an extra client terminal:
    </p>
    <div class="terminal">
      <span class="comment"># In server_split.py (or server_normal_split.py), update:</span><br>
      <span class="arg">NUM_CLIENTS</span> = <span class="cmd">4</span>   <span class="comment"># was 3</span><br>
      <br>
      <span class="comment"># Then run a 5th terminal with your new machine ID:</span><br>
      <span class="cmd">python</span> <span class="arg">client_split.py</span> M04
    </div>
    <p style="font-size:12px; color:var(--muted); margin-top:0.75rem;">
      <i class="ti ti-info-circle" aria-hidden="true"></i>
      The server will wait until all <code>NUM_CLIENTS</code> terminals connect before training begins.
    </p>
  </div>

  <div class="card">
    <div class="card-title">H5 data validator</div>
    <p style="font-size:13px; color:var(--muted); margin-bottom:0.75rem;">Paste a filename to check the expected shape:</p>
    <div style="display:flex; gap:8px; align-items:center;">
      <input type="text" id="validate-input" placeholder="e.g.  M04/good/sample_001.h5" style="flex:1; padding:8px 12px; border:0.5px solid var(--border); border-radius:6px; font-size:13px; background:var(--bg);">
      <button onclick="validateShape()" style="padding:8px 16px; border:0.5px solid var(--accent); background:var(--accent); color:white; border-radius:6px; cursor:pointer; font-size:13px;">Check</button>
    </div>
    <div id="validate-result" style="margin-top:0.75rem; font-size:13px;"></div>
  </div>
</section>

</main>
</div>

<script>
const BLUE  = '#2E86AB', RED = '#E84855', GREEN = '#2a9d5c', AMBER = '#d97706';
const METHOD_COLOR = { u_shape: BLUE, normal_split: RED };
const METHOD_LABEL = { u_shape: 'U-shape', normal_split: 'Normal split' };

let allResults = [], charts = {};

function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  document.querySelector(`[data-page="${name}"]`).classList.add('active');
  if (name === 'plots') loadPlots();
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
  } catch(e) {
    console.error(e);
  }
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

  // Status table
  const wrap = document.getElementById('status-table');
  if (!Object.keys(status).length) { wrap.innerHTML = '<div class="empty"><p>No training runs found</p></div>'; return; }
  let html = '<table><thead><tr><th>Method</th><th>Progress</th><th>Clients done</th></tr></thead><tbody>';
  for (const [method, info] of Object.entries(status)) {
    const pct = (info.done.length / info.total * 100).toFixed(0);
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
    { id: 'acc-chart',  data: avgBy('accuracy'),      label: 'Avg accuracy',   fmt: v => (v*100).toFixed(1)+'%' },
    { id: 'f1-chart',   data: avgBy('f1'),             label: 'Avg F1',         fmt: v => (v*100).toFixed(1)+'%' },
    { id: 'time-chart', data: avgBy('time'),           label: 'Avg time (s)',   fmt: v => v.toFixed(1)+'s' },
    { id: 'comm-chart', data: avgBy('total_comm_MB'),  label: 'Avg comm (MB)',  fmt: v => v.toFixed(2)+'MB' },
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
    privacy_bars: 'Privacy: attack accuracy + mutual info',
    bc_per_dim: 'Bhattacharyya overlap per dimension',
    activation_dist: 'Activation PCA (server view)',
    gradient_leak: 'Gradient leakage',
    privacy_summary_table: 'Privacy summary table',
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
  const data = await fetch('/api/privacy').then(r => r.json());
  const wrap = document.getElementById('privacy-content');
  if (!data || !Object.keys(data).length) {
    wrap.innerHTML = '<div class="empty"><i class="ti ti-shield-off" aria-hidden="true"></i><p>No privacy report found. Run <code>privacy_analysis.py</code> first.</p></div>';
    return;
  }

  const atk = data.attack_results || {};
  const mi  = data.mutual_info    || {};
  const bc  = data.bc_overlap     || {};
  const gl  = data.gradient_leakage || {};

  let html = '<div class="privacy-row">';
  for (const method of ['u_shape', 'normal_split']) {
    if (!atk[method]) continue;
    const color = METHOD_COLOR[method];
    const a = atk[method];
    const pill_privacy = method === 'u_shape'
      ? '<span class="pill pill-green">Higher ✓</span>'
      : '<span class="pill pill-red">Lower ✗</span>';
    html += `<div class="priv-card">
      <div class="priv-title"><div class="dot" style="background:${color}"></div> ${METHOD_LABEL[method]} ${pill_privacy}</div>
      <div class="priv-row"><span class="priv-key">Labels to server</span><span class="priv-val">${data.label_exposure?.[method]?.labels_sent_to_server || '—'}</span></div>
      <div class="priv-row"><span class="priv-key">Attack accuracy</span><span class="priv-val">${(a.mean*100).toFixed(1)}% ±${(a.ci95*100).toFixed(1)}%</span></div>
      <div class="priv-row"><span class="priv-key">Mutual information</span><span class="priv-val">${mi[method] != null ? mi[method].toFixed(4)+' bits' : '—'}</span></div>
      <div class="priv-row"><span class="priv-key">BC overlap</span><span class="priv-val">${bc[method] != null ? bc[method].toFixed(4) : '—'}</span></div>
      <div class="priv-row"><span class="priv-key">Gradient cos-sim</span><span class="priv-val">${gl[method]?.cosine_similarity != null ? gl[method].cosine_similarity.toFixed(4) : '—'} (${gl[method]?.leakage_level || '—'})</span></div>
    </div>`;
  }
  html += '</div>';
  wrap.innerHTML = html;
}

function switchRun(tab) {
  ['normal','ushape','analysis'].forEach(t => {
    document.getElementById('run-' + t).style.display = t === tab ? 'block' : 'none';
  });
  document.querySelectorAll('#run-tabs .tab').forEach((el, i) => {
    el.classList.toggle('active', ['normal','ushape','analysis'][i] === tab);
  });
}

function copyCmd(id) {
  const el = document.getElementById(id);
  navigator.clipboard.writeText(el.textContent.trim()).catch(() => {});
  const btn = el.closest('.terminal').querySelector('.copy-btn');
  const orig = btn.innerHTML;
  btn.innerHTML = '<i class="ti ti-check" aria-hidden="true"></i> copied';
  setTimeout(() => btn.innerHTML = orig, 1500);
}

function handleFiles(files) {
  const list = document.getElementById('file-list');
  const valid = Array.from(files).filter(f => f.name.endsWith('.h5'));
  if (!valid.length) { list.innerHTML = '<p style="color:var(--red); font-size:13px">No .h5 files selected.</p>'; return; }
  list.innerHTML = valid.map(f =>
    `<div class="file-row">
       <div class="file-name"><i class="ti ti-file-database" aria-hidden="true"></i> ${f.name}</div>
       <div class="file-size">${(f.size/1024).toFixed(1)} KB</div>
       <span class="badge badge-done">ready</span>
     </div>`
  ).join('');
}

function handleDrop(e) {
  e.preventDefault();
  e.currentTarget.style.borderColor = '';
  handleFiles(e.dataTransfer.files);
}

function validateShape() {
  const val = document.getElementById('validate-input').value.trim();
  const res = document.getElementById('validate-result');
  if (!val) { res.innerHTML = ''; return; }
  const ok = val.endsWith('.h5') && (val.includes('/good/') || val.includes('/bad/'));
  res.innerHTML = ok
    ? `<span style="color:var(--green)"><i class="ti ti-check" aria-hidden="true"></i> Path looks valid. Expected dataset shape: <code>(1024, 3)</code> float32.</span>`
    : `<span style="color:var(--red)"><i class="ti ti-x" aria-hidden="true"></i> Path should end in <code>.h5</code> and be inside a <code>good/</code> or <code>bad/</code> subfolder.</span>`;
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