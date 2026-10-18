# SPDX-License-Identifier: MIT
# OmniFlow — Unified training-free omni-modal LLM orchestration.
# Copyright (c) 2025 OmniFlow Contributors.
"""Single-page HTML dashboard template served at ``/``.

Kept in its own module so :mod:`omnimodal.api.server` stays readable.  The
template is a plain string constant — no Jinja, no client-side build step.
"""
from __future__ import annotations

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>OmniFlow — Training-free omni-modal LLM orchestration</title>
<meta name="description" content="OmniFlow turns any LLM into an omni-modal orchestrator that schedules and fuses the outputs of arbitrary training-free perception experts.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E%F0%9F%A7%AC%3C/text%3E%3C/svg%3E" />
<style>
  :root {
    --bg: #0b1220;
    --bg-soft: #0f1a2e;
    --panel: #131f36;
    --panel-2: #182842;
    --border: #243657;
    --text: #e6edf7;
    --muted: #9fb1cf;
    --accent: #14b8a6;       /* teal-500 */
    --accent-2: #f59e0b;     /* amber-500 */
    --accent-3: #ec4899;     /* pink-500 */
    --accent-soft: rgba(20, 184, 166, 0.12);
    --good: #34d399;
    --warn: #fbbf24;
    --bad: #f87171;
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: var(--sans); -webkit-font-smoothing: antialiased; }
  body { background:
      radial-gradient(900px 600px at 12% -8%, rgba(20,184,166,0.18), transparent 60%),
      radial-gradient(900px 700px at 92% -4%, rgba(236,72,153,0.14), transparent 60%),
      radial-gradient(700px 600px at 50% 110%, rgba(245,158,11,0.10), transparent 60%),
      var(--bg);
    min-height: 100vh;
  }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  code, pre, .mono { font-family: var(--mono); }

  /* sticky header */
  header.topbar {
    position: sticky; top: 0; z-index: 50;
    backdrop-filter: blur(10px);
    background: rgba(11, 18, 32, 0.78);
    border-bottom: 1px solid var(--border);
  }
  .topbar-inner {
    max-width: 1240px; margin: 0 auto; padding: 14px 22px;
    display: flex; align-items: center; gap: 18px;
  }
  .brand { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 18px; letter-spacing: -0.01em; }
  .brand .logo { width: 32px; height: 32px; display: inline-flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, var(--accent), #0ea5a4); border-radius: 9px; font-size: 18px; box-shadow: 0 8px 20px rgba(20,184,166,0.35); }
  .brand small { color: var(--muted); font-weight: 500; font-size: 12px; }
  .nav-spacer { flex: 1; }
  .nav-link { color: var(--muted); font-size: 14px; padding: 6px 10px; border-radius: 8px; }
  .nav-link:hover { color: var(--text); background: var(--panel); text-decoration: none; }
  .badge-pill {
    background: rgba(52,211,153,0.12); color: var(--good);
    border: 1px solid rgba(52,211,153,0.35); border-radius: 999px;
    padding: 5px 11px; font-size: 12px; font-weight: 600; display: inline-flex; align-items: center; gap: 6px;
  }
  .pulse-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--good); box-shadow: 0 0 0 0 rgba(52,211,153,0.7); animation: pulse 2s infinite; }
  @keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(52,211,153,0.6); }
    70% { box-shadow: 0 0 0 8px rgba(52,211,153,0); }
    100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
  }

  /* layout */
  main { max-width: 1240px; margin: 0 auto; padding: 32px 22px 60px; }
  .hero { padding: 28px 0 22px; }
  .hero h1 { font-size: clamp(32px, 4.5vw, 54px); line-height: 1.05; margin: 0 0 14px; letter-spacing: -0.02em; font-weight: 800; }
  .hero h1 .grad { background: linear-gradient(120deg, var(--accent) 0%, #5eead4 40%, var(--accent-2) 80%); -webkit-background-clip: text; background-clip: text; color: transparent; }
  .hero p.lead { color: var(--muted); font-size: clamp(15px, 1.4vw, 18px); max-width: 720px; line-height: 1.65; margin: 0 0 22px; }
  .cta-row { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 4px; }
  .btn {
    display: inline-flex; align-items: center; gap: 8px; padding: 11px 18px;
    border-radius: 10px; font-weight: 600; font-size: 14px; cursor: pointer; border: 1px solid transparent;
    transition: transform .12s ease, background .12s ease, border-color .12s ease;
  }
  .btn:hover { transform: translateY(-1px); }
  .btn-primary { background: linear-gradient(135deg, var(--accent), #0d9488); color: #06221f; border-color: rgba(20,184,166,0.6); box-shadow: 0 8px 22px rgba(20,184,166,0.35); }
  .btn-ghost { background: var(--panel); color: var(--text); border-color: var(--border); }
  .btn-ghost:hover { border-color: var(--accent); }

  /* hero stat strip */
  .stat-strip {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 26px;
  }
  .stat-card {
    background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
    border: 1px solid var(--border); border-radius: 14px; padding: 16px 18px;
    position: relative; overflow: hidden;
  }
  .stat-card .stat-label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }
  .stat-card .stat-value { font-size: 26px; font-weight: 800; letter-spacing: -0.02em; margin-top: 4px; }
  .stat-card .stat-sub { color: var(--muted); font-size: 12px; margin-top: 2px; }
  .stat-card .accent-bar { position: absolute; top: 0; left: 0; right: 0; height: 3px; background: var(--accent); }
  .stat-card.amber .accent-bar { background: var(--accent-2); }
  .stat-card.pink .accent-bar { background: var(--accent-3); }
  .stat-card.teal .accent-bar { background: var(--accent); }

  /* section heading */
  .section { margin-top: 56px; }
  .section h2 { font-size: 24px; font-weight: 700; letter-spacing: -0.01em; margin: 0 0 6px; }
  .section .section-sub { color: var(--muted); font-size: 14px; margin: 0 0 20px; max-width: 760px; line-height: 1.6; }
  .eyebrow { color: var(--accent); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; }

  /* pillars grid */
  .pillars { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  .pillar {
    background: var(--panel); border: 1px solid var(--border); border-radius: 16px;
    padding: 22px 22px 20px; position: relative; overflow: hidden;
  }
  .pillar .pillar-num { position: absolute; right: 18px; top: 14px; font-size: 46px; font-weight: 800; color: rgba(159,177,207,0.10); letter-spacing: -0.04em; }
  .pillar .pillar-icon { width: 42px; height: 42px; border-radius: 11px; display: flex; align-items: center; justify-content: center; font-size: 20px; margin-bottom: 14px; }
  .pillar.t1 .pillar-icon { background: var(--accent-soft); color: var(--accent); border: 1px solid rgba(20,184,166,0.35); }
  .pillar.t2 .pillar-icon { background: rgba(245,158,11,0.12); color: var(--accent-2); border: 1px solid rgba(245,158,11,0.35); }
  .pillar.t3 .pillar-icon { background: rgba(236,72,153,0.12); color: var(--accent-3); border: 1px solid rgba(236,72,153,0.35); }
  .pillar h3 { margin: 0 0 4px; font-size: 18px; font-weight: 700; }
  .pillar .pillar-tag { color: var(--muted); font-size: 13px; margin: 0 0 10px; }
  .pillar p { color: var(--muted); font-size: 14px; line-height: 1.65; margin: 0 0 12px; }
  .pillar .module-tag { font-family: var(--mono); font-size: 12px; color: var(--accent); background: var(--accent-soft); padding: 4px 8px; border-radius: 6px; }

  /* live demo */
  .demo-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
  .demo-input, .demo-output { background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 20px; }
  .demo-input label { display: block; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
  .demo-input textarea {
    width: 100%; min-height: 92px; background: var(--bg-soft); color: var(--text);
    border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px;
    font-family: var(--sans); font-size: 14px; line-height: 1.55; resize: vertical;
  }
  .demo-input textarea:focus { outline: 2px solid var(--accent); border-color: var(--accent); }
  .chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 16px; }
  .chip {
    background: var(--bg-soft); border: 1px solid var(--border); color: var(--muted);
    padding: 6px 12px; border-radius: 999px; font-size: 12px; cursor: pointer;
    transition: all .12s ease;
  }
  .chip:hover { color: var(--text); border-color: var(--accent); background: var(--accent-soft); }
  .demo-input .actions { display: flex; align-items: center; gap: 10px; margin-top: 6px; }
  .demo-output h4 { margin: 0 0 14px; font-size: 14px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
  .kv-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 14px; }
  .kv { background: var(--bg-soft); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; }
  .kv .k { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; }
  .kv .v { font-family: var(--mono); font-size: 16px; font-weight: 700; margin-top: 2px; }

  .plan-track { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 6px 0 14px; }
  .plan-step {
    display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px;
    border-radius: 999px; font-size: 12px; font-weight: 600;
    background: var(--accent-soft); color: var(--accent); border: 1px solid rgba(20,184,166,0.45);
  }
  .plan-arrow { color: var(--muted); font-size: 14px; }

  .step-list { display: flex; flex-direction: column; gap: 10px; max-height: 360px; overflow-y: auto; padding-right: 6px; }
  .step-list::-webkit-scrollbar { width: 8px; }
  .step-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 8px; }
  .step-list::-webkit-scrollbar-track { background: transparent; }
  .step-card { background: var(--bg-soft); border: 1px solid var(--border); border-radius: 12px; padding: 12px 14px; }
  .step-card .step-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 8px; }
  .step-card .step-id { font-family: var(--mono); font-size: 12px; color: var(--accent); }
  .step-card .step-toks { display: flex; flex-wrap: wrap; gap: 6px; }
  .expert-tag { font-family: var(--mono); font-size: 11px; padding: 3px 8px; border-radius: 6px; background: var(--panel-2); color: var(--text); border: 1px solid var(--border); }
  .expert-tag.terminal { background: rgba(236,72,153,0.10); color: var(--accent-3); border-color: rgba(236,72,153,0.40); }
  .resp-line { font-size: 12px; color: var(--muted); margin-top: 8px; font-family: var(--mono); }
  .resp-line .conf { color: var(--good); }
  .resp-line .lat { color: var(--accent-2); }

  /* experts grid */
  .experts-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; }
  .expert-card { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 18px; position: relative; }
  .expert-card .expert-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 8px; }
  .expert-card .expert-name { font-family: var(--mono); font-size: 14px; font-weight: 700; color: var(--accent); }
  .expert-card .expert-mod { font-size: 11px; padding: 3px 8px; border-radius: 6px; background: var(--bg-soft); color: var(--muted); border: 1px solid var(--border); }
  .expert-card p { margin: 0 0 10px; color: var(--muted); font-size: 13px; line-height: 1.55; }
  .expert-card .dep { font-size: 11px; color: var(--muted); }
  .expert-card .dep strong { color: var(--text); }
  .expert-card .terminal-pill { font-size: 10px; padding: 2px 6px; border-radius: 5px; background: rgba(236,72,153,0.10); color: var(--accent-3); border: 1px solid rgba(236,72,153,0.40); }

  /* benchmark table */
  .bench-wrap { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }
  table.bench { width: 100%; border-collapse: collapse; font-size: 14px; }
  table.bench th, table.bench td { padding: 12px 16px; text-align: left; border-bottom: 1px solid var(--border); }
  table.bench th { background: var(--bg-soft); color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
  table.bench tr:last-child td { border-bottom: none; }
  table.bench td.cat { font-family: var(--mono); color: var(--accent); }
  table.bench td.num { font-family: var(--mono); text-align: right; }
  .bench-summary { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; padding: 14px 16px; background: var(--bg-soft); border-top: 1px solid var(--border); }
  .bench-summary .k { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }
  .bench-summary .v { font-family: var(--mono); font-size: 18px; font-weight: 700; margin-top: 2px; }

  /* architecture diagram */
  .arch-diagram {
    background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 26px;
    display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 18px; align-items: stretch;
  }
  .arch-col { background: var(--bg-soft); border: 1px solid var(--border); border-radius: 12px; padding: 16px; position: relative; }
  .arch-col h4 { margin: 0 0 10px; font-size: 14px; color: var(--accent); font-family: var(--mono); }
  .arch-col .arch-stage { background: var(--panel-2); border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; margin-bottom: 8px; font-size: 12px; }
  .arch-col .arch-stage .stage-label { color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; }
  .arch-col .arch-stage .stage-val { color: var(--text); font-family: var(--mono); margin-top: 2px; }
  .arch-col .arch-arrow { text-align: center; color: var(--accent); font-size: 16px; margin: 4px 0; }

  /* memory panel */
  .memory-list { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 14px; max-height: 320px; overflow-y: auto; }
  .memory-list::-webkit-scrollbar { width: 8px; }
  .memory-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 8px; }
  .mem-row { display: flex; gap: 10px; padding: 8px 4px; border-bottom: 1px dashed var(--border); font-size: 13px; }
  .mem-row:last-child { border-bottom: none; }
  .mem-row .mem-src { font-family: var(--mono); font-size: 11px; color: var(--accent); white-space: nowrap; }
  .mem-row .mem-mod { font-family: var(--mono); font-size: 11px; color: var(--accent-2); white-space: nowrap; }
  .mem-row .mem-text { color: var(--muted); flex: 1; line-height: 1.5; word-break: break-word; }

  /* footer */
  footer.site {
    margin-top: 70px;
    border-top: 1px solid var(--border);
    background: rgba(11,18,32,0.6);
    backdrop-filter: blur(8px);
  }
  .footer-inner { max-width: 1240px; margin: 0 auto; padding: 26px 22px; display: flex; flex-wrap: wrap; gap: 18px; justify-content: space-between; align-items: center; }
  .footer-inner .copy { color: var(--muted); font-size: 13px; }
  .footer-inner .links { display: flex; gap: 14px; flex-wrap: wrap; }
  .footer-inner .links a { color: var(--muted); font-size: 13px; }
  .footer-inner .links a:hover { color: var(--accent); }

  /* spinner */
  .spinner { width: 14px; height: 14px; border: 2px solid rgba(255,255,255,0.25); border-top-color: #06221f; border-radius: 50%; animation: spin 0.7s linear infinite; display: inline-block; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* toast */
  .toast { position: fixed; bottom: 22px; right: 22px; background: var(--panel-2); border: 1px solid var(--accent); color: var(--text); padding: 12px 16px; border-radius: 10px; box-shadow: 0 10px 30px rgba(0,0,0,0.4); opacity: 0; transform: translateY(10px); transition: opacity .15s ease, transform .15s ease; max-width: 360px; font-size: 14px; z-index: 100; }
  .toast.show { opacity: 1; transform: translateY(0); }

  /* responsive */
  @media (max-width: 900px) {
    .stat-strip { grid-template-columns: repeat(2, 1fr); }
    .pillars, .demo-grid, .arch-diagram { grid-template-columns: 1fr; }
    .nav-link { display: none; }
    .nav-link.always { display: inline; }
  }
  @media (max-width: 560px) {
    .stat-strip { grid-template-columns: 1fr 1fr; }
    .kv-grid { grid-template-columns: 1fr; }
    .bench-summary { grid-template-columns: 1fr; }
    main { padding: 22px 14px 40px; }
    .topbar-inner { padding: 12px 14px; }
  }
</style>
</head>
<body>
<header class="topbar">
  <div class="topbar-inner">
    <div class="brand">
      <span class="logo">🧬</span>
      <div>
        OmniFlow
        <div><small>training-free omni-modal orchestration</small></div>
      </div>
    </div>
    <div class="nav-spacer"></div>
    <nav>
      <a class="nav-link" href="#demo">Live demo</a>
      <a class="nav-link" href="#experts">Experts</a>
      <a class="nav-link" href="#architecture">Architecture</a>
      <a class="nav-link" href="#benchmarks">Benchmarks</a>
      <a class="nav-link" href="#memory">Memory</a>
    </nav>
    <span class="badge-pill"><span class="pulse-dot"></span> <span id="health-status">offline-ready</span></span>
  </div>
</header>

<main>
  <!-- HERO -->
  <section class="hero">
    <div class="eyebrow">v0.1.0 · MIT licensed · 69 / 69 tests passing</div>
    <h1>Turn any <span class="grad">LLM</span> into an omni-modal orchestrator.</h1>
    <p class="lead">
      OmniFlow is a <strong>training-free, model-agnostic</strong> framework that lets a
      text LLM <strong>plan</strong> a directed acyclic graph of perception experts
      (OCR, layout, chart, caption, VQA, reasoning, ASR), <strong>resolve</strong>
      dependencies with topological sort, <strong>batch</strong> independent calls
      in parallel, and <strong>fuse</strong> their outputs into a single answer —
      without retraining a single weight.
    </p>
    <div class="cta-row">
      <a class="btn btn-primary" href="#demo">▶ Run a live query</a>
      <a class="btn btn-ghost" href="#architecture">View architecture</a>
      <a class="btn btn-ghost" href="/docs" target="_blank" rel="noopener">FastAPI docs ↗</a>
    </div>

    <div class="stat-strip">
      <div class="stat-card teal">
        <div class="accent-bar"></div>
        <div class="stat-label">Tests passing</div>
        <div class="stat-value">69<span style="color:var(--muted);font-size:18px">/69</span></div>
        <div class="stat-sub">offline · 0.48s suite</div>
      </div>
      <div class="stat-card amber">
        <div class="accent-bar"></div>
        <div class="stat-label">Expert modules</div>
        <div class="stat-value">7</div>
        <div class="stat-sub">OCR · layout · chart · caption · VQA · reason · ASR</div>
      </div>
      <div class="stat-card pink">
        <div class="accent-bar"></div>
        <div class="stat-label">Pillars</div>
        <div class="stat-value">3</div>
        <div class="stat-sub">LLM orchestration · ThinkOmni · Foresight</div>
      </div>
      <div class="stat-card teal">
        <div class="accent-bar"></div>
        <div class="stat-label">Latency (mock)</div>
        <div class="stat-value" id="hero-lat">— ms</div>
        <div class="stat-sub">end-to-end · deterministic</div>
      </div>
    </div>
  </section>

  <!-- LIVE DEMO -->
  <section class="section" id="demo">
    <div class="eyebrow">Live orchestration</div>
    <h2>Run a query through the pipeline.</h2>
    <p class="section-sub">
      Type any natural-language query and the controller will emit a control-token plan,
      resolve dependencies, batch parallel calls, dispatch experts, write cross-modal
      memory, and return a full trace. All offline — no PyTorch, no API keys.
    </p>

    <div class="demo-grid">
      <div class="demo-input">
        <label for="query">User query</label>
        <textarea id="query" placeholder="e.g. What does this bar chart show about quarterly revenue?">What does this bar chart show about quarterly revenue?</textarea>

        <label for="images" style="margin-top:14px">Image assets (comma-separated, optional)</label>
        <input id="images" type="text" placeholder="chart.png, scene.jpg" value="chart.png"
          style="width:100%;background:var(--bg-soft);color:var(--text);border:1px solid var(--border);border-radius:10px;padding:11px 14px;font-size:14px;font-family:var(--mono);" />

        <div class="chip-row" id="suggest-row">
          <!-- suggestion chips injected by JS -->
        </div>

        <div class="actions">
          <button class="btn btn-primary" id="run-btn">▶ Orchestrate</button>
          <button class="btn btn-ghost" id="bench-btn">⚡ Run benchmark</button>
          <span id="run-status" style="color:var(--muted);font-size:13px;"></span>
        </div>
      </div>

      <div class="demo-output">
        <h4>Orchestration trace</h4>
        <div class="kv-grid">
          <div class="kv"><div class="k">Total latency</div><div class="v" id="out-lat">—</div></div>
          <div class="kv"><div class="k">Expert calls</div><div class="v" id="out-calls">—</div></div>
          <div class="kv"><div class="k">Planning</div><div class="v" id="out-plan">—</div></div>
          <div class="kv"><div class="k">Steps</div><div class="v" id="out-steps">—</div></div>
        </div>

        <div class="plan-track" id="plan-track">
          <span style="color:var(--muted);font-size:12px;">Plan will appear here after a run.</span>
        </div>

        <div class="step-list" id="step-list">
          <div style="color:var(--muted);font-size:13px;padding:14px 0;">No runs yet — hit “Orchestrate”.</div>
        </div>
      </div>
    </div>
  </section>

  <!-- ARCHITECTURE / PILLARS -->
  <section class="section" id="architecture">
    <div class="eyebrow">Three orthogonal contributions</div>
    <h2>One framework, three pillars.</h2>
    <p class="section-sub">
      OmniFlow composes three independent ideas into a single system. Each pillar lives
      in its own sub-package and can be used in isolation — or wired together for the
      full training-free stack.
    </p>

    <div class="pillars" id="pillars-row">
      <div class="pillar t1">
        <span class="pillar-num">01</span>
        <div class="pillar-icon">🧭</div>
        <h3>LLM Orchestration</h3>
        <p class="pillar-tag">Control-token DAG planning</p>
        <p>The LLM emits a sequence of <span class="mono">ControlToken</span> instructions selecting training-free experts. The resolver performs topological sorting (OCR → layout → chart) and groups independent calls into parallel batches.</p>
        <span class="module-tag">omnimodal.orchestration</span>
      </div>
      <div class="pillar t2">
        <span class="pillar-num">02</span>
        <div class="pillar-icon">🧩</div>
        <h3>ThinkOmni Decoding</h3>
        <p class="pillar-tag">Adaptive contrastive fusion</p>
        <p>At decode time the LLM's next-token distribution is adaptively fused with a perceptual expert's distribution, with built-in conflict detection that gates the merge when experts disagree.</p>
        <span class="module-tag">omnimodal.decoding</span>
      </div>
      <div class="pillar t3">
        <span class="pillar-num">03</span>
        <div class="pillar-icon">🔭</div>
        <h3>Foresight Streaming</h3>
        <p class="pillar-tag">Dual-stream lookahead video-LLM</p>
        <p>A shared-encoder dual-stream video-LLM: a low-latency "now" stream + a lookahead "future" stream over the same backbone, with graceful frame-drop interpolation.</p>
        <span class="module-tag">omnimodal.streaming</span>
      </div>
    </div>

    <div class="arch-diagram" style="margin-top:18px;">
      <div class="arch-col">
        <h4>① Plan</h4>
        <div class="arch-stage">
          <div class="stage-label">User query</div>
          <div class="stage-val" id="arch-q">"What does this chart show?"</div>
        </div>
        <div class="arch-stage">
          <div class="stage-label">Controller</div>
          <div class="stage-val">LLMController.plan()</div>
        </div>
        <div class="arch-stage">
          <div class="stage-label">Control tokens</div>
          <div class="stage-val" id="arch-tokens">[ImageCaption, ChartReading, VQA]</div>
        </div>
      </div>
      <div class="arch-col">
        <h4>② Execute</h4>
        <div class="arch-stage">
          <div class="stage-label">Dependency resolve</div>
          <div class="stage-val">toposort + parallel batch</div>
        </div>
        <div class="arch-stage">
          <div class="stage-label">Parallel experts</div>
          <div class="stage-val" id="arch-stages">3 sequential stages</div>
        </div>
        <div class="arch-stage">
          <div class="stage-label">Cross-modal memory</div>
          <div class="stage-val">append after every call</div>
        </div>
      </div>
      <div class="arch-col">
        <h4>③ Answer</h4>
        <div class="arch-stage">
          <div class="stage-label">Terminal expert</div>
          <div class="stage-val">VQA / TextReason</div>
        </div>
        <div class="arch-stage">
          <div class="stage-label">Memory snapshot</div>
          <div class="stage-val" id="arch-mem">text + image + chart entries</div>
        </div>
        <div class="arch-stage">
          <div class="stage-label">Trace</div>
          <div class="stage-val">OrchestrationTrace</div>
        </div>
      </div>
    </div>
  </section>

  <!-- EXPERT POOL -->
  <section class="section" id="experts">
    <div class="eyebrow">Training-free expert pool</div>
    <h2>Seven swappable perception experts.</h2>
    <p class="section-sub">
      Each expert is a small, deterministic module with a typed contract. Real backends
      (Tesseract OCR, layoutparser, easyocr, transformers vision encoders, the z-ai SDK)
      can be plugged in via <span class="mono">backend=</span> — the framework ships with
      deterministic mock backends so the full pipeline runs offline.
    </p>
    <div class="experts-grid" id="experts-grid">
      <div style="color:var(--muted);grid-column:1/-1;padding:14px 0;">Loading experts…</div>
    </div>
  </section>

  <!-- BENCHMARKS -->
  <section class="section" id="benchmarks">
    <div class="eyebrow">Offline benchmark sweep</div>
    <h2>Five samples, deterministic mock experts.</h2>
    <p class="section-sub">
      Click “Run benchmark” to drive the full pipeline over five representative query
      categories (document understanding, chart QA, image captioning, math reasoning,
      video summary) and measure latency, expert-call count, and answer presence.
    </p>
    <div class="bench-wrap">
      <table class="bench">
        <thead>
          <tr><th>Category</th><th>Query</th><th class="num">Latency</th><th class="num">Calls</th><th class="num">Steps</th><th class="num">Answered</th></tr>
        </thead>
        <tbody id="bench-body">
          <tr><td colspan="6" style="color:var(--muted);padding:18px 16px;">No benchmark run yet.</td></tr>
        </tbody>
      </table>
      <div class="bench-summary">
        <div><div class="k">Samples</div><div class="v" id="bench-total">—</div></div>
        <div><div class="k">Answer rate</div><div class="v" id="bench-rate">—</div></div>
        <div><div class="k">Avg latency</div><div class="v" id="bench-avg">—</div></div>
      </div>
    </div>
  </section>

  <!-- MEMORY -->
  <section class="section" id="memory">
    <div class="eyebrow">Cross-modal working memory</div>
    <h2>What the pipeline currently remembers.</h2>
    <p class="section-sub">
      Every expert call writes a typed <span class="mono">MemoryEntry</span> into the
      cross-modal working memory. The terminal reasoning expert
      (<span class="mono">TextReason</span>) reads the whole snapshot to synthesise
      the final answer. Below is the live state — refreshes after each run.
    </p>
    <div class="memory-list" id="memory-list">
      <div style="color:var(--muted);padding:8px 4px;">No memory yet — run a query first.</div>
    </div>
  </section>
</main>

<footer class="site">
  <div class="footer-inner">
    <div class="copy">
      © 2026 OmniFlow Contributors · MIT License · built with FastAPI + Uvicorn
    </div>
    <div class="links">
      <a href="/docs" target="_blank" rel="noopener">API docs ↗</a>
      <a href="/api/health" target="_blank" rel="noopener">/api/health</a>
      <a href="https://github.com/BhaskarKaphle/omniflow" target="_blank" rel="noopener">GitHub ↗</a>
    </div>
  </div>
</footer>

<div class="toast" id="toast"></div>

<script>
(function () {
  "use strict";

  // ---------- helpers ----------
  const $ = (id) => document.getElementById(id);
  const toast = (msg, ok = true) => {
    const el = $("toast");
    el.textContent = msg;
    el.style.borderColor = ok ? "var(--accent)" : "var(--bad)";
    el.classList.add("show");
    clearTimeout(window.__toast_t);
    window.__toast_t = setTimeout(() => el.classList.remove("show"), 2600);
  };

  const SUGGESTIONS = [
    { q: "What does this bar chart show about quarterly revenue?", imgs: "chart.png" },
    { q: "Extract the text from this scanned invoice and identify line items.", imgs: "invoice.png" },
    { q: "Describe this scene in one sentence.", imgs: "scene.jpg" },
    { q: "If a train travels 80 km in 1.5 hours, what is its average speed?", imgs: "" },
    { q: "Summarise the events in this 10-second clip.", imgs: "clip.mp4" },
  ];

  // ---------- suggestion chips ----------
  const chipRow = $("suggest-row");
  SUGGESTIONS.forEach((s) => {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = s.q.length > 50 ? s.q.slice(0, 50) + "…" : s.q;
    b.title = s.q;
    b.addEventListener("click", () => {
      $("query").value = s.q;
      $("images").value = s.imgs;
    });
    chipRow.appendChild(b);
  });

  // ---------- experts ----------
  async function loadExperts() {
    try {
      const r = await fetch("/api/experts");
      const data = await r.json();
      const grid = $("experts-grid");
      grid.innerHTML = "";
      data.forEach((e) => {
        const card = document.createElement("div");
        card.className = "expert-card";
        const term = e.is_terminal ? '<span class="terminal-pill">terminal</span>' : "";
        const dep = e.dependency_index !== null
          ? `<span class="dep">depends on: <strong>${["OCR","LayoutAnalysis","ChartReading"][e.dependency_index]}</strong> (chain pos ${e.dependency_index})</span>`
          : '<span class="dep">no upstream dependency</span>';
        card.innerHTML = `
          <div class="expert-head">
            <span class="expert-name">${e.name}</span>
            <span class="expert-mod">${e.modality}</span>
          </div>
          <p>${e.capability}</p>
          ${dep}
          <div style="margin-top:8px;">${term}</div>
        `;
        grid.appendChild(card);
      });
    } catch (err) {
      console.error("experts load failed", err);
    }
  }

  // ---------- orchestrate ----------
  async function orchestrate() {
    const btn = $("run-btn");
    const status = $("run-status");
    const query = $("query").value.trim();
    const imagesRaw = $("images").value.trim();
    if (!query) { toast("Please type a query first.", false); return; }
    const images = imagesRaw ? imagesRaw.split(",").map(s => s.trim()).filter(Boolean) : [];

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Running…';
    status.textContent = "dispatching control tokens…";

    try {
      const t0 = performance.now();
      const r = await fetch("/api/orchestrate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, images }),
      });
      if (!r.ok) throw new Error("HTTP " + r.status);
      const data = await r.json();
      const elapsed = (performance.now() - t0).toFixed(1);

      // summary
      $("out-lat").textContent = data.summary.total_latency_ms + " ms";
      $("out-calls").textContent = String(data.summary.expert_calls);
      $("out-plan").textContent = data.summary.planning_ms + " ms";
      $("out-steps").textContent = String(data.summary.step_count);
      $("hero-lat").textContent = data.summary.total_latency_ms + " ms";

      // plan track
      const track = $("plan-track");
      track.innerHTML = "";
      if (data.plan && data.plan.length) {
        data.plan.forEach((tok, i) => {
          const span = document.createElement("span");
          span.className = "plan-step";
          span.textContent = `▢ ${tok.expert_type} · stage ${tok.sequence_order}`;
          track.appendChild(span);
          if (i < data.plan.length - 1) {
            const arrow = document.createElement("span");
            arrow.className = "plan-arrow";
            arrow.textContent = "→";
            track.appendChild(arrow);
          }
        });
      } else {
        track.innerHTML = '<span style="color:var(--muted);font-size:12px;">No control tokens emitted.</span>';
      }

      // steps
      const list = $("step-list");
      list.innerHTML = "";
      data.steps.forEach((s) => {
        const card = document.createElement("div");
        card.className = "step-card";
        const toks = s.control_tokens.map(t =>
          `<span class="expert-tag ${/VQA|TextReason/.test(t.expert) ? "terminal" : ""}">${t.expert}</span>`
        ).join("");
        const resps = s.expert_responses.map(r => {
          const out = typeof r.output === "string" ? r.output.slice(0, 120) : JSON.stringify(r.output).slice(0, 140);
          return `<div class="resp-line">↳ ${r.expert}: ${out} <span class="conf">conf=${r.confidence.toFixed(2)}</span> <span class="lat">${r.latency_ms}ms</span></div>`;
        }).join("");
        card.innerHTML = `
          <div class="step-head">
            <span class="step-id">${s.step_id}</span>
            <span style="color:var(--muted);font-size:11px;">${Object.entries(s.timing_ms || {}).map(([k,v]) => `${k}=${v.toFixed(2)}ms`).join(" · ") || "no timing"}</span>
          </div>
          <div class="step-toks">${toks}</div>
          ${resps}
        `;
        list.appendChild(card);
      });

      // arch diagram tokens
      $("arch-tokens").textContent = "[" + data.plan.map(t => t.expert_type).join(", ") + "]";
      $("arch-stages").textContent = `${data.summary.step_count} sequential stage${data.summary.step_count===1?"":"s"}`;
      $("arch-mem").textContent = `${data.memory.length} memory entries`;

      status.textContent = `✓ done in ${elapsed}ms (incl. HTTP)`;
      toast(`Orchestrated in ${data.summary.total_latency_ms}ms — ${data.summary.expert_calls} expert calls`);
      loadMemory();
    } catch (err) {
      console.error(err);
      status.textContent = "";
      toast("Orchestration failed: " + err.message, false);
    } finally {
      btn.disabled = false;
      btn.innerHTML = "▶ Orchestrate";
    }
  }

  // ---------- memory ----------
  async function loadMemory() {
    try {
      const r = await fetch("/api/memory");
      const data = await r.json();
      const list = $("memory-list");
      list.innerHTML = "";
      if (!data.entries.length) {
        list.innerHTML = '<div style="color:var(--muted);padding:8px 4px;">No memory yet — run a query first.</div>';
        return;
      }
      data.entries.slice().reverse().forEach((m) => {
        const row = document.createElement("div");
        row.className = "mem-row";
        row.innerHTML = `
          <span class="mem-src">${m.source_expert}</span>
          <span class="mem-mod">${m.modality}</span>
          <span class="mem-text">${m.text}</span>
        `;
        list.appendChild(row);
      });
    } catch (err) {
      console.error("memory load failed", err);
    }
  }

  // ---------- benchmarks ----------
  async function runBenchmark() {
    const btn = $("bench-btn");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Running…';
    try {
      const r = await fetch("/api/benchmarks");
      const data = await r.json();
      const body = $("bench-body");
      body.innerHTML = "";
      data.samples.forEach((s) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td class="cat">${s.category}</td>
          <td style="color:var(--muted);font-size:13px;">${s.query.length > 60 ? s.query.slice(0,60)+"…" : s.query}</td>
          <td class="num">${s.latency_ms} ms</td>
          <td class="num">${s.expert_calls}</td>
          <td class="num">${s.step_count}</td>
          <td class="num" style="color:${s.answered ? "var(--good)" : "var(--bad)"};">${s.answered ? "✓" : "✗"}</td>
        `;
        body.appendChild(tr);
      });
      $("bench-total").textContent = String(data.total_samples);
      $("bench-rate").textContent = (data.answer_rate * 100).toFixed(0) + "%";
      $("bench-avg").textContent = data.avg_latency_ms + " ms";
      toast(`Benchmark done — ${data.answered}/${data.total_samples} answered`);
    } catch (err) {
      console.error(err);
      toast("Benchmark failed: " + err.message, false);
    } finally {
      btn.disabled = false;
      btn.innerHTML = "⚡ Run benchmark";
    }
  }

  // ---------- health ----------
  async function checkHealth() {
    try {
      const r = await fetch("/api/health");
      const d = await r.json();
      $("health-status").textContent = `v${d.version} · ${d.use_real_llm ? "real LLM" : "offline mock"}`;
    } catch {
      $("health-status").textContent = "checking…";
    }
  }

  // ---------- wire up ----------
  $("run-btn").addEventListener("click", orchestrate);
  $("bench-btn").addEventListener("click", runBenchmark);
  $("query").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); orchestrate(); }
  });

  checkHealth();
  loadExperts();
  loadMemory();
})();
</script>
</body>
</html>
"""
