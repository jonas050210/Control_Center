"""Zero-dependency local monitoring dashboard and JSON API."""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from monitoring.experiments import ExperimentTracker
from monitoring.state import SystemTelemetry

DASHBOARD_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>SandboxAI Control Center</title><style>
:root{color-scheme:dark;--bg:#0b1018;--card:#141c28;--line:#273449;--accent:#59d8a1;--muted:#92a2b8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#ecf4ff;font:14px system-ui,sans-serif}
header{padding:22px 5vw;border-bottom:1px solid var(--line);display:flex;justify-content:space-between}
h1{margin:0;font-size:23px}.stage{color:var(--accent);font-weight:700}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;padding:20px 5vw}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}.wide{grid-column:1/-1}h2{font-size:13px;text-transform:uppercase;color:var(--muted);letter-spacing:.12em;margin:0 0 12px}.metric{font-size:28px;font-weight:700}.sub{color:var(--muted);line-height:1.7}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:8px;border-bottom:1px solid var(--line)}canvas{width:100%;height:180px}code{font-size:12px;color:#bdebd8}.ok{color:var(--accent)}
</style></head><body><header><h1>SandboxAI <span class="sub">Control Center</span></h1><div id="stage" class="stage">LOADING</div></header>
<main class="grid"><section class="card"><h2>Dataset</h2><div id="steps" class="metric">—</div><div id="dataset" class="sub"></div></section>
<section class="card"><h2>Behavior cloning</h2><div id="bc" class="metric">—</div><div id="bcsub" class="sub"></div></section>
<section class="card"><h2>Reinforcement learning</h2><div id="rl" class="metric">—</div><div id="rlsub" class="sub"></div></section>
<section class="card"><h2>Runtime</h2><div id="runtime" class="metric">—</div><div id="action" class="sub"></div></section>
<section class="card wide"><h2>Latest policy benchmark</h2><canvas id="chart" width="1000" height="180"></canvas><table><thead><tr><th>Policy</th><th>Reward</th><th>Hits</th><th>Kills</th><th>Accuracy</th><th>Health</th></tr></thead><tbody id="benchmark"></tbody></table></section>
<section class="card wide"><h2>Recent experiments</h2><table><thead><tr><th>Run</th><th>Kind</th><th>Status</th><th>Started</th></tr></thead><tbody id="runs"></tbody></table></section></main>
<script>
function text(id,v){document.getElementById(id).textContent=v}
function chart(rows){const c=document.getElementById('chart'),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);if(!rows.length)return;const vals=rows.map(r=>r[1].mean_reward),lo=Math.min(0,...vals),hi=Math.max(1,...vals),span=hi-lo;rows.forEach((r,i)=>{const y=25+i*35,w=(r[1].mean_reward-lo)/span*700;x.fillStyle='#273449';x.fillRect(150,y,700,22);x.fillStyle='#59d8a1';x.fillRect(150,y,w,22);x.fillStyle='#ecf4ff';x.font='14px system-ui';x.fillText(r[0].toUpperCase(),8,y+16);x.fillText(r[1].mean_reward.toFixed(3),865,y+16)})}
async function refresh(){try{const [s,r]=await Promise.all([fetch('/api/state').then(x=>x.json()),fetch('/api/experiments').then(x=>x.json())]);text('stage',s.pipeline_stage);text('steps',(s.datasets?.total_steps||0).toLocaleString()+' steps');text('dataset',`${s.datasets?.total_sessions||0} sessions · ${s.datasets?.total_mb||0} MB`);text('bc',s.bc?.best_val_loss==null?'not trained':Number(s.bc.best_val_loss).toFixed(4));text('bcsub',s.bc?.active_checkpoint||'No checkpoint');text('rl',(s.rl?.timesteps||0).toLocaleString()+' steps');text('rlsub',`${s.rl?.steps_per_sec||0} steps/s · ${s.rl?.active_checkpoint||'No checkpoint'}`);text('runtime',`${s.runtime?.steps_per_sec||0} steps/s`);text('action',JSON.stringify(s.runtime?.last_action||{}));const rows=Object.entries(s.evaluation?.latest_benchmark||{}).filter(x=>x[1]?.mean_reward!==undefined);document.getElementById('benchmark').innerHTML=rows.map(([n,m])=>`<tr><td>${n.toUpperCase()}</td><td>${m.mean_reward} ± ${m.std_reward}</td><td>${m.total_hits}</td><td>${m.total_kills||0}</td><td>${m.hit_rate_pct}%</td><td>${m.mean_final_health??'—'}</td></tr>`).join('');chart(rows);document.getElementById('runs').innerHTML=r.slice(0,10).map(v=>`<tr><td><code>${v.run_id}</code></td><td>${v.kind}</td><td class="ok">${v.status}</td><td>${v.started_at}</td></tr>`).join('')}catch(e){text('stage','OFFLINE')}}refresh();setInterval(refresh,2000)
</script></body></html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    state_file = Path("logs/system_state.json")
    experiments_dir = Path("logs/experiments")

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = DASHBOARD_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/state":
            self._json(SystemTelemetry(self.state_file).state)
            return
        if parsed.path == "/api/experiments":
            self._json(ExperimentTracker.list_runs(self.experiments_dir))
            return
        if parsed.path == "/api/metrics":
            run_id = parse_qs(parsed.query).get("run_id", [""])[0]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
                self._json({"error": "invalid run_id"}, 400)
                return
            path = self.experiments_dir / run_id / "metrics.jsonl"
            if not path.is_file():
                self._json([], 404)
                return
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            self._json(events)
            return
        self._json({"error": "not found"}, 404)

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    state_file: str = "logs/system_state.json",
    experiments_dir: str = "logs/experiments",
) -> None:
    DashboardHandler.state_file = Path(state_file)
    DashboardHandler.experiments_dir = Path(experiments_dir)
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"SandboxAI dashboard: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local SandboxAI dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state_file", default="logs/system_state.json")
    parser.add_argument("--experiments_dir", default="logs/experiments")
    args = parser.parse_args()
    serve(args.host, args.port, args.state_file, args.experiments_dir)


if __name__ == "__main__":
    main()
