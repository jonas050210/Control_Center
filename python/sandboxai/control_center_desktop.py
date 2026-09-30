"""Tk desktop Control Center backed exclusively by :mod:`sandboxai.adapter`.

Tk is a small, supported local desktop dependency on Windows and ships with
CPython.  The GUI is intentionally a thin view/controller: all values and
mutations go through the adapter, and polling is performed off the Tk thread.
"""
from __future__ import annotations
import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

from .adapter import SandboxAIAdapter
from .config import TrainingConfig

PAGES = ("Dashboard", "Training", "Agents", "Benchmarks", "Evaluations", "Runs / Checkpoints", "System / Telemetry", "Settings")

class ControlCenter(tk.Tk):
    def __init__(self, adapter: SandboxAIAdapter | None = None) -> None:
        super().__init__()
        self.title("SandboxAI Control Center")
        self.geometry("1280x820")
        self.minsize(980, 620)
        self.adapter = adapter or SandboxAIAdapter()
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._active_process: str | None = None
        self.vars: dict[str, tk.StringVar] = {}
        self._build()
        self.after(150, self._consume)
        self.after(1000, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _build(self) -> None:
        style = ttk.Style(self); style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Section.TLabel", font=("Segoe UI", 11, "bold"))
        outer = ttk.Frame(self, padding=12); outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer); header.pack(fill="x")
        ttk.Label(header, text="SANDBOXAI  /  CONTROL CENTER", style="Title.TLabel").pack(side="left")
        self.state = ttk.Label(header, text="READY", foreground="#216e39"); self.state.pack(side="right")
        body = ttk.Frame(outer); body.pack(fill="both", expand=True, pady=(12, 0))
        nav = ttk.Frame(body, width=190); nav.pack(side="left", fill="y", padx=(0, 12)); nav.pack_propagate(False)
        self.content = ttk.Frame(body); self.content.pack(side="left", fill="both", expand=True)
        for page in PAGES:
            ttk.Button(nav, text=page, command=lambda p=page: self.show(p)).pack(fill="x", pady=2)
        ttk.Separator(nav).pack(fill="x", pady=12)
        ttk.Label(nav, text="Real backend\nNo simulated values", foreground="#555").pack(anchor="w")
        self.pages = {p: ttk.Frame(self.content) for p in PAGES}
        self.show("Dashboard")

    def _clear(self, page: str) -> ttk.Frame:
        frame = self.pages[page]
        for child in frame.winfo_children(): child.destroy()
        frame.pack_forget(); frame.pack(fill="both", expand=True)
        return frame

    def show(self, page: str) -> None:
        for frame in self.pages.values(): frame.pack_forget()
        frame = self.pages[page]; frame.pack(fill="both", expand=True)
        builders = {"Dashboard": self.dashboard, "Training": self.training, "Benchmarks": self.benchmarks,
                    "Evaluations": self.evaluations, "Runs / Checkpoints": self.runs,
                    "System / Telemetry": self.system, "Agents": self.agents, "Settings": self.settings}
        builders[page](frame)

    def _heading(self, frame: ttk.Frame, title: str, subtitle: str) -> None:
        ttk.Label(frame, text=title, style="Title.TLabel").pack(anchor="w")
        ttk.Label(frame, text=subtitle, foreground="#666").pack(anchor="w", pady=(2, 12))

    def _cards(self, frame: ttk.Frame, values: dict[str, Any]) -> None:
        grid = ttk.Frame(frame); grid.pack(fill="x", pady=4)
        for i, (key, value) in enumerate(values.items()):
            box = ttk.LabelFrame(grid, text=key.replace("_", " ").title(), padding=10)
            box.grid(row=0, column=i, sticky="nsew", padx=(0, 8)); grid.columnconfigure(i, weight=1)
            ttk.Label(box, text="N/A" if value is None else str(value), font=("Segoe UI", 13, "bold")).pack()

    def dashboard(self, frame: ttk.Frame) -> None:
        frame = self._clear("Dashboard"); self._heading(frame, "Dashboard", "Live process state and measured project data")
        data = self.adapter.project_status(); active = data.get("active_processes", [])
        latest = (data.get("runs", {}).get("runs") or [{}])[-1]
        status = latest.get("status", {}) if isinstance(latest, dict) else {}
        progress = latest.get("progress", {}) if isinstance(latest, dict) else {}
        self._cards(frame, {"training state": status.get("state", "No runs"), "active process": len(active),
                            "timestep": progress.get("timesteps"), "target": progress.get("target_timesteps")})
        text = tk.Text(frame, height=20, state="normal", wrap="word"); text.pack(fill="both", expand=True, pady=12)
        text.insert("end", json.dumps({"active_processes": active, "latest_run": latest}, indent=2, default=str)); text.configure(state="disabled")

    def _field(self, parent: ttk.Frame, name: str, default: str) -> tk.StringVar:
        var = tk.StringVar(value=default); self.vars[name] = var
        row = ttk.Frame(parent); row.pack(fill="x", pady=3); ttk.Label(row, text=name.replace("_", " ").title(), width=24).pack(side="left")
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True); return var

    def training(self, frame: ttk.Frame) -> None:
        frame = self._clear("Training"); self._heading(frame, "Training", "Launch the existing PPO trainer; advanced options remain in a config file")
        form = ttk.LabelFrame(frame, text="Core configuration", padding=12); form.pack(fill="x")
        self._field(form, "environment_count", "8"); self._field(form, "env_workers", "1"); self._field(form, "total_training_steps", "1000000")
        self._field(form, "rollout_length", "0"); self._field(form, "device", "auto"); self._field(form, "torch_threads", "0")
        actions = ttk.Frame(frame); actions.pack(fill="x", pady=12)
        ttk.Button(actions, text="START TRAINING", command=self._start_training).pack(side="left")
        ttk.Button(actions, text="STOP SAFELY", command=self._stop).pack(side="left", padx=8)
        self.log = tk.Text(frame, height=18, state="disabled"); self.log.pack(fill="both", expand=True)

    def _start_training(self) -> None:
        try:
            values = {k: int(v.get()) if k not in {"device"} else v.get() for k, v in self.vars.items()}
            values.update(output_root="training", run_id="control-" + __import__("time").strftime("%Y%m%d-%H%M%S"))
            result = self.adapter.start_training(TrainingConfig.from_dict(values))
            self._active_process = result["process_id"]; self.state.configure(text="TRAINING", foreground="#9a6700")
            self._append(json.dumps(result, indent=2, default=str))
        except Exception as exc: messagebox.showerror("Training could not start", str(exc))

    def _stop(self) -> None:
        if self._active_process: self.adapter.cancel(self._active_process)

    def _append(self, value: str) -> None:
        if hasattr(self, "log"): self.log.configure(state="normal"); self.log.insert("end", value + "\n"); self.log.see("end"); self.log.configure(state="disabled")

    def runs(self, frame: ttk.Frame) -> None:
        frame = self._clear("Runs / Checkpoints"); self._heading(frame, "Runs / Checkpoints", "Read-only inventory from run_inspection.py")
        rows = self.adapter.list_runs().get("runs", [])
        tree = ttk.Treeview(frame, columns=("state", "steps", "checkpoint"), show="headings"); tree.pack(fill="both", expand=True)
        for col in tree["columns"]: tree.heading(col, text=col.title())
        for row in rows: tree.insert("", "end", values=(row.get("status", {}).get("state"), row.get("progress", {}).get("timesteps"), row.get("checkpoints", {}).get("count")))

    def benchmarks(self, frame: ttk.Frame) -> None:
        frame = self._clear("Benchmarks"); self._heading(frame, "Benchmarks", "Measured benchmark artifacts; start uses benchmark_simulation")
        self._cards(frame, {"latest": self.adapter.benchmark_results().get("directory"), "status": "not running"})
        ttk.Button(frame, text="RUN BENCHMARK", command=self._start_benchmark).pack(anchor="w", pady=8)
        self._json(frame, self.adapter.benchmark_results())

    def _start_benchmark(self) -> None:
        result = self.adapter.start_benchmark(environment_counts=(1, 2, 4, 8), steps=2000); self._active_process = result["process_id"]; self.state.configure(text="BENCHMARK")

    def evaluations(self, frame: ttk.Frame) -> None:
        frame = self._clear("Evaluations"); self._heading(frame, "Evaluations", "Evaluate a selected frozen checkpoint with the existing evaluator")
        path = self._field(frame, "checkpoint", ""); self._field(frame, "episodes", "20"); self._field(frame, "environment_count", "1")
        ttk.Button(frame, text="CHOOSE CHECKPOINT", command=lambda: path.set(filedialog.askopenfilename(filetypes=[("Checkpoints", "*.zip")]))).pack(anchor="w", pady=5)
        ttk.Button(frame, text="START EVALUATION", command=lambda: self._start_evaluation(path.get())).pack(anchor="w")

    def _start_evaluation(self, checkpoint: str) -> None:
        if not checkpoint: messagebox.showwarning("Checkpoint required", "Select an existing checkpoint."); return
        self._active_process = self.adapter.start_evaluation(checkpoint)["process_id"]; self.state.configure(text="EVALUATION")

    def system(self, frame: ttk.Frame) -> None:
        frame = self._clear("System / Telemetry"); self._heading(frame, "System / Telemetry", "Machine probes are best effort; unavailable values remain unavailable")
        self._json(frame, self.adapter.system_status())

    def agents(self, frame: ttk.Frame) -> None:
        frame = self._clear("Agents"); self._heading(frame, "Agents", "Worker/process state from the adapter")
        self._json(frame, self.adapter.project_status().get("active_processes", []))

    def settings(self, frame: ttk.Frame) -> None:
        frame = self._clear("Settings"); self._heading(frame, "Settings", "Project and data roots used by the adapter")
        self._json(frame, {"project_root": str(self.adapter.project_root), "output_root": str(self.adapter.output_root)})

    def _json(self, frame: ttk.Frame, value: Any) -> None:
        box = tk.Text(frame, wrap="none"); box.pack(fill="both", expand=True); box.insert("end", json.dumps(value, indent=2, default=str)); box.configure(state="disabled")

    def _poll(self) -> None:
        if self._active_process:
            self._queue.put(("process", self.adapter.process_status(self._active_process)))
        threading.Thread(target=lambda: self._queue.put(("dashboard", self.adapter.project_status())), daemon=True).start()
        self.after(1000, self._poll)

    def _consume(self) -> None:
        try:
            while True:
                kind, value = self._queue.get_nowait()
                if kind == "process":
                    self._append(json.dumps(value, default=str))
                    if value.get("state") in {"finished", "failed"}: self.state.configure(text=value["state"].upper())
        except queue.Empty: pass
        self.after(150, self._consume)

    def _close(self) -> None:
        self.adapter.close(); self.destroy()

def main() -> int:
    app = ControlCenter(); app.mainloop(); return 0

if __name__ == "__main__": raise SystemExit(main())
