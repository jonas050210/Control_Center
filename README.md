# NEURAL ARENA · Control Center

Headless 3D-Shooter-Trainingssandbox (Gymnasium + Stable-Baselines3 PPO) mit einer
eigenen Web-Oberfläche. **Simulation, Physik und Training laufen in Python; alle
3D-Szenen und Diagramme rendert der Browser** (three.js/WebGL + Canvas 2D).

* Kein Streamlit, kein Plotly, kein pandas, kein Pygame, kein X11/Display nötig.
* Läuft komplett offline, sobald `install.py` einmal durchgelaufen ist.
* Entwickelt für WSL-Ubuntu mit CPU-Training; bedient wird über den Browser.

## 1. Schnellstart

```bash
cd /pfad/zu/Control_Center
python3 install.py        # legt .venv an, installiert alles, lädt die Web-Assets
python3 start.py          # startet den Server und öffnet den Browser
```

`install.py` und `start.py` sind eigenständige Skripte ohne Fremdabhängigkeiten:
Sie laufen mit dem System-Python und wechseln danach automatisch in `.venv`.

Danach läuft das Control Center unter **http://127.0.0.1:8501**.
Beenden mit **STRG+C**.

### Optionen

```bash
python3 install.py --skip-torch      # ohne PyTorch/SB3 (kein PPO-Training)
python3 install.py --no-venv         # in den aktuellen Interpreter installieren
python3 install.py --force-assets    # three.js erneut herunterladen
python3 install.py --remove-legacy   # Streamlit/Plotly/pandas ohne Rückfrage löschen
python3 install.py --keep-legacy     # nicht nach dem Aufräumen von Altlasten fragen

python3 start.py --port 8600         # anderer Port (belegte Ports werden übersprungen)
python3 start.py --host 0.0.0.0      # Zugriff aus Windows/LAN erlauben
python3 start.py --no-browser        # Browser nicht automatisch öffnen
python3 start.py --reload            # Auto-Reload für die Entwicklung
```

## 2. WSL-Ubuntu Hinweise

```bash
sudo apt update && sudo apt install -y python3-venv python3-dev build-essential
```

* **Browser öffnen:** `start.py` erkennt WSL und nutzt `wslview`, `cmd.exe` bzw.
  `powershell.exe`, um die Seite im Windows-Browser zu öffnen. Klappt das nicht,
  einfach die angezeigte URL manuell aufrufen – die Ausgabe nennt sie immer.
* **Speicherort:** Das Projekt darf unter `/mnt/c/...` liegen; die virtuelle
  Umgebung `.venv` gehört in das Projektverzeichnis oder nach `~/.venvs`
  (nichts, was durch OneDrive synchronisiert wird).
* **Windows-Firewall:** Mit `--host 0.0.0.0` ist der Server aus Windows erreichbar;
  Windows fragt ggf. nach einer Freigabe für Python.
* **GPU:** Das Training läuft bewusst auf der CPU (`install.py` installiert das
  CPU-Wheel von PyTorch), damit die GPU für andere Arbeit frei bleibt.

## 3. Panels

| Panel | Zweck |
| --- | --- |
| 🎮 ARENA | Zwei Kämpfer im Duell: Bots, trainierte Policies oder der Mensch steuern einen Agenten, Live-3D, Trails, Trefferzonen. |
| 🔫 PLAYGROUND | Freies Üben mit Tastatursteuerung (inkl. Springen, Ducken, Sprinten, Lean), Trefferstatistik und Demo-Aufzeichnung für Imitation Learning. |
| 🏋️ TRAINING | PPO-Training im Hintergrund-Thread: Start/Pause/Resume/Stop, Checkpoints, Curriculum-Phasen, Live-Metriken. |
| 📊 STATS | Kennzahlen aus `logs/`: FPS, Reward, Win-Rate, Kill-TTK, Headshots, Waffen-/Gegner-Verteilung. |
| 🔧 BENCHMARK | 20-Sekunden-Durchsatztest über alle gültigen CPU-Konfigurationen (`workers × envs`), Ergebnis-Ranking. |
| 🎯 TTK-TESTER | Time-to-Kill-Simulation über Waffen, Distanzen und Trials inkl. Lua-Export für Roblox. |
| 🗺️ MAPS | Sechs Karten ansehen, Randomisieren, Deckung platzieren/löschen, Custom Map importieren/exportieren. |
| 🔥 HEATMAP | Kill-/Death-Zonen pro Karte, gefiltert nach Waffe, Episode und Distanz. |

Dazu die kleinen Panels für Statushinweise: Kopfzeile mit Server-/Trainingsstatus.

## 4. Projektstruktur

```text
.
├── install.py              # Einrichtung: venv, Pakete, Web-Assets, Selbsttest
├── start.py                # Start: uvicorn, Browser öffnen, WSL-Erkennung
├── requirements.txt        # Laufzeit (Simulation + Backend)
├── requirements-training.txt  # optional: PyTorch (CPU) + Stable-Baselines3
├── env/
│   ├── shooter_env.py      # Gymnasium-Umgebung: Duell, MultiDiscrete-Aktionen, 31er-Observation
│   ├── weapons.py          # sechs Waffen mit Magazin, Reload, Rückstoß
│   ├── maps.py             # sechs Karten, Deckung, Spawns, Randomisierung
│   ├── map_io.py           # validierter, atomarer Custom-Map-Import/-Export
│   └── physics.py          # Bewegung, Stance, Gravitation, Kollision, Raycast
├── training/
│   ├── train.py            # PPO im Hintergrund-Thread, Curriculum, Checkpoints, CSV-Logging
│   ├── workers.py          # CPU-Kernwahl, VecEnv-Aufbau, Benchmark-Runner, CPU-Job-Lock
│   ├── rewards.py          # nachvollziehbares Reward-Shaping
│   ├── imitation.py        # Behavior Cloning aus aufgezeichneten Demos
│   └── weapon_lab.py       # TTK-Simulation + Lua-Export
├── server/                 # FastAPI-Backend (JSON-API, hält den gesamten Zustand)
│   ├── app.py              # Routen
│   ├── state.py            # Session-Zustand, Arena-/Playground-/Trainingssteuerung
│   ├── scene.py            # Szenen- und Frame-Payloads für den Renderer
│   ├── analytics.py        # Stats, Heatmap, Benchmark-Aufbereitung
│   ├── minigames.py        # Aim-Driller und Dodge-Grid
│   ├── actions.py          # Tastenzuordnung → Aktionsvektor
│   ├── policies.py         # Modell-Cache für trainierte Checkpoints
│   └── config.py           # Pfade, Presets, Abhängigkeitsstatus
├── web/                    # Frontend ohne Build-Schritt (ES-Module)
│   ├── index.html          # Importmap auf web/vendor/three.module.min.js
│   ├── css/style.css       # Cyberpunk-Grün-Palette
│   ├── js/
│   │   ├── scene.js        # Arena3D: three.js-Renderer für Matches, Maps, Zielscheiben, Dodge
│   │   ├── charts.js       # Canvas-2D-Diagramme (ersetzt Plotly)
│   │   ├── app.js          # Navigation, Panels, Toasts, Statuspoller
│   │   └── panels/*.js     # acht Panels
│   └── vendor/             # three.js (von install.py geladen, CDN als Fallback)
├── data/demos.csv          # Start-Datensatz für Imitation Learning (41 Spalten)
├── models/                 # Checkpoints (*.zip) und best_model.json
├── logs/                   # training_metrics.csv, heatmap_events.csv
└── tests/                  # unittest-Suite, headless (ohne Browser/GPU)
```

## 5. Tests

```bash
python3 -m unittest discover -s tests -v      # 56 Tests in unter einer Sekunde
```

Die Suite deckt Umgebung/Physik, Karten-JSON, Rewards, Minigames, TTK-Simulation,
den CPU-Job-Lock, alle API-Routen (FastAPI `TestClient`), Szenen-Payloads und die
Analytics-Aufbereitung ab. Das PPO-Integrationstest läuft nur mit installiertem
Stable-Baselines3 und wird sonst übersprungen.

## 6. API

Alles, was die Oberfläche tut, geht über JSON-Routen unter `/api/...`
(u. a. `/api/health`, `/api/meta`, `/api/arena/*`, `/api/playground/*`,
`/api/training/*`, `/api/stats`, `/api/heatmap`, `/api/benchmark/*`,
`/api/ttk/simulate`, `/api/maps/*`). Die interaktive Dokumentation liefert
**http://127.0.0.1:8501/api/docs** (Swagger UI von FastAPI).

## 7. Aufräumen / Analyse

`ANALYSE.md` enthält die Deep-Analyse des Projekts: was erledigt ist, welche
Bugs gefunden wurden, was als Nächstes sinnvoll ist und was gelöscht werden darf
(z. B. noch installierte Streamlit-/Plotly-/pandas-Reste).
