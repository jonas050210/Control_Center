# NEURAL ARENA · Control Center

Headless 3D-Shooter-Trainingssandbox (Gymnasium + Stable-Baselines3 PPO) mit einer
eigenen Web-Oberfläche. **Simulation, Physik und Training laufen in Python; alle
3D-Szenen und Diagramme rendert der Browser** (three.js/WebGL + Canvas 2D).

**Die AI sieht nur, was sie sehen kann:** keine Gegner-Koordinaten, sondern grobe
Richtungs-Sektoren, Entfernungs- und HP-Bänder, Sichtlinie und Gedächtnis – sie
kann nicht durch Wände zielen (Details unter „5. Wahrnehmung der AI").

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
│   ├── rewards.py          # nachvollziehbares Reward-Shaping (Zielbonus nur bei Sicht)
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
├── tools/evaluate_policy.py # Checkpoint/Zufallspolitik gegen den Bot messen
├── data/demos.csv          # Start-Datensatz für Imitation Learning (41 Spalten)
├── models/                 # Checkpoints (*.zip) und best_model.json
├── logs/                   # training_metrics.csv, heatmap_events.csv
└── tests/                  # unittest-Suite, headless (ohne Browser/GPU)
    └── dom/                # optionale jsdom-Tests für Frontend und 3D-Geometrie
```

## 5. Wahrnehmung der AI (Observation-Version 2)

Die Beobachtung hat 31 Werte (`OBSERVATION_SIZE`), aber der Gegner wird **nicht**
mehr als exakte Position geliefert. Stattdessen (Indizes 5–12):

| Wert | Bedeutung |
| --- | --- |
| `enemy_bearing_sin/cos` | Richtung zum Gegner, auf 12 Sektoren (30°) gerundet – null Information, wenn er nie gesehen wurde |
| `enemy_distance_band` | Entfernungsband: <5, 5–10, 10–20, 20–35, 35–60, >60 m |
| `enemy_visible` | Sichtkontakt jetzt? (120°-Sichtkegel **und** freie Sichtlinie auf Augenhöhe) |
| `enemy_hp_band` | Gegner-HP in 4 Bändern (>75 %, 50–75 %, 25–50 %, <25 %), unbekannt = 0 |
| `enemy_time_since_seen` | −1 = nie/verblasst, 0 = jetzt, 1 = vor 5 s (Gedächtnis, danach vergessen) |
| `enemy_memory_sin/cos` | exakte Richtung des letzten Sichtkontakts (nur Gesehenes, 0/0 wenn nichts) |
| `enemy_alive` (Index 22) | lebt der Gegner? |

Dazu die eigenen Werte (Position, Blickrichtung, 8 Wand-Raycasts, HP, Munition,
Bewegungszustand, Timer). Der Zielbonus im Reward greift nur bei Sichtkontakt –
belohnt wird also nichts, was der Agent nicht wissen kann.

**Vier Modi** (`vision_mode`, im ARENA- und TRAINING-Panel wählbar):

* `coarse_los` (**Standard**): echte Sicht – Deckung blendet den Gegner wirklich aus.
* `coarse`: Sektor/Bänder, aber der Gegner gilt immer als verfolgt.
* `noisy`: exakte Werte plus Rauschen (Zwischenstufe fürs Training).
* `exact`: altes Verhalten (kennt Position durch Wände) – nur zum Vergleich.

Checkpoints tragen einen Stempel (`*_meta.json` mit `observation_version` und
`vision_mode`); alte Modelle werden mit klarer Meldung abgelehnt statt falsch
benutzt. Ein unbekannter Wert für `vision` wird mit HTTP 400 abgelehnt (kein
stilles Zurückfallen auf den Standard).

### Trainingsrezept

| Hebel | Standard | Warum |
| --- | --- | --- |
| `gamma` | **0.995** | 15 Entscheidungen/s × 0,99 = nur ~100 Steps Horizont; Suchen+Zielen+Töten dauert länger. 0.995 ≈ 200 Steps. |
| Episodenlänge | 60 s (im Panel einstellbar) | kürzere Runden = mehr Kämpfe pro Minute |
| `curriculum_min_win_rate` | 35–40 % **Kill**-Rate | die nächste Phase wird erst freigeschaltet, wenn die aktuelle wirklich gewonnen wird |
| Startabstand der Phasen | 25 / 45 / 70 / 100 % der Kartendistanz | auf Kartendistanz (Dust: 36 m) treffen selbst perfekt zielende Schützen nur 0–1,8 % – eine frühe Phase dort kann keinen Kill lehren |
| „Bestes Modell“ | höchste Kill-Rate | ein Sieg nach HP-Vergleich am Zeitlimit wäre eine Belohnung fürs Verstecken |

Das Training protokolliert **Siege und Kills getrennt** (`win_rate`, `kill_rate`):
ein „Sieg“ am Zeitlimit ist nur ein HP-Vergleich und kein Kill.

## 6. Tests

```bash
python3 -m unittest discover -s tests -v      # 87 Tests in ~20 s
```

Die Suite deckt Umgebung/Physik, Karten-JSON, Rewards, Minigames, TTK-Simulation,
den CPU-Job-Lock, alle API-Routen (FastAPI `TestClient`), Szenen-Payloads, die
Analytics-Aufbereitung und die **Wahrnehmung** ab (`test_perception.py`: Sektoren,
Bänder, Sichtkegel, Deckung, Gedächtnis, Leak-Tests). Mit installiertem
Stable-Baselines3 läuft zusätzlich der PPO-Integrationstest (`test_training_smoke.py`).

### Trainieren und bewerten

```bash
# im TRAINING-Panel der Oberfläche: Dauer, Karte, Sichtmodus, Episodenlänge, Start
python3 tools/evaluate_policy.py --random --episodes 20          # Zufalls-Baseline
python3 tools/evaluate_policy.py --model best_model.zip --episodes 30
```

`tools/evaluate_policy.py` spielt nur ab (kein Training) und berichtet Siege,
Unentschieden, Niederlagen, Time-to-Kill (nur bestätigte Kills), den Blindanteil
pro Episode und prüft den `observation_version`-Stempel des Checkpoints. Mit
`--bots stationary|walker|shooter|full` lässt sich der Gegner festlegen und mit
`--phase 1..4` die Curriculum-Bedingungen (Abstand) nachstellen, unter denen ein
Checkpoint trainiert wurde. Mehrere Seeds vergleicht `tools/seed_sweep.py`.

Gemessen auf dieser Maschine (CPU, 2 Worker, `coarse_los`): Zufallspolitik
0/20 Siege – trainierter Checkpoint unter Phase-1-Bedingungen 10/10 Siege mit
bestätigten Kills (TTK ≈ 20 s), auf Kartendistanz 0 Kills (dort trifft niemand,
siehe ANALYSE.md).

Aufgezeichnete Demos (`data/demos.csv`) tragen einen Sidecar
`data/demos_meta.json` mit `observation_version`; ein veralteter Datensatz wird
beim Laden abgelehnt, statt Behavioral Cloning mit falschen Spalten zu füttern.

## 7. API

Alles, was die Oberfläche tut, geht über JSON-Routen unter `/api/...`
(u. a. `/api/health`, `/api/meta`, `/api/arena/*`, `/api/playground/*`,
`/api/training/*`, `/api/stats`, `/api/heatmap`, `/api/benchmark/*`,
`/api/ttk/simulate`, `/api/maps/*`). Die interaktive Dokumentation liefert
**http://127.0.0.1:8501/api/docs** (Swagger UI von FastAPI).

## 8. Aufräumen / Analyse

`ANALYSE.md` enthält die Deep-Analyse des Projekts: was erledigt ist, welche
Bugs gefunden wurden, was als Nächstes sinnvoll ist und was gelöscht werden darf
(z. B. noch installierte Streamlit-/Plotly-/pandas-Reste).
