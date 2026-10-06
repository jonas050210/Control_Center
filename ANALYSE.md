# NEURAL ARENA · Deep-Analyse

Stand: Umbau abgeschlossen (Streamlit vollständig entfernt). Dieses Dokument ist
die geforderte Analyse: was geändert wurde, welche Bugs gefunden und behoben
wurden, was noch offen ist, was verbessert werden kann und was gelöscht gehört.

---

## 1. Umbau: Streamlit raus, FastAPI + WebGL rein

**Vorher:** acht Streamlit-Tabs (`gui/`), alle 3D-Szenen als Plotly-Figuren
serverseitig gebaut, Diagramme als Plotly, Zustand in `st.session_state`,
Start über `streamlit run gui/app.py`.

**Jetzt:**

| Schicht | Vorher | Jetzt |
| --- | --- | --- |
| UI | Streamlit (Python rendert HTML) | statisches Frontend in `web/` (HTML + ES-Module, kein Build-Schritt) |
| Server | Streamlit-Laufzeit | FastAPI + uvicorn (`server/`), JSON unter `/api/...` |
| 3D | Plotly-Figuren pro Frame | three.js/WebGL im Browser (`web/js/scene.js`), instanzierte Deckung |
| Diagramme | Plotly (≈3 MB JS) | Canvas 2D (`web/js/charts.js`) |
| Zustand | `st.session_state` | `server/state.py` (ein thread-sicherer Serverzustand) |
| Start | `streamlit run ...` | `python3 install.py` → `python3 start.py` (öffnet den Browser) |
| Abhängigkeiten | streamlit, plotly, pandas, torch, sb3 | numpy, gymnasium, fastapi, uvicorn, psutil, httpx (+ optional torch/sb3) |

Die Simulation, Physik, Karten, Waffen, Rewards, Curriculum, Imitation Learning
und das Training sind unverändert – nur die Darstellungsschicht wurde ersetzt.
Die Streamlit- und Plotly-Kopplung im Code ist restlos verschwunden (`grep` auf
`streamlit`/`plotly` findet nur noch Aufräum-/Erkennungscode).

**Neue/umbenannte Dateien**

* `server/`: `app.py` (Routen), `state.py` (Zustand + Steuerung), `scene.py`
  (Szenen-/Frame-Payloads), `analytics.py` (Stats/Heatmap/Benchmark),
  `minigames.py` (Aim/Dodge), `actions.py` (Tasten → Aktionsvektor),
  `policies.py` (Modell-Cache), `config.py` (Pfade, Presets).
* `web/`: `index.html`, `css/style.css`, `js/{app,api,store,dom,charts,scene}.js`,
  `js/panels/*.js` (acht Panels), `vendor/three.module.min.js` (three r160, MIT).
* `training/weapon_lab.py`: TTK-Simulation + Lua-Export (aus `gui/tabs/ttk.py`).
* `tests/test_api.py`, `tests/test_scene.py`, `tests/test_analytics.py`,
  `tests/test_policies.py`: neue Suiten für die neue Schicht.
* `install.py`, `start.py`: Einrichtung und Start inkl. Browser.
* `tests/dom/`: optionale jsdom-Tests für Frontend und 3D-Geometrie.

**Frontend-Verträge (wichtig für Änderungen)**

* Importmap `three` → `./vendor/three.module.min.js`; Fallback-Kette
  importmap → relativer Pfad → CDN.
* Koordinaten: Simulation `(x, y_ground, z_up)` → Three.js `(x, z, y)`.
* Panels sind Objekte `{ id, label, mount(root), onShow, onHide, unmount }` und
  werden in `web/js/panels/index.js` registriert.
* Szene pro Frame: `{ agents[], trails[], frame, elapsed }` (max. 40 Trails).

---

## 2. Gefundene Bugs

Alles hier Gelistete wurde **behoben** und – wo sinnvoll – durch einen Test
abgesichert.

| # | Bug | Auswirkung | Fix |
| --- | --- | --- | --- |
| 1 | `np.histogram2d` lieferte `float32`-Werte im Heatmap-Grid | `/api/heatmap` → HTTP 500, Heatmap-Panel unbrauchbar | Grid/Peak in Python-Floats konvertiert + Test, der die JSON-Serialisierbarkeit prüft |
| 2 | **TTK-Selbstbetrug**: `ttk` war immer die Episodendauer, auch bei Zeitlimit-Entscheidungen, Niederlagen und Unentschieden | „Ø TTK", TTK-Histogramm und Trainingsmetriken zeigten Episodenlängen, nicht Time-to-Kill | `killed`-Flag in `env/shooter_env.py`; `avg_ttk` + Histogramm nur noch über bestätigte Kills (`server/analytics.py`, `training/train.py`), UI-Label „Ø TTK (KILLS)" mit Kill-Zähler; Tests |
| 3 | `read_csv_rows()` prüfte `key is None` statt `value is None` | eine halb geschriebene letzte CSV-Zeile (Training läuft) leakte als `None`-Zeile in alle Diagramme | Prüfung auf Spaltenanzahl umgestellt + Test |
| 4 | Vollständiges Einlesen von `logs/heatmap_events.csv` bei **jedem** Heatmap-Request | nach langen Trainingsläufen mehrere hundert MB I/O pro Klick | Tail-Read-Fenster (4 MB) + Limit 50 000 Events + Test |
| 5 | `start.py` erkannte die venv über `Path(sys.executable).resolve()` | `.venv/bin/python` ist eine Symlink-Kette auf den System-Python → `start.py` wechselte nie in die venv, frische Installationen scheiterten mit „Fehlende Pakete: …" | Erkennung über `sys.prefix`, Reexec nur wenn Pakete fehlen, `flush()` vor `execve` + Test im Sandbox-Durchlauf |
| 6 | `Path.relative_to(PROJECT_ROOT)` auf Pfaden außerhalb des Repos | `/api/meta` und Demo-Speichern → HTTP 400 (z. B. in Tests mit temporären Ordnern) | `relative_path()` mit Fallback auf den absoluten Pfad |
| 7 | Arena-Stepschleife brach ab, wenn `running == False` (`session.running is False and requested > 1`) | mehrstufige `/api/arena/step`-Aufrufe außerhalb des Auto-Runs bewirkten nichts | Guard entfernt, es werden immer die angeforderten Schritte gerechnet |
| 8 | Agenten-Label griff ungeprüft auf `canvas.getContext('2d')` zu | `TypeError … clearRect of null` beendete die Frame-Aktualisierung, wenn kein 2D-Kontext verfügbar ist | Guard: Sprite bleibt unsichtbar, Render-Loop läuft weiter |
| 9 | `pip install -r requirements.txt` zog unter Linux das ~2,5 GB große CUDA-PyTorch | unerwartet riesiger Download, GPU-Pakete auf einem CPU-Projekt | Aufteilung in `requirements.txt` (Laufzeit) und `requirements-training.txt` (CPU-Wheel-Anleitung); `install.py` installiert weiterhin automatisch das CPU-Wheel |
| 10 | Modell-Cache (`PolicyCache`) wuchs unbegrenzt, jeder Checkpoint blieb im RAM | Speicherverbrauch steigt über eine Sitzung mit vielen Modellen | LRU mit `MAX_CACHED_POLICIES = 3` + Tests |
| 11 | Agenten-Labels und Diagramme crashten in Umgebungen ohne Canvas-2D | jsdom-/headless-Tests brachen ab | Fallbacks, dokumentiert in `tests/dom/README.md` |

### Auffälligkeiten ohne Codeänderung

* `duel_result()["ttk"]` bleibt die Matchuhr – das ist jetzt korrekt dokumentiert
  und wird nur über das `killed`-Flag als Kill gewertet.
* Der `heuristic_action()`-Gegner ist die einzige eingebaute KI; trainierte
  Policies müssen als Checkpoint vorliegen (`models/*.zip` + `_vecnormalize.pkl`).
* `logs/` ist leer nach einem frischen Klon: STATS und HEATMAP zeigen bewusst
  leere Diagramme/Nullwerte, bis Training oder Runden Daten geschrieben haben.

---

## 3. Nice-to-have / offene To-dos (priorisiert)

### Wichtig

1. **Heatmap-Datenquelle für Handrunden**: Arena- und Playground-Runden landen
   nur im Session-Speicher, `heatmap_events.csv` wird ausschließlich vom
   Training geschrieben. Sinnvoll: Arena-Runden optional in eine eigene CSV
   schreiben und im Heatmap-Panel eine Datenquelle auswählen.
2. **Aggregation statt Rohzeilen**: `heatmap_events.csv` wächst pro Episode.
   Mittelfristig Tages-/Lauf-Rotation oder voraggregierte Zellen speichern
   (dann sind auch 500 000+ Episoden instant).
3. **Absicherung für `--host 0.0.0.0`**: aktuell kann jedes Gerät im LAN
   Training starten, Karten ändern und Dateien speichern. Lösung: optionaler
   Token (`--token`, Header/Query) oder Bind-Warnung im UI.
4. **CI**: `python -m unittest discover -s tests` + die jsdom-Tests als
   GitHub-Action (`.github/workflows/tests.yml`). Aktuell läuft nur lokal.
5. **Sprachkonsistenz**: UI-Texte mischen Deutsch („Zielscheiben", „Deckung")
   und Englisch („Start Training", „POSITION"). Eine Sprache festlegen.

### Verbesserungen

6. **Live-Frames per WebSocket/SSE** statt HTTP-Polling: flüssigere Darstellung
   bei gleicher Serverlast, Trails ohne 40er-Kappung.
7. **Trail-Länge/Effekte einstellbar** (Regler „Trail-Länge", Treffer-Marker).
8. **Adaptive Renderqualität**: `Performance/Balanced/Ultra` gibt es für die
   Geometrie; zusätzlich Pixelratio/Detail automatisch senken, wenn die FPS
   unter einen Schwellwert fallen.
9. **Ergebnis-Historie für Aim/Dodge** (Highscores in `logs/minigames.csv`).
10. **Checkpoint-Browser**: Modellliste mit Trainingsschritten/Belohnung,
    direkt im Arena-Panel wählbar (heute nur Dateiname-Dropdown).
11. **Trainingsprofile speichern/laden** (JSON) statt Formular neu ausfüllen.
12. **Curriculum-Fortschritt sichtbar machen** (Phase, Schwellenwerte, Verlauf).
13. **Windows-Begleiter**: `start.py --windows` könnte die WSL-IP ermitteln und
    die passende `http://<ip>:8501`-URL samt Firewall-Hinweis ausgeben.
14. **Screenshots/GIF im README** und ein Architekturbild (Datenfluss
    Browser ↔ API ↔ Env/Threads).
15. **`server/state.py` (≈900 Zeilen) aufteilen** in `state_arena.py`,
    `state_playground.py`, `state_training.py`, `state_maps.py` – reine
    Wartbarkeit, keine Funktionsänderung.
16. **Strukturierte Fehlermeldungen**: `ValueError` → 400 ist grob; ein
    `{"error": {"code", "message"}}`-Schema wäre für externe Clients klarer.
17. **Performance-Budget dokumentieren**: gemessene FPS-Werte pro Panel/Karte in
    der README-Tabelle (hilft, Regressionen zu erkennen).

### Bekannte Grenzen der Verifikation

18. **Kein echtes Browser-/GPU-Testing möglich** (Sandbox): WebGL, Maus-Drag und
    WSL-Browserstart wurden nicht live ausgeführt. Ersatz: `tests/dom/smoke.mjs`
    (30 Prüfungen, jsdom) und `tests/dom/scene.mjs` (Geometrie mit echtem
    three.js und Renderer-Stub). Vor dem ersten echten Start also einmal alles
    im Browser durchklicken.
19. **PPO-Training wurde nicht end-to-end getestet** (kein torch/SB3 in der
    Sandbox installierbar). `tests/test_training_smoke.py` läuft mit SB3
    automatisch mit – nach `install.py` einmal `python3 -m unittest discover -s
    tests` ausführen.

---

## 4. Gelöscht / aussortiert

| Was | Warum |
| --- | --- |
| `gui/` (komplett, inkl. `app.py`, `tabs/*`, `visuals.py`, `style.css`) | Streamlit-Oberfläche, vollständig ersetzt |
| `tests/test_playground_ui.py`, `tests/test_visuals.py` | testeten Streamlit-/Plotly-Interna |
| `.streamlit/secrets.toml`-Eintrag in `.gitignore` | gehörte zu Streamlit |
| `pandas`-Abhängigkeit | nur für Streamlit-Tabellen gebraucht, ersetzt durch Canvas |
| `torch`/`stable-baselines3` in `requirements.txt` | verschoben nach `requirements-training.txt` (optional, CPU-Wheel) |
| Plotly-Datenpfade (`fig.to_json()`, `st.plotly_chart`) | durch JSON + Canvas ersetzt |

**Ebenfalls unnötig, falls du noch mehr aufräumen willst**

* `exports/`-Eintrag in `.gitignore` (alter ZIP-Bundle-Workflow) – kann raus,
  schadet aber nicht.
* `data/demos.csv` (~40 Demo-Zeilen) ist der Startdatensatz für Behavior
  Cloning; löschen nur, wenn du kein Imitation Learning brauchst.
* `web/vendor/three.module.min.js` (670 KB, three r160) ist bewusst im Repo,
  damit alles offline läuft; `install.py --force-assets` lädt es neu, ohne
  vendor-Datei nutzt das Frontend das CDN.

**Noch installierte Altlasten** (nicht im Repo, nur in deiner Umgebung):
`streamlit`, `plotly`, `pandas`. Entfernen mit:

```bash
python3 install.py --remove-legacy     # oder interaktiv: python3 install.py
```

---

## 5. Quick Reference

```bash
python3 install.py                     # venv, Pakete, three.js, Selbsttest
python3 start.py                       # Server + Browser
python3 install.py --skip-torch        # ohne PPO-Training
python3 install.py --remove-legacy     # Streamlit/Plotly/pandas entfernen
python3 start.py --host 0.0.0.0        # aus Windows/LAN erreichbar
python3 -m unittest discover -s tests -v          # 62 Python-Tests
cd tests/dom && npm install && node smoke.mjs     # Frontend-Smoke-Test
```

API-Dokumentation: <http://127.0.0.1:8501/api/docs>
