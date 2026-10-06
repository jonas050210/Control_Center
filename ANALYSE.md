# NEURAL ARENA · Deep-Analyse

Stand: Umbau abgeschlossen (Streamlit vollständig entfernt). Dieses Dokument ist
die geforderte Analyse: was geändert wurde, welche Bugs gefunden und behoben
wurden, was noch offen ist, was verbessert werden kann und was gelöscht gehört.

---

## 0. Runde 2: Die AI sieht nur noch, was sie sehen kann

**Auftrag:** „…die ai weiß nicht direkt wo sie hinschießen muss, also reinforcement learning."

**Vorher (der Leak):** Die Beobachtung lieferte die exakte Gegnerposition, die exakte
Entfernung, den exakten Zielwinkel, die Gegner-Blickrichtung und die exakte Gegner-HP –
auch dann, wenn eine Säule dazwischen war. Gemessen: Gegner hinter einer Säule
(`has_line_of_sight == False`), trotzdem `Obs[5] = 0.60` → x = 12.0 exakt,
`Obs[10] = −0.823` → 5.00 m exakt, `Obs[11] = 0.000` → Zielfehler 0°.

**Jetzt (Observation-Version 2, 31 Werte wie vorher):**

* Richtung nur als **Sektor** von 12 (30°-Raster, Fehler ≤ 15°) als Sinus/Cosinus.
* Entfernung nur als **Band** (<5, 5–10, 10–20, 20–35, 35–60, >60 m).
* Gegner-HP nur als **Band** (4 Stufen), unbekannt = neutral.
* **Sichtkontakt-Bit**: 120°-Sichtkegel **und** freie Sichtlinie auf Augenhöhe.
* **Gedächtnis**: Richtung/Distanz/HP des letzten Sichtkontakts plus „zuletzt
  gesehen vor X s"; nach 5 s verblasst es vollständig.
* **Leak-Test**: zwei Gegnerpositionen im selben Sektor und Band liefern hinter
  Deckung exakt identische Beobachtungen (Test in `tests/test_perception.py`).
* **Reward fair**: Der Zielbonus greift nur bei Sichtkontakt und ist klein
  (0,005/Step ≈ 2,2 pro 30-s-Episode) – siehe „Pazifisten-Bug" unten.

**Vier Modi** (`vision_mode`): `coarse_los` (Standard, echte Sicht), `coarse`
(Sektor/Bänder ohne Deckung), `noisy` (exakt + Rauschen), `exact` (alt, nur zum
Vergleich). Wählbar im ARENA- und TRAINING-Panel; `/api/meta` liefert Feldnamen,
Version und Modi, `/api/arena/perception` den Wahrnehmungszustand.

**Fünf Gegner-Verhalten** (`opponent_mode`, validiert): `stationary` (steht),
`mover` (läuft und zielt, **schießt nicht**), `walker` (schießt zurück, weite
Toleranz), `shooter` (zielt genau), `full` (taktisch, springt, Sprint). Im
ARENA-Panel sind „Passive" und „Mover" zusätzlich wählbar; ein unbekannter Wert
wird mit `ValueError`/HTTP 400 abgelehnt statt still als `stationary` zu laufen.

**Zusätzliche Fixes und Verbesserungen dieser Runde:**

1. **Bot-Navigation**: Der heuristische Gegner blieb an Kisten/Säulen stehen
   (auf Arena z. B. an der Säule bei x = ±12) – mit Deckung dazwischen fror das
   Duell ein, es gab **keinen Sichtkontakt und kein Trainingssignal**. Jetzt
   erkennt der Bot „will vorwärts, bewegt sich nicht" und weicht 45 Frames lang
   zur offeneren Seite aus (Seiten-Raycasts). Ergebnis: Duelle enden in ~50–80
   Frames statt 7200.
2. **Checkpoint-Stempel**: `*_meta.json` neben jedem Checkpoint enthält
   `observation_version` und `vision_mode`; der Modell-Cache lehnt alte Layouts
   mit klarer Meldung ab, statt still Unsinn zu rechnen.
3. **Episodenlänge konfigurierbar** (TRAINING-Panel, Standard 60 s statt 120 s):
   kürzere Episoden = mehr abgeschlossene Kämpfe pro Minute = deutlich mehr
   Lernsignal. Vorher: 22 Episoden in 5 Minuten.
4. **`tools/evaluate_policy.py`**: bewertet einen Checkpoint oder die
   Zufallspolitik (Siege, Time-to-Kill nur bei bestätigten Kills, Blindanteil).
5. **UI**: „👁 WAS DIE AI SIEHT" im ARENA-Panel (Sektor, Band, Sichtkontakt,
   Gedächtnis) – man sieht live, dass die AI hinter Deckung nichts weiß.

### Ehrliche Messwerte (alle auf dieser Maschine, CPU, 2 Worker)

| Messung | Ergebnis |
| --- | --- |
| Zufallspolitik (`--random --episodes 20`) | 0 Siege, 20 Niederlagen, Ø 7,7 s, 31 % blind |
| Trainingstempo | 400–550 Steps/s, 100 000 Steps ≈ 4 min |
| Trainierter Checkpoint (`models/best_model.zip`, 28k Steps, `coarse_los`) unter Phase-1-Bedingungen (9 m Abstand, `stationary`) | **10/10 Siege, alle mit bestätigtem Kill, TTK 20,4 s**, 40 % blinde Frames |
| derselbe Checkpoint, Phase-2-Abstand (16 m) | 20/20 Siege, aber nur 3 Kills (TTK 40 s) – der Rest HP-Entscheidungen |
| derselbe Checkpoint auf Kartendistanz (36 m) | 0 Kills (20 Unentschieden) |
| derselbe Checkpoint gegen `walker`/`shooter`/`full` (die schießen zurück) | 0 Siege, 20 Niederlagen |
| Trainings-Kill-Rate im 100k-Lauf (geschlossener Abstand) | 0 % → 15 %, `best_model` mit 13,8 % |

*Aufnahme dieser Runde. `models/` ist nicht versioniert: die Checkpoints in dieser
Tabelle existieren nicht mehr, die Läufe dahinter sind seither mit der neuen
Gegner-Leiter und `norm_obs=False` wiederholt worden (siehe „Runde 3").*

**Was daraus folgt (und was nicht):** Der Trainer lernt nachweisbar, einen
passiven Gegner in Reichweite zu finden, anzuvisieren und zu töten – mit der
ehrlichen Sichtwahrnehmung und ohne je dessen Koordinaten zu sehen. Gegen einen
Gegner, der selbst schießt, reicht ein Sandbox-Lauf von 4–10 Minuten nicht: dafür
braucht PPO hier Millionen Steps. Diese Grenze wird nicht schöngeredet.

### Der entscheidende Struktur-Fund: die Kartendistanz ist außerhalb der Reichweite

Gemessen mit **perfekter Zielausrichtung** (Bot im `shooter`-Modus, Gegner passiv)
auf Dust:

| Waffe | Abstand | Schüsse | Treffer | Trefferquote |
| --- | --- | --- | --- | --- |
| Pistol | 36 m (Karten-Spawns) | 139 | 0 | **0 %** |
| AK-47 | 36 m | 275 | 5 | **1,8 %** |
| Pistol | 7,8 m (nach Annäherung) | 17 | 4 | 23,5 % |
| AK-47 | 7,8 m | 34 | 4 | 11,8 % |

Auf Kartendistanz kann **niemand** treffen – auch nicht der Skript-Bot, auch nicht
mit perfektem Ziel. Deshalb:

* brach das Curriculum in Phase 1/2 zusammen (der Agent musste erst laufen lernen),
* sah man in Trainingsläufen „Siege" ohne einen einzigen Kill,
* stieg die Kill-Rate erst, nachdem die frühen Curriculum-Phasen näher starten.

**Behoben:** Die Curriculum-Phasen starten jetzt mit 25 %, 45 % bzw. 70 % der
Kartendistanz (Phase 4 = volle Distanz, ARENA/Playground unverändert), mit
Kollisionsprüfung (`position_is_free`) und Tests. Kartendistanzen auf Dust:
Phase 1 = 9 m, Phase 2 = 16,2 m, Phase 3 = 25,2 m, Phase 4 = 36 m.

`tools/evaluate_policy.py --phase N` stellt dieselben Bedingungen zum Nachmessen
her (Standard 4 = echte Spawns).

### Gefundene und behobene Fehler dieser Runde

1. **Wahrnehmungs-Leak** (exakte Gegnerposition/HP durch Wände) – behoben,
   Leak-Tests in `tests/test_perception.py`.
2. **Pazifisten-Bug in der Belohnung**: Der Zielbonus war 0,05/Step ≈ 22 pro
   Episode, ein Kill bringt +5. Der Agent farmte also die ganze Runde den
   Crosshair-Bonus statt zu schießen; Ergebnis waren 0 Kills bei „35 % Siegen".
   Jetzt 0,005/Step (≈2,2/Episode), mit Test-Regel „Episode-Zielbonus < Kill-Bonus".
3. **Der Trainings-Gegner `walker` schoss nie** (`may_fire` nur `shooter`/`full`):
   gegen einen harmlosen Gegner gab es keinen Grund zu kämpfen. Jetzt schießt der
   Walker (jede 3. Gelegenheit, 14°-Toleranz) – Regressionstest.
4. **Curriculum lief nach Zeit statt nach Können**: Der Lauf sprang von Phase 2
   (68 % „Siege") auf 0 %, weil die Phase rein nach Timesteps wechselte. Jetzt
   schaltet die nächste Phase erst frei, wenn die **Kill**-Rate der letzten 30
   Episoden über der Schwelle liegt (Standard 40 %, im Panel einstellbar, 0 = aus).
5. **Absturz im Gate** (`UnboundLocalError: now`) – der Lauf starb nach 50k
   Steps; Regressionstest `test_curriculum_gate_holds_a_phase_without_crashing`.
6. **„Bestes Modell" wurde nach Siegen gewählt** (also nach HP-Vergleichen am
   Zeitlimit, was Verstecken belohnt) und nur an 50k-Grenzen geprüft – traf die
   Grenze eine schlechte Phase, war der einzige gespeicherte Checkpoint
   wertlos. Jetzt laufende Bewertung nach Kill-Rate (Verbesserung ≥ 5 Prozentpunkte,
   mindestens 15 neue Episoden), Altbestände ohne Kill-Stempel werden nicht
   überschrieben.
7. **`heatmap_events.csv` verlor das `killed`-Flag**: Die Analytics konnte
   Zeitlimit-Entscheidungen nicht von echten Kills trennen. Spalte ergänzt.
8. **Tippfehler im Sichtmodus fiel still auf den Standard zurück** (ARENA,
   PLAYGROUND, TRAINING) – jetzt HTTP 400 mit Klartext.
9. **`data/demos.csv` war noch im alten Beobachtungs-Layout**: neu aufgezeichnet
   (1 600 Zustände, `coarse_los`) und mit `data/demos_meta.json` gestempelt;
   veraltete Datensätze werden beim Laden abgelehnt statt Behavioral Cloning mit
   falschen Spalten zu füttern.

10. **Die automatische Bewertung starb an einem abgeschnittenen Checkpoint**:
   `VecNormalize.save()` entfernt beim Pickeln das umgebende Vektor-Env. Wird die
   Datei danach mit `pickle.load` gelesen, landet jeder Zugriff auf ein fehlendes
   Attribut in der Endlosrekursion von SB3s `VecEnv.__getattr__`
   (`RecursionError`) – der Lauf endete mit „Automatic evaluation failed", das
   Panel blieb leer. Jetzt liest ein eigener Loader (`load_normalizer`) nur die
   reinen Statistikzahlen und verträgt auch gekappte Objekte; dieselbe Funktion
   versorgt die ARENA-Inferenz. Regressionstests in `tests/test_policies.py`.
11. **`norm_obs=False`**: Die Wahrnehmung ist bereits auf [-1, 1] begrenzt und
   Nullen bedeuten „nie gesehen" – eine laufende Mittelwert-/Varianz-Normalisierung
   verschiebt genau diese Aussage. Neue Läufe trainieren deshalb auf den rohen
   Zahlen (`VecNormalize(norm_obs=False, norm_reward=True)`); alte Checkpoints mit
   `norm_obs=True` werden weiter normalisiert (`normalize()` prüft das Flag).
12. **Start-Methode konnte das Training killen**: `forkserver`/`spawn` importieren
   das aufrufende Skript in jedem Kindprozess neu. Ohne
   `if __name__ == "__main__":`-Schutz bricht das Training mit einer irreführenden
   `RuntimeError` ab – genau das passierte beim Messlauf. `make_vector_env` fällt
   jetzt automatisch auf `fork` zurück, statt zu sterben.
13. **Kurze Läufe speicherten keinen Checkpoint**: Die Bewertung fand nichts zum
   Nachprüfen. Am Ende jedes Laufs wird jetzt `final_model.*` geschrieben, und die
   Bewertung nimmt `best_model` → `final_model` → Pfad aus dem Job.

### Runde 3: warum nichts gelernt wurde – und was es behoben hat

**Befund 1: `VecNormalize(norm_obs=True)` hat das Lernsignal zerstört.** Die
Wahrnehmung ist bereits auf [-1, 1] begrenzt, viele Werte sind **exakt 0** und
bedeuten „nicht gesehen" (Sichtkontakt 0, Gedächtnis 0, Gegner-HP-Band 0). Die
laufende Mittelwert-/Varianz-Normalisierung verschiebt genau diese Nullen zu
Zufallszahlen und zerstört damit die einzige Information, die das Netz über
Deckung hat. Gemessen:

| Lauf (2 CPU-Kerne, Dust, `coarse_los`) | Kill-Rate in Phase 1 |
| --- | --- |
| frühere Läufe mit `norm_obs=True` (100k Steps) | Peak 15 % |
| Lauf mit `norm_obs=False` (62k Steps, Sicht-Curriculum) | **100 %** (30/30 Episoden mit bestätigtem Kill) |

Das ist kein sauberer A/B-Test (dazwischen kamen Kürzere Episoden und das
Sicht-Curriculum), aber die Größenordnung – 15 % gegen 100 % – ist eindeutig,
und die Begründung ist strukturell: Nullen, die „nie gesehen" heißen, darf man
nicht weg-normieren.

**Befund 2: die Phase-2-Klippe.** Der Sprung von `stationary` (steht still,
schießt nie) auf `walker` (läuft **und** schießt zurück) bei gleichzeitig
größerem Abstand (16,2 m) und neuer Waffe (SMG) war zu groß: der Lauf lieferte
**~500 Episoden hintereinander ohne einen einzigen Sieg oder Kill** und blieb
dort bis zum Budget-Ende. Der Fehler steckte zusätzlich im Code: zwei
`_curriculum_spawns`-Definitionen, die ältere überschrieb die neuere (dasselbe
Muster wie beim Vision-Modus). Jetzt:

* eine einzige Distanzrampe **9,0 / 12,6 / 21,6 / 36,0 m** (Phase 4 = Kartenspawns),
* eine Gegner-Leiter, die pro Phase **genau eine** neue Lektion bringt:
  `stationary` → **`mover`** (läuft und zielt, schießt nicht) → `walker`
  (erste Gegenwehr) → `full`,
* unbekannte Gegnermodi werden abgelehnt (`ValueError`/HTTP 400) statt still als
  `stationary` zu laufen; im Test steht, dass `mover` niemals schießt.

### Runde 3: gemessen (250k-Steps-Budget, 600 s, 2 CPU-Kerne, Dust)

| Bedingung | Ergebnis |
| --- | --- |
| Trainingslauf Phase 1 (`stationary`, Pistol, 9 m, `norm_obs=False`) | Kill-Rate im 100-Episoden-Fenster bis **100 %**; bester Checkpoint (38 664 Steps) 79,6 % Kills / 89,2 % Siege |
| Phase 2 (`mover`, SMG, 12,6 m) | Siege 70–80 %, davon nur 2–6 % mit bestätigtem Kill → das Freischalt-Gate (25 % Kills) blieb zu |
| Phase 3/4 | nie erreicht; ab ~170k Steps brach die Politik ein (0 % Siege, sehr kurze Episoden) – Gegenfeuer ist die nächste Lernhürde |
| bester Checkpoint, Phase 1, `noisy` (**wie trainiert**), gegen `stationary` | **20/20 Siege, 20/20 bestätigte Kills, TTK 14,3 s**, Trefferquote 67,6 %, 1 % blinde Frames |
| derselbe Checkpoint gegen `mover` (läuft, schießt nicht), `noisy` | 15/20 Siege (davon 2 Kills), 5 Unentschieden, **0 Niederlagen** |
| derselbe Checkpoint unter der **Ziel-Wahrnehmung `coarse_los`** | 17/20 Siege, aber **0 Kills** (2,2 % Trefferquote, 52–59 % blind) |

**Was das heißt:** Der Trainer lernt mit der ehrlichen Beobachtung und der
kill-basierten Belohnung wirklich zu kämpfen – 20/20 Siege *mit bestätigtem Kill*
gegen den passiven Gegner, im Schnitt nach 14 s. Die zweite Zeile ist die
ehrliche Einschränkung: das Ziel-Sichtmodell `coarse_los` (Sektor, Deckung,
Gedächtnis) überträgt sich **nicht von selbst** auf ein Netz, das mit `noisy`
trainiert wurde – dafür muss mit `coarse_los` weitertrainiert werden. Genau das
ist der Zweck der Phasen 3/4 im Sicht-Curriculum, die dieser Lauf wegen des
Kill-Gates nicht mehr erreicht hat. Deshalb bewertet die automatische Auswertung
jetzt **beide** Modelle: den besten Checkpoint unter seiner eigenen Wahrnehmung
und das Endmodell unter der Ziel-Wahrnehmung (`evaluation.target` im Panel).

### Runde 4: der Trainer lernt mit der ehrlichen Sicht (der eigentliche Beleg)

Der entscheidende Lauf dieser Runde trainiert **von der ersten Episode an mit
`coarse_los`** (Sektor, Band, Deckung, Gedächtnis – keine Koordinaten, kein
`noisy`-Vortraining). Ergebnis nach 32 000 Steps: Kill-Rate **60 %** (Fenster der
letzten 50 Episoden), 76 % Siege – und die unabhängige Nachmessung mit
`tools/evaluate_policy.py` auf dem gespeicherten Checkpoint:

| Bedingung (Phase 1, 9 m, Pistol vs Pistol, `coarse_los`) | Ergebnis |
| --- | --- |
| Gegner `stationary` | **20/20 Siege, davon 12 mit bestätigtem Kill**, TTK 20,6 s, Trefferquote 19,5 %, 24 % blinde Frames |
| Gegner `mover` (läuft, schießt nicht) | **20/20 Siege**, 0 Niederlagen |

Das ist der Beweis, der in Runde 3 noch fehlte: **mit der ehrlichen Wahrnehmung
entstehen echte Kills.** Vorher hatte ich nur ein mit `noisy` trainiertes Netz
unter `coarse_los` gemessen (0 Kills) – der Unterschied war das Training, nicht
die Wahrnehmung.

**Die verbleibende Bruchstelle ist die Gegenwehr.** Beide Läufe dieser Runde
brechen ein, sobald das Curriculum auf die nächste Stufe schaltet, in der der Bot
zurückschießt: ab ~130k Steps fällt die Siegrate auf 0 %, die Episoden werden
~4,5 s kurz, und der Lauf erholt sich im restlichen Budget nicht mehr
(≈2 000 Episoden ohne Sieg). Zwei Konsequenzen sind umgesetzt:

* **Curriculum mit Rücknahme**: Steigt die Phase auf, wird nach 40 Episoden
  geprüft; liegt die Kill-Rate unter der Hälfte der Schwelle, geht der Lauf eine
  Phase zurück und verlangt für den nächsten Versuch 10 Prozentpunkte mehr
  (`curriculum_backtrack_needed`, eigene Tests). Eine Phase, die nur Niederlagen
  produziert, wird nicht mehr bis zum Budgetende durchgehalten.
* **Status erst nach der Prüfung**: `complete` wird erst gesetzt, wenn die
  automatische Bewertung durch ist. Vorher beendeten Clients (und mein
  Mess-Harness) den Prozess genau dann, wenn „fertig" dastand – die Bewertung
  wurde abgeschnitten und der Bericht fehlte (der `SIGABRT` am Ende war die Folge,
  nicht die Ursache).

### Runde 5: der Verifikationslauf (Curriculum mit Rücknahme) – es läuft

Lauf: 260 000 Steps Budget, 800 s, 2 CPU-Kerne, Dust, **`coarse_los` von der
ersten Episode an**, `norm_obs=False`, Curriculum mit Rücknahme.

| Phase des Laufs | Ergebnis |
| --- | --- |
| Phase 1 (`stationary`, Pistol, 9 m) | Kill-Rate steigt 0 % → **96 %** (Fenster der letzten 50 Episoden bei 43k Steps), Siege 87 % |
| Phase 2 (`mover`, SMG, 12,6 m), ab 92k Steps | Siege 96 % → 60 % → pendelt sich bei **26–34 % Siegen/Kills** ein |
| Phase 3 (Gegenwehr) | Der Lauf **bricht nicht mehr ein**: statt 0 % (vorher) bleiben ~30 % Siege bis zum Budgetende |
| Automatische Bewertung des besten Checkpoints (Phase 1, `coarse_los`, 8 Episoden/Gegner) | gegen `stationary`: **8/8 Siege, 8/8 mit bestätigtem Kill, TTK 6,1 s**, Trefferquote 52 % · gegen `walker`/`full`: 0/8 (die schießen zurück) |

Vorher (ohne Rücknahme) endeten beide Läufe nach dem Phasenwechsel bei 0 % Siegen
und ~4,5 s kurzen Episoden. Jetzt bleibt die Politik handlungsfähig, und die
automatische Bewertung läuft bis zum Ende durch (der Status springt erst nach der
Prüfung auf „fertig").

**Was noch fehlt (ehrlich):** gegen Gegner, die selbst schießen, verliert der
Phase-1-Checkpoint jedes Duell auf 9 m. Das ist die Aufgabe der Phasen 3/4 – sie
brauchen aber deutlich mehr Steps, als ein Sandbox-Lauf von 13 Minuten liefert.
Der Weg dahin ist jetzt messbar: Phase 1 gemeistert (8/8 Kills), Phase 2 auf
Augenhöhe (Rücknahme verhindert den Absturz), Gegenwehr als nächste Hürde.

**Zusätzlich abgesichert:** der Bewertungsbericht wird nach **jedem** Gegner auf
die Platte geschrieben und in den Job-Snapshot geschoben, `final_model.zip` wird
immer gespeichert (vorher nur, wenn kein „Bestes Modell" existierte). Eine
abgebrochene Prüfung hinterlässt damit Teilergebnisse statt gar nichts.

**Kosten/Hinweis:** die ehrliche Sicht ist *kein* Trainingshindernis. Der Vergleich
in Runde 4 kippt die frühere Annahme: mit `coarse_los` von Anfang an 20/20 Siege
und 12 Kills gegen den passiven Gegner, mit dem `noisy`-Vortraining unter
derselben Sicht 17/20 Siege und **0** Kills. Deshalb ist der Verifikationslauf
(Runde 5) ohne Sicht-Curriculum gelaufen (`vision_curriculum=False`); das Flag ist
im Panel/Config weiter vorhanden, falls jemand erst mit vergebender Wahrnehmung
vortrainieren will. Unabhängig davon: mehrere Seeds vergleichen
(`python3 tools/seed_sweep.py 2 100000`).

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
