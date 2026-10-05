# RocketAI – Plan

Ziel: Eine eigene KI lernt Rocket League. Sie trainiert in einer Simulation
(ohne das Spiel) und spielt danach **im echten Rocket League**: offline über
RLBot v5 gegen Psyonix-Bots, Community-Bots, sich selbst oder dich.

> **Regel:** nur offline bzw. lokale Matches, die RLBot startet. Seit April 2026
> hat Rocket League Easy Anti-Cheat; online, in Casual, Ranked oder privaten
> Online-Matches sind Bots verboten (Bann-Risiko). RocketAI startet
> ausschließlich Offline-Matches.

## Bausteine

| Baustein | Datei(en) | Technik | Aufgabe |
| --- | --- | --- | --- |
| Konfiguration | `rocketai/config.py` | Dataclass + Vorlagen | alle Trainingsparameter, Ordnerstruktur eines Runs |
| Simulation | `rocketai/env.py` | RocketSim 2.2 + RLGym 2.0 | Physik ~2 100 Schritte/s pro Prozess (gemessen); Anstoß- und Zufallsstarts |
| Beobachtung | `rocketai/obs.py` | `DefaultObs` + Zusatzblock | 172 Basiswerte + 12 Zusatzwerte (Ball relativ, Abstand, Drehung, Höhe, Boost, Torgefahr …) = 184; Training und echtes Spiel bauen sie identisch |
| Laufzeit | `rocketai/runtime.py` | Dateisperre + Lebenszeichen | ein Trainingsprozess pro Run; erkennt auch fremde Trainer (Kommandozeile vs. Weboberfläche) |
| Messen | `rocketai/benchmark.py` | echte Simulation + PPO | Schritte/s, Entscheidungen/s, Echtzeit-Faktor, Lernschritt (CPU/GPU), Empfehlungen |
| Belohnungen | `rocketai/rewards.py` | gestuft (Stufe 1–3) | erst Ball treffen, dann Richtung Tor, dann komplettes Spiel |
| Netz | `rocketai/model.py` | PyTorch-MLP, Actor + Critic getrennt | 184 Beobachtungen → 90 Aktionen; Checkpoint-Format v1, Gerät frei wählbar (`device`: CPU oder Grafikkarte) |
| Erfahrung sammeln | `rocketai/rollout.py` | Worker-Prozesse (spawn, Windows-tauglich) | Self-Play, GAE direkt im Worker |
| Lernen | `rocketai/ppo.py` | PPO mit Clipping, KL-Notstopp | getrenntes Gradient-Clipping für Actor/Critic |
| Trainingsschleife | `rocketai/trainer.py` | Prozess pro Training | Checkpoints, Messwerte, Bewertung, Replays, Stopp-Datei |
| Gegner | `rocketai/opponents.py` | Skript-Bots | Stillstand < Zufall < Balljäger < Verteidiger |
| Matches | `rocketai/match.py` | Simulation | Bewertung (S/U/N, Tore) und Replays (15 Bilder/s) |
| Spiel-Brücke | `rocketai/rlbot_convert.py` | eigener Konverter | RLBot-`GamePacket` → RLGym-`GameState` (identische Beobachtung) |
| Bot | `rocketai/rlbot_bot/` | RLBot v5 `Bot` | fährt im echten Spiel, 15 Entscheidungen/s |
| Match-Start | `rocketai/play.py` | RLBot `MatchManager` | schreibt `match.toml`, startet RLBotServer + Rocket League |
| Web-App | `rocketai/server.py`, `rocketai/web/` | FastAPI + HTML/CSS/JS ohne Build | Übersicht, Training, Arena, Spielen, Einrichtung |
| Einrichtung | `install.py`, `start.py` | nur Standardbibliothek | venv, CPU-PyTorch, RLBotServer mit Prüfsumme |

## Datenfluss

```
 Simulation (RocketSim)          Training                      Echtes Spiel
 RLGym-Env  --obs 184-->  PPO-Policy (MLP)  --checkpoint-->  RLBot-Bot  <-->  Rocket League
            <--Aktion 90--                                   (gleiche Beobachtung,
                                                              gleiche 90 Aktionen,
                                                              8 Ticks pro Entscheidung)
```

Damit die KI im echten Spiel so spielt wie im Training:

- **Gleiche Beobachtung:** `DefaultObs` mit Platz für bis zu 3v3 (172 Werte)
  plus 12 Zusatzwerte aus `rocketai/obs.py` (184). Der Konverter baut daraus
  denselben `GameState` wie RocketSim; der Bot wählt den Bauplan anhand der im
  Checkpoint gespeicherten Größe. Ein Test vergleicht beide Beobachtungen Wert
  für Wert — für die Basis- und die erweiterte Variante.
- **Gleiche Aktionen:** Lookup-Tabelle mit 90 Aktionen, jede 8 Ticks gehalten.
- **Gleiches Timing:** Das Training simuliert RLBots Verzögerung von einem Tick.
- **Gleiche Boost-Pads:** RLBot nummeriert Pads anders; sie werden über ihre
  Position zugeordnet.

## Ordner eines Trainings (`runs/<name>/`)

| Datei | Inhalt |
| --- | --- |
| `config.json` | alle Einstellungen |
| `status.json` | Zustand (läuft/fertig/…), Schritte, Tempo, Echtzeit-Faktor, Gerät, Herzschlag |
| `metrics.jsonl` | eine Zeile pro PPO-Runde (Ballkontakte, Tore, Belohnung, KL, …) |
| `evaluations.jsonl` | Ergebnisse gegen Balljäger und Verteidiger |
| `checkpoints/` | `<schritte>.pt` und `latest.pt` (mit Optimizer zum Fortsetzen) |
| `replays/` | aufgezeichnete Bewertungsspiele für die Arena |
| `train.log`, `process.log` | Protokoll und rohe Prozessausgabe |
| `control.json` | von der Web-App geschrieben: `{"stop": true}` beendet sauber |
| `trainer.lock`, `trainer.lock.guard` | Prozessnummer und Dateisperre des laufenden Trainings (verschwinden beim Beenden) |

## Phasen

### Phase 1 – Fundament (fertig)

Simulation, PPO-Self-Play, Checkpoints, Bewertung gegen Skript-Bots,
Replays, Web-App, Installer/Starter, RLBot-Bot samt Konverter, Tests, CI.

Erste Messungen in der Sandbox (2 CPU-Kerne):

- Tempo (**reine Simulation**, ohne Lernschritt): ~4 200 Schritte/s mit 2
  Prozessen (ca. 2 100 pro Prozess, ~70× Echtzeit); ein PPO-Lernschritt
  schafft ~20 000 Schritte/s. Ende-zu-Ende gerechnet bleibt davon deutlich
  weniger (siehe Phase 7).
- PPO-Diagnose: Nach dem Trennen des Gradient-Clippings stieg die KL pro
  Update von ~0,00004 auf ~0,002–0,003; die Clip-Rate liegt bei 1–2 %.
- Skript-Bots: Der Balljäger schlägt „Stillstand“ und „Zufall“ deutlich;
  der Verteidiger ist im Schnitt stärker als der Balljäger.

### Phase 2 – Erstes echtes Training (du, Windows-PC)

1. `python3 install.py`, dann `python3 start.py`.
2. Vorlage **Schnelltest**: prüft in ~2 Minuten, ob alles läuft.
3. Vorlage **Autopilot** (startet bei Stufe 1, schaltet selbst auf 2 und 3
   um). Laufen lassen; Stufenwechsel erscheinen als gestrichelte Linie in den
   Kurven und im Protokoll.
4. Manuell geht es weiterhin: **Anfänger 1v1** → **Torjäger 1v1**.
5. Arena: Bewertungsspiele ansehen. Spielen: Checkpoint gegen Psyonix
   *Beginner* und *Rookie* testen.

Woran man Fortschritt erkennt:

| Signal | gut | Warnzeichen |
| --- | --- | --- |
| **eigene** Ballkontakte/min | steigt über Stunden | bleibt flach > 30 Mio. Schritte (die Zeile „Gegner“ daneben zeigt, ob nur die andere Seite am Ball ist) |
| KL pro Update | 0,002–0,02 | dauerhaft < 0,0005 (lernt nicht) oder Notstopps in jeder Runde |
| Entropie | sinkt langsam von 4,5 | fällt schnell unter 2 (KI wird starr) |
| Erklärte Varianz | steigt Richtung 0,5–0,9 | bleibt um 0 |

### Phase 3 – Stärker werden

- ~~**Gegner-Pool**~~ (fertig): 20 % der Spiele gegen die letzten 5
  Checkpoints, Siegquote als Kurve.
- ~~**Automatischer Lehrplan**~~ (fertig): Autopilot wechselt die Stufe.
- **Stufe 3:** Luftkontakte, Boost-Management, höhere Torbelohnung.
- **Belohnungs-Feinschliff:** Ballgeschwindigkeit Richtung Tor statt nur
  Richtung; Strafe für Gegentore aus eigenem Fehler.
- **2v2/3v3:** eigenes Training; die Beobachtung ist schon dafür ausgelegt.
- **Bewertung ausbauen:** Elo-Rangliste aller Checkpoints (Turniermodus in
  der Arena).

### Phase 4 – Feinschliff

- Schwierigkeitsstufen gegen Menschen (z. B. ältere Checkpoints oder
  Reaktionsverzögerung).
- Vergleich mit Community-Bots aus dem RLBot-Botpack (Ziel: Nexto-nahe).
- Optional schnelleres Lernen über `rlgym-learn` (Rust-Worker), falls der
  eigene Lerner zum Engpass wird.

### Phase 5 – Der Lehrer (Nexto)  [erledigt]

- [x] Nextos Netz laden (TorchScript), Dateien per SHA-256 pruefen, lokal
      unter `tools/nexto` ablegen (nicht im Repo, GPL)
- [x] Uebersetzer von unserem Spielzustand in Nextos Beobachtung
      (Test: Abweichung < 1e-4 gegen Nextos eigene Funktion)
- [x] Lehrer als Spieler: Live-Ansicht, Replays, echte Spiele (RLBot)
- [x] Lehrer als Trainingsgegner (`teacher_opponent_prob`)
- [x] Nachahmung als Zusatzsignal (`teacher_weight`, faellt ueber das Training)
- [x] Messungen, ehrliche Einordnung und Wissensdokumentation in docs/WISSEN.md
- [ ] Optional spaeter: Lehrer in ein groesseres Aufmerksamkeitsnetz destillieren,
      damit die Nachahmung wirklich traegt


### Phase 6 – Funktion und Ehrlichkeit (fertig)

- [x] Multi-Worker-Fehler behoben: `teacher_rows` werden beim Zusammenlegen der
      Worker-Stapel verschoben (vorher lernte die KI vom Lehrer für die falschen
      Situationen, sobald mehr als ein Prozess lief)
- [x] Ballkontakte und Tore werden nach Team getrennt ausgewertet („eigene“ vs.
      „Gegner“); Autopilot-Schwellen entsprechend angepasst
- [x] `teacher_weight = 0` schaltet den Lehrer wirklich ab (keine Aufzeichnung,
      kein Laden, keine Restgewichtung)
- [x] Dateisperre pro Run (`trainer.lock`) gegen doppelte Trainingsprozesse,
      Anzeige „extern“, wenn die Kommandozeile trainiert
- [x] Reproduzierbarkeit: Torch-Seed, gespeicherter Zufallszustand, Fortsetzen
      ohne Sprung
- [x] Erweiterte Beobachtung (`obs_extras`) inkl. rotierender Bälle bei den
      Zufallsstarts und identischem Bauplan im echten Spiel
- [x] Lernen auf der Grafikkarte möglich (`device: auto/cpu/cuda`)
- [x] `rocketai benchmark` + Knopf in der Oberfläche; Hinweise zu Einstellungen
- [x] RLBot-Kennungen vereinheitlicht (`--agent-id`, eigene Config für
      „KI gegen sich selbst“) — vorher startete der zweite Bot ohne Auto
- [x] Oberfläche: „Was die KI sieht“, getrennte Kennzahlen, Hinweise, Messwerte

### Phase 7 – Tempo und Werkzeuge (fertig)

Gemessen in der 2-Kern-Sandbox, `rocketai benchmark` (echte Messung):

| Einstellung | Schritte/s | Echtzeit |
|---|---|---|
| 1 Prozess × 1 Spiel | 1 353 | 43× |
| 1 Prozess × 2 Spiele | 1 668 | 52× |
| 1 Prozess × 4 Spiele | 1 903 | 42× |
| Lernschritt (5 000 Schritte × 3 Epochen, 1 Thread) | 14 700–16 400 | – |
| **Ende-zu-Ende (sammeln + lernen)** | **1 371** | – |

- [x] **Ende-zu-Ende-Tempo** gemessen und zur Grundlage aller Zeitangaben
      gemacht: Ein Stapel braucht `S/Sim + epochs·S/Lern` Sekunden, weil
      Sammeln und Lernen nacheinander laufen und während des Lernens alle
      Simulationsprozesse stehen. Vorher waren Prognosen 2–6× zu optimistisch.
- [x] **Lern-Threads**: Standard jetzt `Kerne − 1` (vorher ein Viertel, max. 8)
      — auf 16 Kernen warteten während des Lernens ~11 Kerne.
- [x] **Benchmark** testet auch *Spiele pro Prozess* (1/2/4) und vergleicht nur
      gleiche Einstellungen; Empfehlung setzt den Wert nicht mehr auf 1 zurück.
- [x] **Tempo-Zerlegung** in der Oberfläche (Sammeln vs. Lernen) samt
      Klartext-Hinweis, was gerade bremst.
- [x] **Fortsetzen mit Einstellungen**: Stufe, Lehrer als Gegner, Nachahmung,
      Prozesse, Spiele pro Prozess, Episodenlänge — ohne `config.json` von Hand.
      Der erreichte Stand kommt aus dem Checkpoint, nicht aus `status.json`.
- [x] **Auto-Fortsetzen**: Banner „Unterbrochen“ mit *Jetzt fortsetzen* und
      `start.py --resume-interrupted` (z. B. per Aufgabenplanung nach Reboot).
- [x] **Zwei Fehler behoben:** `teacher_rows` wurden bei mehreren Workern falsch
      verschoben, wenn ein Worker nichts aufzeichnete; `teacher_weight = 0`
      schaltete die Nachahmung nicht ab (Validierungsfehler, dadurch in der
      Oberfläche nicht wählbar).
- [x] **Dokumentation ehrlich nachgezogen** (u. a. 172→184 Eingaben,
      15→10 Ballkontakte Schwelle, erfundene „Form-Noten“ entfernt,
      Testzahl, CPU/GPU-Satz, veraltete Messwerte gekennzeichnet).

### Nächste Schritte (Phase 8, Kandidaten)

- **Sammeln und Lernen überlappen** (Worker und Lerner parallel, z. B. mit
  einer Warteschlange): Der Gewinn ist **nicht automatisch** — solange beide
  Phasen mit mehr Kernen schneller werden, ist die Summe (heutiges Verhalten)
  genauso schnell wie das Maximum. Er lohnt erst, wenn eine Phase sättigt
  (typisch: Simulation ab ~8-12 Prozessen wegen Speicherbandbreite, während die
  Lern-Threads noch Luft haben). Vor dem Umbau auf dem Zielrechner messen:
  `rocketai benchmark` zeigt die Sättigung im Vergleich 1/2/4 Prozesse.
  Der Umbau braucht einen zweiten Puffer und einen Versatz von einer Runde
  (die KI sammelt mit dem Stand von vor dem Update) — deshalb bewusst noch
  nicht eingebaut.
- **Episodenlängen je Stufe** in den Vorlagen (Stufe 1 kurz: mehr Ballkontakte
  pro Stunde) — gemessen werden muss, ob es das Lernen wirklich beschleunigt.
- **Zufallsstarts-Anteil** als Einstellung (mehr Situationen pro Stunde).
- **Elo-Rangliste** aller Checkpoints in der Arena (Turniermodus).

## Grobe Erwartung

**Nicht die Rechenleistung ist der Engpass, sondern ob das Lernziel stimmt.**
Die *Simulation* schafft auf einem normalen PC schnell ein paar tausend
Schritte pro Sekunde und Kern; Ende-zu-Ende (sammeln + lernen) bleibt davon ein
deutlich kleinerer Teil übrig, weil der Lernschritt die Zeit der Simulations-
prozesse blockiert. Auf mehreren Kernen liegt man realistisch bei einigen
tausend bis wenigen zehntausend Schritten pro Sekunde — die Schrittzahlen unten
sind damit in Stunden bis Tagen erreicht, nicht in Wochen. Ob die KI dabei
wirklich besser spielt, entscheiden Belohnung, Startbedingungen und Bewertung.

| Stand | Schritte (grob, Erfahrungswerte) |
| --- | --- |
| fährt zum Ball, trifft ihn | 20–50 Mio. |
| schießt gezielt Tore | 100–300 Mio. |
| schlägt Psyonix Rookie/Pro | 0,3–1 Mrd. |
| Gold/Platin-Niveau | mehrere Mrd. |

Das echte Tempo zeigt `python3 -m rocketai benchmark` bzw. der Knopf auf der
Einrichtungsseite. Wichtig ist dort die Zeile **Ende-zu-Ende**: Simulation und
Lernschritt laufen nacheinander, deshalb liegt sie deutlich unter der reinen
Simulationsrate (Sandbox-Beispiel: 1 903 → 1 371 Schritte/s). Die Zeiten in der
App rechnen mit dieser Zahl.

## Grenzen und Risiken

- **RLBot v5 ist eine Vorabversion** (Server `v5.0.0-rc17`, Python-Paket
  `2.0.0b56`). Die Versionen sind fest eingetragen; Updates sind bewusste
  Schritte.
- **Sim-zu-Spiel-Lücke:** Einige Werte schätzt der Konverter (z. B. Zeit seit
  dem Sprung, Boost-Pad-Restzeit, Handbremse). Das ist Standard bei
  RLGym-Bots, kann aber in Einzelfällen anders reagieren als im Training.
- **Nur Windows** für das echte Spiel; trainieren geht auch unter Linux.
