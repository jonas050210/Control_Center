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
| Simulation | `rocketai/env.py` | RocketSim 2.2 + RLGym 2.0 | Physik ~2 300 Schritte/s pro Prozess; Anstoß- und Zufallsstarts |
| Belohnungen | `rocketai/rewards.py` | gestuft (Stufe 1–3) | erst Ball treffen, dann Richtung Tor, dann komplettes Spiel |
| Netz | `rocketai/model.py` | PyTorch-MLP, Actor + Critic getrennt | 172 Beobachtungen → 90 Aktionen; Checkpoint-Format v1 |
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
 RLGym-Env  --obs 172-->  PPO-Policy (MLP)  --checkpoint-->  RLBot-Bot  <-->  Rocket League
            <--Aktion 90--                                   (gleiche Beobachtung,
                                                              gleiche 90 Aktionen,
                                                              8 Ticks pro Entscheidung)
```

Damit die KI im echten Spiel so spielt wie im Training:

- **Gleiche Beobachtung:** `DefaultObs` mit Platz für bis zu 3v3 (172 Werte).
  Der Konverter baut daraus denselben `GameState` wie RocketSim. Ein Test
  vergleicht beide Beobachtungen Wert für Wert.
- **Gleiche Aktionen:** Lookup-Tabelle mit 90 Aktionen, jede 8 Ticks gehalten.
- **Gleiches Timing:** Das Training simuliert RLBots Verzögerung von einem Tick.
- **Gleiche Boost-Pads:** RLBot nummeriert Pads anders; sie werden über ihre
  Position zugeordnet.

## Ordner eines Trainings (`runs/<name>/`)

| Datei | Inhalt |
| --- | --- |
| `config.json` | alle Einstellungen |
| `status.json` | Zustand (läuft/fertig/…), Schritte, Tempo, Herzschlag |
| `metrics.jsonl` | eine Zeile pro PPO-Runde (Ballkontakte, Tore, Belohnung, KL, …) |
| `evaluations.jsonl` | Ergebnisse gegen Balljäger und Verteidiger |
| `checkpoints/` | `<schritte>.pt` und `latest.pt` (mit Optimizer zum Fortsetzen) |
| `replays/` | aufgezeichnete Bewertungsspiele für die Arena |
| `train.log`, `process.log` | Protokoll und rohe Prozessausgabe |
| `control.json` | von der Web-App geschrieben: `{"stop": true}` beendet sauber |

## Phasen

### Phase 1 – Fundament (fertig)

Simulation, PPO-Self-Play, Checkpoints, Bewertung gegen Skript-Bots,
Replays, Web-App, Installer/Starter, RLBot-Bot samt Konverter, Tests, CI.

Erste Messungen in der Sandbox (2 CPU-Kerne):

- Tempo: ~4 700 Schritte/s mit 2 Prozessen (ca. 2 350 pro Prozess)
- PPO-Diagnose: Nach dem Trennen des Gradient-Clippings stieg die KL pro
  Update von ~0,00004 auf ~0,002–0,003; die Clip-Rate liegt bei 1–2 %.
- Skript-Bots: Der Balljäger schlägt „Stillstand“ und „Zufall“ deutlich;
  der Verteidiger ist im Schnitt stärker als der Balljäger.

### Phase 2 – Erstes echtes Training (du, Windows-PC)

1. `python install.py`, dann `python start.py`.
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
| Ballkontakte/min | steigt über Stunden | bleibt flach > 30 Mio. Schritte |
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

## Grobe Erwartung

Erfahrungswerte der RLGym-Community; Grundlage ist das gemessene Tempo von
~2 300 Schritten/s pro Prozess, also etwa 15 000/s auf einem 8-Kern-PC.

| Stand | Schritte (grob) | Zeit auf 8 Kernen |
| --- | --- | --- |
| fährt zum Ball, trifft ihn | 20–50 Mio. | 0,5–1 h |
| schießt gezielt Tore | 100–300 Mio. | 2–6 h |
| schlägt Psyonix Rookie/Pro | 0,3–1 Mrd. | 6–20 h |
| Gold/Platin-Niveau | mehrere Mrd. | Tage bis Wochen |

Das echte Tempo zeigt die Web-App nach den ersten Minuten Training.

## Grenzen und Risiken

- **RLBot v5 ist eine Vorabversion** (Server `v5.0.0-rc17`, Python-Paket
  `2.0.0b56`). Die Versionen sind fest eingetragen; Updates sind bewusste
  Schritte.
- **Sim-zu-Spiel-Lücke:** Einige Werte schätzt der Konverter (z. B. Zeit seit
  dem Sprung, Boost-Pad-Restzeit, Handbremse). Das ist Standard bei
  RLGym-Bots, kann aber in Einzelfällen anders reagieren als im Training.
- **Nur Windows** für das echte Spiel; trainieren geht auch unter Linux.


## Phase 5 - Der Lehrer (Nexto)  [erledigt]

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
