# RocketAI

Trainiere deine eigene **Rocket-League-KI**. Sie lernt in einer schnellen
Physik-Simulation (RocketSim) ganz ohne das Spiel und spielt danach im
**echten Rocket League** – offline über [RLBot v5](https://rlbot.org) gegen
Psyonix-Bots, Community-Bots, sich selbst oder dich.

Alles wird über eine lokale Web-App im Browser bedient.

> **Nur offline.** Rocket League nutzt seit April 2026 Easy Anti-Cheat. Bots
> sind online, in Casual, Ranked und privaten Online-Matches verboten.
> RocketAI startet ausschließlich lokale Offline-Matches über RLBot.

## Schnellstart

Voraussetzungen: **Python 3.11–3.13**. Für das echte Spiel zusätzlich
**Windows** und **Rocket League** (Steam oder Epic). Eine Grafikkarte ist
nicht nötig, trainiert wird auf der CPU.

```bash
python install.py   # einmalig: venv, PyTorch (CPU), RocketAI, RLBotServer
python start.py     # öffnet http://127.0.0.1:8765
```

In der App:

1. **Training → Neues Training →** Vorlage *Schnelltest* (2 Minuten) prüft,
   ob alles läuft.
2. Danach *Anfänger 1v1*: Die KI lernt zuerst, zum Ball zu fahren und ihn zu
   treffen. Die Kurve „Ballkontakte pro Minute“ zeigt den Fortschritt.
3. **Arena**: Bewertungsspiele als animierte Draufsicht ansehen oder zwei
   Gegner antreten lassen.
4. **Spielen**: Checkpoint wählen, Gegner wählen (Psyonix-Bots, du selbst,
   Community-Bot, KI gegen KI) → Rocket League startet ein Offline-Match.

## Die Web-App

| Seite | Was sie zeigt |
| --- | --- |
| Übersicht | aktive Trainings, Fortschritt, letzte Replays |
| Training | alle Runs; pro Run Live-Kurven, Checkpoints, Bewertungen, Protokoll, Stoppen/Fortsetzen |
| Arena | 2D-Replay-Player (Zeitleiste mit Toren, 0,5–4× Tempo) und neue Simulations-Matches |
| Spielen | Voraussetzungen und Match-Start im echten Rocket League |
| Einrichtung | Systemprüfung, Befehle, Ordner, Zeitabschätzung |

Training läuft als eigener Prozess weiter, auch wenn der Browser zu ist. Ein
gestopptes Training lässt sich jederzeit fortsetzen.

## Ohne Oberfläche

```bash
python -m rocketai train --preset beginner --name mein-bot   # trainieren
python -m rocketai train --resume mein-bot --steps 200000000 # fortsetzen mit neuem Ziel
python -m rocketai eval runs/mein-bot/checkpoints/latest.pt  # gegen Balljäger/Verteidiger
python -m rocketai replay runs/mein-bot/checkpoints/latest.pt chaser --out spiel.json
python -m rocketai play runs/mein-bot/checkpoints/latest.pt --mode psyonix --skill rookie
python -m rocketai doctor                                    # Installation prüfen
```

## Wie es funktioniert

- **Simulation:** RLGym 2 + RocketSim, Self-Play (alle Autos steuert die
  aktuelle KI). Episoden starten zur Hälfte als Anstoß, zur Hälfte zufällig.
- **Lernen:** eigenes PPO in PyTorch mit parallelen Simulationsprozessen.
- **Belohnung in Stufen:** 1 = Ball treffen, 2 = Tore schießen,
  3 = komplettes Spiel inkl. Luftspiel und Boost.
- **Echtes Spiel:** Ein Konverter übersetzt RLBots Spielzustand in genau die
  Beobachtung aus dem Training (per Test abgesichert). Der Bot entscheidet wie
  im Training 15-mal pro Sekunde aus 90 Aktionen.

Details, Phasen und Zeitabschätzungen: [docs/PLAN.md](docs/PLAN.md).

## Entwicklung

```bash
python install.py --dev
.venv/bin/python -m pytest      # Windows: .venv\Scripts\python -m pytest
.venv/bin/ruff check .
```

Trainingsdaten (`runs/`), heruntergeladene Werkzeuge (`tools/`) und lokale
Match-Dateien (`.rocketai/`) werden nicht eingecheckt.

## Lizenz

MIT – siehe [LICENSE](LICENSE).
