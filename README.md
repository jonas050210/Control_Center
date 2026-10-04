# RocketAI

Trainiere deine eigene **Rocket-League-KI**. Sie lernt in einer schnellen
Physik-Simulation (RocketSim) ganz ohne das Spiel und spielt danach im
**echten Rocket League** – offline über [RLBot v5](https://rlbot.org) gegen
Psyonix-Bots, Community-Bots, sich selbst oder dich.

Alles wird über eine lokale Web-App im Browser bedient.

> **Nur offline.** Rocket League nutzt seit April 2026 Easy Anti-Cheat. Bots
> sind online, in Casual, Ranked und privaten Online-Matches verboten.
> RocketAI startet ausschließlich lokale Offline-Matches über RLBot.

> **Befehle:** Alle Beispiele gelten für **Linux/Ubuntu** (und macOS) und
> benutzen deshalb `python3`. Unter **Windows** heißt der Befehl `python`.
> Ausgeführt wird alles im Projektordner (Terminal: `cd` in den Ordner).

## Schnellstart

Voraussetzungen: **Python 3.11–3.13**. Für das echte Spiel zusätzlich
**Windows** und **Rocket League** (Steam oder Epic). Eine Grafikkarte ist
nicht nötig, trainiert wird auf der CPU.

```bash
python3 install.py   # einmalig: venv, PyTorch (CPU), RocketAI, RLBotServer
python3 start.py     # öffnet http://127.0.0.1:8765
```

In der App:

1. **Training → Neues Training →** Vorlage *Schnelltest* (2 Minuten) prüft,
   ob alles läuft.
2. Danach *Autopilot (empfohlen)*: Die KI lernt zuerst, zum Ball zu fahren und
   ihn zu treffen, und schaltet von selbst auf „Tore schießen“ und dann
   „Komplettes Spiel“ um, sobald sie so weit ist. Nebenbei spielt sie 20 % der
   Spiele gegen ältere eigene Versionen (Gegner-Pool), damit sie Gelerntes
   nicht wieder vergisst. Die Kurven „Ballkontakte pro Minute“ und „Siegquote
   gegen ältere Versionen“ zeigen den Fortschritt.
3. **Arena**: Bewertungsspiele als animierte Draufsicht ansehen oder zwei
   Gegner antreten lassen.
4. **Spielen**: Checkpoint wählen, Gegner wählen (Psyonix-Bots, du selbst,
   Community-Bot, KI gegen KI) → Rocket League startet ein Offline-Match.

## Die Web-App

| Seite | Was sie zeigt |
| --- | --- |
| Übersicht | Status in drei Schritten, aktive Trainings, letzte Replays, Rocket-League-Status |
| Training | alle Runs; pro Run Prognose (wann welches Niveau), Kurven mit Checkpoint- und Stufenwechsel-Markern (u. a. Siegquote gegen ältere Versionen, Neugier), Checkpoints, Bewertungen, Protokoll, Stoppen/Fortsetzen |
| **Live** | die KI spielt in Echtzeit in 3D (5 Kameras, 2D umschaltbar) – daneben ihr „Gehirn“: gewählte Aktion, Sicherheit, Controller-Eingaben, Erwartung des Kritikers, Top-5-Alternativen und ein Eingabe-Verlauf der letzten 5 Sekunden. Bei mehreren KI-Autos per Klick auf das Auto-Kärtchen umschalten. Tore mit Effekt und „TOR!“-Einblendung. „Folgt dem Training“ lädt jeden neuen Checkpoint automatisch |
| Arena | Replay-Player in 3D oder 2D (Zeitleiste mit Toren, 0,5–4×) und neue Simulations-Matches |
| Spielen | Rocket-League-Check (installiert? Steam/Epic? läuft es – normal oder im Bot-Modus?) und Match-Start im echten Spiel. Wählbar, wer fährt: **deine KI** oder der **Lehrer (Nexto)** |
| Einrichtung | Systemprüfung, Rocket-League-Check, Befehle, Tastenkürzel, Zeitabschätzung |

Tastenkürzel in Live/Arena: `1`–`5` Kamera, `V` 3D/2D, `F` Vollbild, Leertaste Pause (Replay), `←`/`→` ±5 s.

Training läuft als eigener Prozess weiter, auch wenn der Browser zu ist. Ein
gestopptes Training lässt sich jederzeit fortsetzen.

## Ohne Oberfläche

```bash
python3 -m rocketai teacher                                    # Lehrer laden/prüfen (--test: Testspiel)
python3 -m rocketai train --preset student --name mein-bot     # mit Lehrer (Nexto)
python3 -m rocketai train --preset autopilot --name mein-bot   # ohne Lehrer
python3 -m rocketai train --resume mein-bot --steps 200000000 # fortsetzen mit neuem Ziel
python3 -m rocketai eval runs/mein-bot/checkpoints/latest.pt --opponent chaser teacher
python3 -m rocketai replay runs/mein-bot/checkpoints/latest.pt chaser --out spiel.json
python3 -m rocketai play runs/mein-bot/checkpoints/latest.pt --mode psyonix --skill rookie
python3 -m rocketai play --brain teacher --mode psyonix        # Nexto fährt selbst
python3 -m rocketai doctor                                    # Installation prüfen
```

## Wie es funktioniert

- **Simulation:** RLGym 2 + RocketSim, Self-Play (alle Autos steuert die
  aktuelle KI). Episoden starten zur Hälfte als Anstoß, zur Hälfte zufällig.
- **Lernen:** eigenes PPO in PyTorch mit parallelen Simulationsprozessen.
- **Belohnung in Stufen:** 1 = Ball treffen, 2 = Tore schießen,
  3 = komplettes Spiel inkl. Luftspiel und Boost. Der **Autopilot** wechselt
  die Stufe selbst: 1 → 2 ab ≥ 15 Ballkontakten/min (Schnitt der letzten 20
  Updates, frühestens nach 10 Mio. Schritten), 2 → 3 ab ≥ 1 Tor/min
  (frühestens nach 50 Mio.).
- **Gegner-Pool:** Ein Teil der Spiele läuft gegen die letzten 5 gespeicherten
  Checkpoints. Nur die aktuelle KI lernt daraus; die Siegquote zeigt, ob neue
  Versionen wirklich besser werden.
- **Lehrer (Nexto):** Der stärkste frei verfügbare Community-Bot (Grand
  Champion) kann drei Dinge: (1) in einem einstellbaren Anteil der Spiele
  **gegen** deine KI spielen – der stärkste Hebel, weil die KI dann gegen einen
  richtig guten Gegner lernt, (2) als **Vorbild** dienen, dessen Tasten die KI
  vorhersagen lernt (Anteil fällt über das Training), (3) selbst
  **spielen** – in der Live-Ansicht und im echten Spiel. Seine Dateien (GPL)
  werden geprüft heruntergeladen und bleiben lokal, sie sind nicht Teil des
  Projekts.
- **Echtes Spiel:** Ein Konverter übersetzt RLBots Spielzustand in genau die
  Beobachtung aus dem Training (per Test abgesichert). Der Bot entscheidet wie
  im Training 15-mal pro Sekunde aus 90 Aktionen.

Gemessen (2 Kerne): Der Lehrer gewinnt 6:0 gegen den eingebauten Balljäger, und
Trainingsspiele gegen ihn heben die Ballkontakte der KI von ~1 auf 51–93 pro
Minute. Die Nachahmung allein ist dagegen schwach – warum, steht ausführlich in
[docs/WISSEN.md](docs/WISSEN.md).

**Alles Wissen zum Projekt** (wie die KI sieht und lernt, der Lehrer,
Zeitabschätzungen, Fehlerbehebung, Glossar): [docs/WISSEN.md](docs/WISSEN.md).
Details, Phasen und Zeitabschätzungen: [docs/PLAN.md](docs/PLAN.md).

## Entwicklung

```bash
python3 install.py --dev
.venv/bin/python -m pytest       # Windows: .venv\Scripts\python -m pytest
.venv/bin/ruff check .
```

Trainingsdaten (`runs/`), heruntergeladene Werkzeuge (`tools/`) und lokale
Match-Dateien (`.rocketai/`) werden nicht eingecheckt.

## Lizenz

MIT – siehe [LICENSE](LICENSE).
