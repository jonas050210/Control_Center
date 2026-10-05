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
nicht nötig (die Simulation läuft immer auf der CPU); ist eine da, nutzt
RocketAI sie automatisch für den Lernschritt.

```bash
python3 install.py   # einmalig: venv, PyTorch (CPU), RocketAI, RLBotServer
python3 start.py     # öffnet http://127.0.0.1:8765
```

Mit NVIDIA-Grafikkarte lohnt `python3 install.py --cuda`: Dann installiert
RocketAI die CUDA-Variante von PyTorch, und der **Lernschritt** läuft auf der
Grafikkarte (die Physik-Simulation bleibt auf der CPU — dort zählt jeder Kern).
Ob es etwas bringt, zeigt `python3 -m rocketai benchmark`.

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
| Übersicht | Status in drei Schritten, aktive Trainings, letzte Replays, Rocket-League-Status; unterbrochene Trainings erscheinen als Banner mit *Jetzt fortsetzen* |
| Training | alle Runs; pro Run Prognose (wann welches Niveau), **Tempo-Zerlegung** (wie viel Zeit sammeln, wie viel lernen), Kurven mit Checkpoint- und Stufenwechsel-Markern (u. a. Siegquote gegen ältere Versionen, Neugier), Checkpoints, Bewertungen, Protokoll, Stoppen/Fortsetzen mit Einstellungen |
| **Live** | die KI spielt in Echtzeit in 3D (5 Kameras, 2D umschaltbar) – daneben ihr „Gehirn“: gewählte Aktion, Sicherheit, Controller-Eingaben, Erwartung des Kritikers, Top-5-Alternativen, ein Eingabe-Verlauf der letzten 5 Sekunden und **„Was die KI sieht“** (Ballabstand, Ballhöhe, Drehung, Boost, Gefahr am eigenen Tor …). Bei mehreren KI-Autos per Klick auf das Auto-Kärtchen umschalten. Tore mit Effekt und „TOR!“-Einblendung. „Folgt dem Training“ lädt jeden neuen Checkpoint automatisch |
| Arena | Replay-Player in 3D oder 2D (Zeitleiste mit Toren, 0,5–4×) und neue Simulations-Matches |
| Spielen | Rocket-League-Check (installiert? Steam/Epic? läuft es – normal oder im Bot-Modus?) und Match-Start im echten Spiel. Wählbar, wer fährt: **deine KI** oder der **Lehrer (Nexto)** |
| Einrichtung | Systemprüfung, Rocket-League-Check, **Geschwindigkeit messen** (echtes Tempo dieses Rechners), „Was die KI sieht“, Befehle, Tastenkürzel, Zeitabschätzung mit dem gemessenen Tempo |

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
python3 -m rocketai benchmark                                 # Tempo dieses Rechners messen
python3 start.py --resume-interrupted                         # nach Absturz/Neustart: unterbrochene Trainings fortsetzen
python3 -m rocketai play runs/mein-bot/checkpoints/latest.pt --mode psyonix --skill rookie
python3 -m rocketai play --brain teacher --mode psyonix        # Nexto fährt selbst
python3 -m rocketai train --resume mein-bot --teacher-opponent 0.25 --teacher-weight 0 \
        --teacher-final-weight 0 --steps 500000000            # Lehrer als Gegner, Nachahmung aus
python3 -m rocketai doctor                                    # Installation prüfen
```

## Wie es funktioniert

- **Simulation:** RLGym 2 + RocketSim, Self-Play (alle Autos steuert die
  aktuelle KI). Episoden starten zur Hälfte als Anstoß, zur Hälfte zufällig.
- **Lernen:** eigenes PPO in PyTorch mit parallelen Simulationsprozessen.
- **Belohnung in Stufen:** 1 = Ball treffen, 2 = Tore schießen,
  3 = komplettes Spiel inkl. Luftspiel und Boost. Der **Autopilot** wechselt
  die Stufe selbst: 1 → 2 ab ≥ 10 *eigenen* Ballkontakten/min (Schnitt der
  letzten 20 Updates, frühestens nach 10 Mio. Schritten), 2 → 3 ab ≥ 1 eigenem
  Tor/min (frühestens nach 50 Mio.). Alle Kennzahlen werden nach Team getrennt
  ausgewiesen: „eigene Ballkontakte“ gehören der lernenden KI, „Gegner“
  der anderen Seite. Vorher zählte die Anzeige die Kontakte **aller** Autos —
  dadurch stieg die Zahl auch dann, wenn nur der Gegner am Ball war.
- **Tempo:** Sammeln und Lernen laufen im selben Prozess nacheinander — während
  eines Lernschritts stehen die Simulationsprozesse still. Deshalb rechnen alle
  Zeitangaben mit dem **Ende-zu-Ende-Tempo** (sammeln + lernen), das
  `rocketai benchmark` in der Zeile *Ende-zu-Ende* ausgibt. Der Lernprozess
  nimmt dabei `torch_threads = Kerne − 1` (im Formular einstellbar), weil ihm
  in dieser Zeit die Kerne allein gehören.
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

- **Was die KI sieht:** Der Zustand wird nicht als Bild, sondern als Zahlen
  übergeben — 172 Werte aus RLGyms `DefaultObs` (Ball samt Drehung,
  Boost-Pads, alle Autos) plus 12 eigene Zusatzwerte (`rocketai/obs.py`):
  Ballposition relativ zum Auto, Abstand, Ballgeschwindigkeit, Balldrehung,
  Ballhöhe, ob der Ball auf das eigene Tor zuläuft, eigene Hälfte, Boost,
  Bodenkontakt, Überschall. Die Live-Ansicht zeigt diese Werte unter
  **„Was die KI sieht“**. Ein Test vergleicht Simulation und echtes Spiel
  Wert für Wert für beide Varianten.
- **Sperre pro Training:** Ein Run kann nur von *einem* Prozess trainiert
  werden (Dateisperre, siehe `rocketai/runtime.py`) — auch wenn Weboberfläche
  und Kommandozeile gleichzeitig starten wollen.
- **Messen statt raten:** `python3 -m rocketai benchmark` (oder der Knopf in
  *Einrichtung*) misst auf dem eigenen Rechner Schritte/s, Echtzeit-Faktor und
  die Dauer eines Lernschritts.

Gemessen (2-Kern-Sandbox, `rocketai benchmark`): 1 353 Schritte/s mit einem
Prozess und einem Spiel, **1 903 Schritte/s** mit vier Spielen pro Prozess
(1,4× mehr — das teilt sich die Startkosten der Prozesse), ein PPO-Lernschritt
mit ~15 000 Schritten/s. Entscheidend ist die letzte Zeile: **Ende-zu-Ende
bleiben 1 371 Schritte/s**, weil Sammeln und Lernen nacheinander laufen und
während des Lernens alle Simulationsprozesse warten. Die früher genannten
„4 200 Schritte/s“ waren die reine Simulationsrate — daher kommen die
optimistischen Zeitangaben von damals. Der Balljäger wird vom Lehrer 6:0
geschlagen. Wie stark Trainingsspiele gegen den Lehrer wirklich helfen, muss
mit den **getrennten** Kennzahlen neu gemessen werden: die früher genannten
„51–93 Ballkontakte pro Minute“ wurden noch mit dem alten Zähler ermittelt, der
die Kontakte beider Teams addierte. Die Nachahmung allein ist schwach – warum,
steht ausführlich in [docs/WISSEN.md](docs/WISSEN.md).

**Alles Wissen zum Projekt** (wie die KI sieht und lernt, der Lehrer,
Zeitabschätzungen, Fehlerbehebung, Glossar): [docs/WISSEN.md](docs/WISSEN.md).
Details, Phasen und Zeitabschätzungen: [docs/PLAN.md](docs/PLAN.md).

## Entwicklung

```bash
python3 install.py --dev
.venv/bin/python -m pytest       # Windows: .venv\Scripts\python -m pytest
                                 # 120 Tests (mit RLBot-Paket 2.0.0b56 und Lehrer alle grün)
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Trainingsdaten (`runs/`), heruntergeladene Werkzeuge (`tools/`) und lokale
Match-Dateien (`.rocketai/`) werden nicht eingecheckt.

## Lizenz

MIT – siehe [LICENSE](LICENSE).
