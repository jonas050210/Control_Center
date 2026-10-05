# RocketAI — das ganze Wissen

Diese Datei ist das Nachschlagewerk: Wie die KI funktioniert, warum sie so
gebaut ist, was gemessen wurde und was (noch) nicht geht. Sie ist bewusst
ehrlich — inklusive der Stellen, an denen etwas nur teilweise klappt.

Inhalt

1. [Das Wichtigste in einem Absatz](#1-das-wichtigste-in-einem-absatz)
2. [Wie die KI das Spiel sieht](#2-wie-die-ki-das-spiel-sieht)
3. [Wie die KI spielt (Beobachtung → Aktion)](#3-wie-die-ki-spielt-beobachtung--aktion)
4. [Wie sie lernt (PPO)](#4-wie-sie-lernt-ppo)
5. [Belohnung: die drei Stufen](#5-belohnung-die-drei-stufen)
6. [Der Lehrer (Nexto)](#6-der-lehrer-nexto)
   - 6.1 [Warum überhaupt ein Lehrer?](#61-warum-überhaupt-ein-lehrer)
   - 6.2 [Die drei Arten, wie der Lehrer benutzt wird](#62-die-drei-arten-wie-der-lehrer-benutzt-wird)
   - 6.3 [Was gemessen wurde (und was nicht funktioniert)](#63-was-gemessen-wurde-und-was-nicht-funktioniert)
   - 6.4 [Technisch: wie Nexto angebunden ist](#64-technisch-wie-nexto-angebunden-ist)
   - 6.5 [Rechtliches / Lizenz](#65-rechtliches--lizenz)
7. [Stärker werden ohne Lehrer: Gegner-Pool und Autopilot](#7-stärker-werden-ohne-lehrer-gegner-pool-und-autopilot)
8. [Das echte Rocket League (RLBot)](#8-das-echte-rocket-league-rlbot)
9. [Was „professionell“ heißt — realistisch](#9-was-professionell-heißt--realistisch)
10. [Zeit- und Hardware-Erwartungen](#10-zeit-und-hardware-erwartungen)
11. [Woran erkenne ich, ob es gut läuft?](#11-woran-erkenne-ich-ob-es-gut-läuft)
12. [Dateien und Befehle](#12-dateien-und-befehle)
13. [Häufige Fehler und Lösungen](#13-häufige-fehler-und-lösungen)
14. [Glossar](#14-glossar)
15. [Regeln, die nie gebrochen werden](#15-regeln-die-nie-gebrochen-werden)

---

## 0. Kurz zur Umgebung (bitte zuerst lesen)

- **Betriebssystem: Linux/Ubuntu.** Alle Befehle heißen dort **`python3`** —
  nicht `python` (das gibt es auf Ubuntu oft gar nicht oder es ist Python 2).
  Unter Windows wäre es `python`, sonst ist alles gleich.
- Jeder Befehl wird **im Projektordner** ausgeführt (`cd SandboxAI`); die
  virtuelle Umgebung (`.venv`) wird von `install.py` angelegt und von
  `start.py` automatisch benutzt.
- Auf der Seite **Einrichtung** in der App steht der passende Befehl für dein
  System immer fertig zum Kopieren.
- Die **Simulation** läuft immer auf der **CPU** (dort zählt jeder Kern). Der
  **Lernschritt** kann auf einer NVIDIA-Grafikkarte laufen (`python3 install.py
  --cuda`, Einstellung `device`), siehe Abschnitt 10.

---

## 1. Das Wichtigste in einem Absatz

RocketAI trainiert eine eigene neuronale KI für Rocket League. Trainiert wird
in **RocketSim**, einer exakten Nachbildung der Spielphysik (schneller als das
echte Spiel, ohne Grafik). Gespielt wird über **RLBot** im echten Rocket League
— nur offline, ohne Anti-Cheat. Gelernt wird mit **PPO** (einem
Standard-Verfahren für Robotik und Spiele). Zusätzlich kann die KI von
**Nexto**, dem stärksten frei verfügbaren Community-Bot (Grand-Champion-Niveau),
lernen: Sie spielt gegen ihn und versucht, ihn nachzuahmen. Deine KI bleibt
immer dein eigenes Netz — Nexto liefert nur Beispiele bzw. den Gegner.

---

## 2. Wie die KI das Spiel sieht

Die KI „sieht“ kein Bild. Sie bekommt 15-mal pro Sekunde eine **Liste von 184
Zahlen** (im Training) bzw. genau dieselbe Liste aus dem echten Spiel (über
RLBot). Die Live-Ansicht zeigt sie unter **„Was die KI sieht“**.

Die ersten 172 Zahlen sind der Standardteil (RLGym `DefaultObs`), die letzten
12 sind eigene Ergänzungen aus `rocketai/obs.py`:

| Bereich | Inhalt |
|---|---|
| Ball | Position, Geschwindigkeit, **Drehung** — je normal und gespiegelt |
| Autos | Position, Tempo, Drehung, Ausrichtung (Nase oben), Boost, am Boden?, Überschall?, wird gerade geboostet?, Demo-Timer, Flip übrig? |
| Boost-Pads | welche gerade voll sind |
| Zusatz (12) | Ball relativ zum Auto (x/y/z), Abstand, Ballgeschwindigkeit, **Balldrehung**, Ballhöhe, „Ball fliegt auf unser Tor zu“, Ball in unserer Hälfte, eigener Boost, Räder am Boden, Überschall |

Die Zusatzwerte sind bewusst einfach und schnell zu rechnen: Sie geben der KI
Dinge, die sie aus den Rohwerten sonst mühsam lernen müsste (z. B. „der Ball
kommt auf mein Tor“). Wichtig ist die **Sperre** zwischen Training und echtem
Spiel: Der Bauplan steckt als Zahl (`obs_size`) im Checkpoint, der RLBot-Bot
baut deshalb immer genau die Beobachtung, mit der seine Gewichte trainiert
wurden. Ein Test vergleicht beide Wege Wert für Wert.

Wichtig: Alles wird **relativ zu deinem Auto** sortiert und gedreht, damit die
KI nicht Blau und Orange doppelt lernen muss. Für sie ist das eigene Tor immer
hinten. Sie ist damit auch nicht auf 1v1 festgelegt: Dieselbe Beschreibung
funktioniert für 1v1, 2v2 und 3v3 (fehlende Mitspieler werden aufgefüllt).

Ein Detail aus der Bauphase: Die Beobachtung enthält die *gespiegelte* Fassung
für beide Teams. Das klingt doppelt, ist aber Absicht — RLGym macht das so, und
die KI lernt dadurch schneller, weil beide Perspektiven in denselben Zahlen
auftauchen.

---

## 3. Wie die KI spielt (Beobachtung → Aktion)

Die KI entscheidet nicht pixelgenau, sondern wie ein Mensch mit Controller:
Es gibt **90 erlaubte Eingaben** (Aktionen). Sie wählt jede Runde genau eine
davon, 15-mal pro Sekunde (alle 8 Physik-Ticks):

- Gas / Rückwärts / Nichts
- Lenken links/rechts
- Springen, und beim Springen: Nicken/Gieren/Rollen in alle Richtungen
  (das ist der Flip bzw. der Aerial in eine Richtung)
- Boost an/aus
- Drift (Handbremse)

Ein Beispiel: „Sprung + Nicken vorwärts + Boost“ ist ein Vorwärts-Flip mit
Boost — die häufigste Art, schnell zum Ball zu kommen. „Springen + Nicken
zurück + Rollen“ startet einen Rückwärts-Aerial.

**Übersprungene Physik.** Die Simulation rechnet 120-mal pro Sekunde, die KI
entscheidet nur jedes 8. Mal. Dazwischen hält sie ihre Eingabe. Genau so ist es
auch im echten Spiel (Sonnenuntergang: keine KI entscheidet 120-mal pro
Sekunde sinnvoll). Wichtig: Das wird in der Simulation **genauso** eingestellt
wie über RLBot, inklusive der einen Tick Verzögerung, die RLBot technisch
bedingt hat. Sonst fährt die KI im echten Spiel an allem vorbei.

**Das Netz** ist ein klassisches kleines Netz: 184 Eingaben (mit den
Zusatzwerten; 172 ohne, siehe Abschnitt 2) → drei Schichten (512, 512, 256) →
90 Werte für die Aktionen, plus ein zweiter Kopf, der den Wert der Situation
schätzt (der „Kritiker“, siehe Abschnitt 4). Es hat rund 1,0 Mio. Parameter
(davon ~512.000 im Akteur). Nexto hat Millionen — aber ein größeres Netz braucht
proportional mehr Daten, es ist also keine Wunderwaffe.

---

## 4. Wie sie lernt (PPO)

**PPO** (Proximal Policy Optimization) ist der Standard für solche Aufgaben.
Der Ablauf alle paar Sekunden:

1. Die KI spielt in vielen Simulationen gleichzeitig (z. B. 8 Spiele parallel).
2. Sie sammelt dabei ca. 50.000 Entscheidungen.
3. Sie schaut, welche Entscheidungen **besser als erwartet** waren (Vorteil),
   und verstärkt sie leicht („Clips“ verhindern zu große Schritte).
4. Sie wiederholt das viele Millionen Mal.

Wichtige Einstellungen (Standard, alle im Formular änderbar):

| Einstellung | Wert | Bedeutung |
|---|---|---|
| Lernrate | 3e-4 | wie stark pro Schritt geändert wird |
| Gamma | 0,99 | Weitsicht (1 = blickt weit in die Zukunft) |
| GAE-Lambda | 0,95 | wie stark die Vorteile geglättet werden |
| Clip | 0,2 | Sicherheitsgrenze pro Update |
| Epochen | 3 | wie oft dieselben Daten durchgekaut werden |
| Minibatch | 10.000 | Häppchengröße pro Update |
| Entropie (Neugier) | 0,01 | Bonus für Abwechslung |

**Ein gemachter Fehler, den man kennen sollte:** Anfangs wurde der
Gradient von Akteur und Kritiker gemeinsam begrenzt. Der Kritiker hat dann mit
seinen großen Werten den Akteur mitgebremst und es wurde fast nichts gelernt.
Jetzt werden beide getrennt begrenzt — das war einer der größten Sprünge der
ganzen Entwicklung.

---

## 5. Belohnung: die drei Stufen

Belohnung = was die KI als „gut“ lernt. Zu viel auf einmal geht nicht, deshalb
gibt es Stufen:

| Stufe | Name | Was zählt |
|---|---|---|
| 1 | Ballkontakt | Ball berühren (Gewicht 10), auf den Ball zufahren, zum Ball schauen; ein Tor zählt 20 |
| 2 | Tore | deutliches Signal fürs Angreifen und Tore |
| 3 | Komplettes Spiel | Verteidigen, Tore, Boost, Tempo |

Die Stufen bestehen aus genau acht Bestandteilen (`rocketai/rewards.py`,
`STAGE_WEIGHTS`): `speed_to_ball` (Tempo in Ballrichtung),
`face_ball` (Ausrichtung), `touch` (Ballberührung), `in_air`
(winziger Anreiz, nicht nur zu fahren), `ball_to_goal` (Ballgeschwindigkeit
Richtung Tor — in Stufe 1 **noch 0**, in Stufe 2 Gewicht 2, in Stufe 3
Gewicht 3), `goal` (Tor, minus bei Gegentor), `boost_keep` (nur Stufe 3) und
`air_touch` (nur Stufe 3). Die frühere Beschreibung („Form-Noten“, Strafen für
„nicht bewegen“) war falsch: einen Gegentor-Term gibt es über `goal = -1`, alles
andere sind positive Formsignale.

**Autopilot**: Statt selbst zu entscheiden, wann die Stufe wechselt, kannst du
den Autopilot anschalten. Er wechselt, wenn die Zahlen aus den letzten 20
Updates es zeigen:

- Stufe 1 → 2: mindestens **10 eigene** Ballkontakte pro Minute (Schnitt der
  letzten 20 Updates) und mindestens 10 Mio. Schritte.
- Stufe 2 → 3: mindestens 1 Tor pro Minute und mindestens 50 Mio. Schritte.

Die Messung: Ein Mini-Lauf mit künstlich niedrigen Schwellen lief durch
(Stufenfolge 1, 1, 2, 2, 3), die Umschaltung selbst ist also getestet.

---

## 6. Der Lehrer (Nexto)

### 6.1 Warum überhaupt ein Lehrer?

Von null zu lernen ist ehrlich, aber langsam: Bis die KI zuverlässig trifft,
kann es auf einem normalen Rechner Stunden bis Tage dauern. Die schnellste Art,
ein starkes Netz zu bekommen, ist deshalb, einem starken Vorbild nachzufolgen
und danach selbst weiterzutrainieren.

**Nexto** ist der beste frei verfügbare Bot der RLBot-Community: ein
Aufmerksamkeitsnetz („Transformer“), trainiert von hunderten Freiwilligen über
Monate auf einem Botpack-Server. Er wurde in RLGym trainiert (genau dem
Framework, das auch hier benutzt wird) und spielt etwa Grand-Champion-Niveau.
Sein Netz liegt öffentlich als Datei mit 1,8 MB vor.

### 6.2 Die drei Arten, wie der Lehrer benutzt wird

1. **Als Gegner im Training (der stärkste Hebel).** In einem einstellbaren
   Anteil der Trainingsspiele steuert Nexto das gegnerische Auto. Deine KI
   lernt dann am echten Wettkampf gegen einen sehr guten Gegner. Das ist
   echtes Reinforcement Learning, kein Abschauen.
2. **Als Vorbild („Nachahmung“).** Zusätzlich kann deine KI versuchen, Nextos
   Tasten vorherzusagen. Das passiert über Kreuzentropie auf die
   Wahrscheinlichkeiten, die Nexto für die 90 Aktionen ausgibt. Der Anteil
   fällt über das Training (z. B. von 100 % auf 10 %), damit die KI zum
   Schluss eigenen Wegen folgt statt zu kopieren.
3. **Zum Zuschauen und Spielen.** Der Lehrer kann in der 3D-Live-Ansicht
   spielen, in Replays, und im echten Rocket League als dein Bot fahren
   (Auswahl „Lehrer (Nexto)“ auf der Spielen-Seite). Damit siehst du sofort,
   wie stark ein Bot sein kann, und kannst gegen ihn spielen.

Was der Lehrer **nicht** ist: Deine KI wird nicht zu Nexto. Deine KI bleibt
dein eigenes Netz mit eigenen Gewichten; der Lehrer liefert Beispiele und
Gegner. Alles, was er beiträgt, ist messbar und abschaltbar.

### 6.3 Was gemessen wurde (und was nicht funktioniert)

Gemessen in dieser Umgebung (2 CPU-Kerne, kleine Netze), damit hier keine
Wunschzahlen stehen:

| Messung | Ergebnis |
|---|---|
| Lehrer gegen „Balljäger“ (eingebauter Bot), 60 s | **6:0**, 60 zu 14 Ballkontakte |
| Geschwindigkeit des Lehrers | ~3.500 Antworten/s (2 Kerne), ~640 Entscheidungen/s im Live-Spiel |
| Training gegen den Lehrer (Anteil 40 %) | Ballkontakte der KI steigen von ~1 auf 51–93 pro Minute — **mit dem alten Zähler** gemessen, der die Kontakte beider Teams addierte. Mit getrennten Zahlen („eigene“ vs. „Gegner“) muss das neu gemessen werden; die Richtung (viel mehr Ballkontakte) bleibt richtig. |
| Nachahmung allein (reine Kreuzentropie, 20.000 Beispiele, mehrere Netze) | Abweichung fällt nur von 4,50 auf ~3,7–4,0 statt auf die Zielgröße 2,9 |

**Ehrliche Einordnung.** Das Spielen **gegen** den Lehrer ist der Durchbruch:
Es erzeugt ein um Größenordnungen dichteres Lernsignal (viele Ballkontakte
statt wenigen). Die **Nachahmung** dagegen ist deutlich schwächer als
gehofft: Nexto entscheidet extrem scharf und hängt an Feinheiten, die ein
kleines Netz aus unserer Beobachtung nur schwer nachbildet. Tests mit den
besten verfügbaren Mitteln (auch mit Nextos eigener Beobachtungsform,
verschiedenen Netzgrößen, mehreren Lernraten, Stapeln von Objekten statt
flachen Netzen) verbesserten die Abweichung nur mäßig. Deshalb:

- Standard ist: **gegen den Lehrer spielen (25 %)** + Nachahmung als Beigabe
  (100 % → 10 %).
- Die Nachahmung ist kein Versprechen auf Grand-Champion-Niveau. Der ehrliche
  Weg zu „richtig gut“ bleibt: gegen den Lehrer trainieren, lange und mit
  vielen Kernen.

### 6.4 Technisch: wie Nexto angebunden ist

Kurzfassung für Interessierte:

- Nextos Netz ist ein **TorchScript**-Modul (`torch.jit.load`) — es läuft in
  unserem PyTorch mit, ohne fremde Pakete.
- Nexto erwartet eine Beobachtung als drei Felder: ein eigenes Merkmal (`q`,
  32 Werte), Merkmale aller Objekte (`kv`, 37 × 24 Werte) und eine Maske.
  Unsere Simulation liefert diese Zahlen über `expand_compacts()`.
- Die Aktionstabelle ist **identisch**: Die 90 Aktionen sind in beiden Systemen
  dieselben (nachgeprüft, `test_teacher.py`).
- Die Aktualisierungsrate passt ebenfalls: Nexto entscheidet wie unsere KI alle
  8 Ticks.
- Unsere Kodierung wird gegen Nextos eigene Funktion geprüft: Die Abweichung
  beträgt weniger als 1/10.000 (nur Fließkomma-Rundung). Die Quaternionen
  dürfen sich im Vorzeichen unterscheiden — das ist dieselbe Drehung.
- Geprüft wurde außerdem, dass unsere Rotationsmatrix-Konvention für „Nase“
  und „oben“ exakt mit der alten RLGym-Version übereinstimmt (Abweichung
  1e-16). Ohne diese Prüfung hätte der Lehrer in unserer Simulation Blindflug
  gespielt.
- Das Netz versteht auch Spiele mit mehr Autos: 1v1, 2v2 und 3v3 wurden
  getestet.

### 6.5 Rechtliches / Lizenz

- Nextos Dateien stehen unter **GPL-3.0** und sind **nicht Teil dieses
  Projekts**. Sie werden beim ersten Gebrauch von GitHub geladen und landen in
  `tools/nexto` (nicht im Git-Repository, siehe `.gitignore`).
- Aus Sicherheitsgründen wird jede Datei über eine fest hinterlegte
  **Prüfsumme (SHA-256)** verifiziert. Passt sie nicht — z. B. weil sich die
  Datei bei GitHub geändert hat — wird sie **nicht** geladen.
- Nutzung nur **lokal und offline**, wie der Rest des Projekts. Nicht
  weiterverbreiten.
- Ohne Lehrer läuft alles weiter: Alle Funktionen außer den Lehrer-Funktionen
  brauchen ihn nicht.

---

## 7. Stärker werden ohne Lehrer: Gegner-Pool und Autopilot

**Das Problem, das der Gegner-Pool löst:** Wenn die KI immer nur gegen sich
selbst spielt, vergisst sie. Sie wird gut gegen ihre eigene aktuelle Marotte
und schlecht gegen alles andere (bekannt als „Rock-Paper-Scissors“-Effekt in
Selbstspiel-Trainings).

**Die Lösung:** Jeder gespeicherte Zwischenstand (Checkpoint) wandert in einen
Pool. Ein Teil der Trainingsspiele läuft gegen einen zufälligen Stand aus
diesem Pool (Standard: 20 %, 5 Stände). Trainiert wird nur auf der Seite der
aktuellen KI. Nebeneffekt: Es gibt eine ehrliche Fortschrittszahl — die
**Siegquote gegen ältere Versionen**. Über 50 % heißt wirklich besser.

**Messwerte aus einem Testlauf** (2 Kerne, ca. 4 Mio. Schritte, ebenfalls noch
mit dem alten, teamübergreifenden Zähler):

| Schritte | Ballkontakte/min |
|---|---|
| 2,4 Mio. | 2,5 |
| 3,25 Mio. | 10,0 |
| 4,0 Mio. | ~15 |

Für neue Läufe zählt nur die Zeile „eigene Ballkontakte“ (und daneben
„Gegner“) in den Kurven — siehe Abschnitt 11.

---

## 8. Das echte Rocket League (RLBot)

- RLBot ist eine von Psyonix offiziell erlaubte Schnittstelle. Sie schickt
  120-mal pro Sekunde den Spielzustand und nimmt Steuerbefehle entgegen.
- Seit April 2026 hat Rocket League Easy Anti-Cheat. Im Startmenü gibt es
  „Spielen ohne Easy Anti-Cheat“ — nur damit funktionieren Bots. Genau das
  startet RLBot.
- **Online, Ranked oder private Online-Matches sind tabu** — dort ist
  Anti-Cheat aktiv, und Bots dort wären Betrug. Das ist keine Einschränkung der
  Technik, sondern eine Regel.
- Für das echte Spiel braucht es **Windows**, eine installierte
  Rocket-League-Version (Steam oder Epic) und den `RLBotServer`
  (`python3 install.py` lädt ihn).
- Der Bot entscheidet im echten Spiel genauso wie im Training: alle 8 Ticks,
  dieselbe Beobachtung. Der Konverter (`rocketai/rlbot_convert.py`) baut die
  Beobachtung aus den RLBot-Paketen; Tests stellen sicher, dass die Reihenfolge
  der Autos und die Boost-Pads zur Simulation passen.

---

## 9. Was „professionell“ heißt — realistisch

Wichtig, damit keine falschen Erwartungen entstehen:

- **Keine KI der Welt spielt Rocket League auf Profi-Niveau.** Turnier-Profis
  bewegen sich in Bereichen, die auch die besten Bots (Nexto, Necto) nicht
  erreichen — unter anderem wegen der Mechanik-Ausführung und der
  Spielintelligenz über Sekunden hinweg.
- **Grand Champion** (oberstes ~1 % der Spieler) ist mit einem Bot erreichbar —
  Nexto beweist das. So weit zu kommen erfordert aber sehr viel Training
  (Monate auf vielen Kernen), nicht ein Wochenende.
- **Professionelle Mechanics** im Sinne von „sieht aus wie ein Pro“: Ein Bot
  bekommt vieles davon hin (schnelle Anstöße, Aerials, Wandspiel, sauberes
  Fahren), ist aber typischerweise unbeweglicher und weniger kreativ als ein
  Mensch.
- Was hier realistisch herauskommt: eine KI, die erkennbar Rocket League
  spielt, die Psyonix-Bots schlagen kann und die gegen den Lehrer Stärke
  zeigt. Alles darüber ist Trainingszeit.

---

## 10. Zeit- und Hardware-Erwartungen

Grobe Erfahrungswerte für **Schritte** (RLGym-Community). Die Zeit hängt vom
Rechner ab — deshalb gibt es `python3 -m rocketai benchmark` (oder den Knopf
*Einrichtung → Geschwindigkeit messen*).

**Die wichtigste Zahl ist das Ende-zu-Ende-Tempo.** Sammeln und Lernen laufen
im selben Prozess *nacheinander*: Erst spielt die KI (Simulation, alle
Simulationsprozesse arbeiten), dann rechnet sie über genau diese Daten
`epochs` mal nach — und **währenddessen stehen alle Simulationsprozesse
still**. Ein Stapel von `S` Schritten braucht deshalb

```
S / Sim-Tempo  +  epochs · S / Lern-Tempo
```

Sekunden. Die reine Simulationsrate (z. B. „4 200 Schritte/s“) ist also *nicht*
das, was du bekommst: Bei 3 Epochen und einem Lernschritt mit 20 000
Schritten/s bleiben von 4 200 nur rund **2 600 Schritte/s** übrig, von 31 000
(16 Kerne) nur rund 5 500. Ab etwa 8 Kernen bringt es deshalb fast nichts mehr,
weitere Simulationsprozesse aufzumachen — der Lernschritt ist dann der
Engpass. Genau diese Rechnung steckt jetzt in jeder Zeitangabe der App
(`report["effective"]` im Benchmark, `steps_per_second` im Protokoll).

**Lern-Threads.** Während des Lernschritts sind alle Simulationsprozesse
blockiert, der Lernprozess darf deshalb fast alle Kerne nehmen: Standard ist
`torch_threads = Kerne − 1` (Einstellung im Formular, 0 = automatisch). Vorher
nahm er nur ein Viertel der Kerne (höchstens 8) — auf einem 16-Kern-Rechner
lernte er dann mit 4 Threads, während ~11 Kerne warteten. Das war einer der
größten Bremsklötze.

| Stufe | Schritte (Erfahrungswerte) |
|---|---|
| Ball treffen | 20–50 Mio. |
| Tore schießen | 100–300 Mio. |
| Psyonix Rookie/Pro schlagen | 0,3–1 Mrd. |
| Gold/Platin | Milliarden |
| Mit Lehrer (gegen ihn gespielt) | 10–100 Mio. |

Gemessen in dieser 2-Kern-Sandbox (echte Messung, `rocketai benchmark`):

| Einstellung | Schritte/s | Echtzeit |
|---|---|---|
| 1 Prozess × 1 Spiel | 1 353 | 43× |
| 1 Prozess × 2 Spiele | 1 668 | 52× |
| 1 Prozess × 4 Spiele | 1 903 | 42× |
| Lernschritt (5 000 Schritte × 3 Epochen) | 14 700–16 400 | – |
| **Ende-zu-Ende** | **1 371** | – |

Zwei Dinge sieht man daran gut: **Mehr Spiele pro Prozess** sind schneller als
eines (1,4×, weil sich die Startkosten der Prozesse teilen) — genau das wurde
vorher nicht gemessen, und die Empfehlung fiel dadurch zu niedrig aus. Und vom
Simulationstempo bleibt Ende-zu-Ende nur ein Teil übrig (hier 72 % Simulation,
28 % Lernen).

Wie die beiden Phasen mit Kernen skalieren (dieselbe Sandbox, je 4 s):

| Kerne/Threads | Simulation (Schritte/s) | Lernschritt (Schritte/s) |
|---|---|---|
| 1 | 1 656 (1 Prozess × 1 Spiel) | 16 982 (1 Thread) |
| 2 | 4 177 (2 Prozesse × 2 Spiele) | 33 269 (2 Threads) |

Beide skalieren hier fast linear. Das ist die ehrliche Begründung dafür, dass
**Überlappung** (sammeln und lernen gleichzeitig) *nicht* eingebaut ist: Sie
bringt nur etwas, wenn eine Phase früher sättigt als die andere (z. B. wenn
zusätzliche Simulationsprozesse wegen Speicherbandbreite nichts mehr bringen,
die Lern-Threads aber noch Luft haben). Ob das auf deinem Rechner so ist, zeigt
der Benchmark-Vergleich „1/2/4 Prozesse“ — die Empfehlung sagt es im Klartext
(„skaliert gut“ bzw. „bringt kaum etwas“). Steht eine Phase still, während die
andere rechnet, wäre Überlappung der nächste große Schritt (siehe
`docs/PLAN.md`, Phase 8).

Ehrliche Einordnung: Bei einigen tausend Schritten pro Sekunde sind die
Schrittzahlen oben *nicht* an einem Tag „abgearbeitet“ — für 200 Mio. Schritte
braucht diese Sandbox rund zwei Tage. Ob die KI dann wirklich gut spielt,
entscheidet nicht die Rechenleistung, sondern ob Belohnung, Startbedingungen
und Bewertung stimmen. Deshalb zeigt die Oberfläche nach Team getrennte
Kennzahlen, den Echtzeit-Faktor und die **Tempo-Zerlegung** (Sammeln vs.
Lernen).

Mehr Kerne = fast proportional schneller, aber nur bis der Lernschritt
dominiert (siehe oben). Eine Grafikkarte beschleunigt nur den Lernschritt,
nicht die Simulation. Der Webbrowser darf zu sein, das Training läuft im
Hintergrund weiter (es ist ein eigener Prozess).

**Läuft der PC aus / Standby oder stürzt etwas ab:** Das Training pausiert. Die
Oberfläche zeigt solche Runs als „Unterbrochen“ mit einem Knopf *Jetzt
fortsetzen*; ohne Oberfläche geht es mit `python3 start.py --resume-interrupted`
weiter (setzt beim letzten `latest.pt` an). Für unbeaufsichtigte Nächte kann man
das in die Windows-Aufgabenplanung eintragen.

---

## 11. Woran erkenne ich, ob es gut läuft?

- **Eigene Ballkontakte pro Minute** steigt zügig: unter 5 → läuft nicht
  richtig; über 10 → die KI trifft den Ball zuverlässig. Die Zeile „Gegner“
  daneben gehört der anderen Seite: Steigt nur sie, wird die KI nicht besser.
- **Eigene Tore pro Minute** kommt später. 1+ ist der Punkt zum Weiterdrehen;
  **Gegentore** sollten dabei nicht stärker wachsen als die eigenen.
- **Tempo-Zerlegung** (Training-Seite, Karte „Tempo-Zerlegung“): Zeigt, wie
  viel Zeit des letzten Updates das Sammeln und wie viel das Lernen gebraucht
  hat. Über 50 % Lernen heißt: Der Lernschritt bremst — dann helfen weniger
  Epochen, mehr `torch_threads` oder eine Grafikkarte, aber *keine* weiteren
  Simulationsprozesse.
- **Echtzeit-Faktor** (`x… Echtzeit` im Protokoll, `realtime_factor` in den
  Messwerten) zeigt, wie viel simulierte Spielzeit pro Sekunde läuft — kein
  Lernfortschritt, sondern Tempo.
- **Belohnung pro Episode** steigt tendenziell, kann aber stark schwanken (das
  ist normal, es wird geglättet angezeigt).
- **Siegquote gegen ältere Versionen** sollte um/über 50 % gehen. Dauerhaft
  unter 45 % heißt: Das Training läuft rückwärts (zu große Lernrate, oder der
  Pool ist zu stark).
- **Siegquote gegen den Lehrer**: bleibt lange bei 0 %. Das ist völlig normal
  und kein Fehler. Die Tore (x:y) zeigen aber, wie viel es wird.
- **Abweichung (Nachahmung)**: sinkt es, lernt die KI den Lehrer; ein Plateau
  um 4 ist zu erwarten (siehe 6.3).

---

## 12. Dateien und Befehle

> **Betriebssystem.** Entwickelt und benutzt wird das Projekt unter
> **Linux/Ubuntu** (und macOS) – dort heißt der Python-Befehl **`python3`**.
> Unter **Windows** heißt er **`python`**; alle Befehle sind sonst identisch.
> Immer zuerst in den Projektordner wechseln (`cd SandboxAI`).
>
> Damit du nicht in die Falle läufst: `python` gibt es auf Ubuntu oft gar nicht
> oder es ist Python 2. Wenn ein Befehl „command not found“ meldet, nimm
> `python3`. Die App selbst zeigt auf der Seite **Einrichtung** genau den
> Befehl an, der zu deinem System passt.

**Starten**

```bash
python3 install.py            # einmalig: Pakete, RLBot-Server
python3 start.py              # Webbrowser-App
python3 start.py --resume-interrupted   # ... und unterbrochene Trainings fortsetzen
python3 -m rocketai doctor    # prüft alles (Python, Torch, RLBot, Lehrer, ...)
python3 -m rocketai teacher   # Lehrer laden und prüfen (--test: Testspiel)
```

**Trainieren**

```bash
python3 -m rocketai train --preset student --name mein-bot   # mit Lehrer
python3 -m rocketai train --preset autopilot --name mein-bot  # ohne Lehrer
python3 -m rocketai train --resume mein-bot --steps 100000000
# Lehrer als Gegner, Nachahmung aus (schnellste Kombination):
python3 -m rocketai train --resume mein-bot --teacher-opponent 0.25 --teacher-weight 0 \
        --teacher-final-weight 0 --steps 500000000
python3 -m rocketai eval runs/mein-bot/checkpoints/latest.pt --opponent teacher --games 5
python3 -m rocketai replay chaser teacher --seconds 60 --out match.json
python3 -m rocketai benchmark                                # Tempo dieses Rechners
```

**Im echten Spiel**

```bash
python3 -m rocketai play runs/mein-bot/checkpoints/latest.pt --mode psyonix --skill rookie
python3 -m rocketai play --brain teacher --mode psyonix   # Nexto spielt
```

**Der Aufbau im Repository**

| Datei / Ordner | Inhalt |
|---|---|
| `rocketai/env.py` | Simulationsumgebung (RocketSim + RLGym) |
| `rocketai/obs.py` | Was die KI sieht: Basiswerte + 12 Zusatzwerte (und ihre Beschriftungen für die Oberfläche) |
| `rocketai/rewards.py` | Belohnungen der drei Stufen |
| `rocketai/model.py` | Netz und Checkpoint-Format |
| `rocketai/ppo.py` | Der Lernschritt (inkl. Nachahmung des Lehrers) |
| `rocketai/rollout.py` | Sammelt Erfahrung in Parallel-Prozessen (inkl. Spiele gegen den Lehrer) |
| `rocketai/trainer.py` | Trainingsschleife, Autopilot, Bewertung, Logs |
| `rocketai/teacher.py` | Der Lehrer: laden, prüfen, übersetzen, spielen |
| `rocketai/opponents.py` | Einfache Bots (Balljäger, Verteidiger) + Spieler-Fabrik |
| `rocketai/match.py` | Matches, Bewertungen, Replays |
| `rocketai/live.py` | Live-Spiele für die 3D-Ansicht |
| `rocketai/server.py`, `web/` | Lokale Web-App (Tempo-Zerlegung, Fortsetzen mit Einstellungen, Auto-Fortsetzen) |
| `rocketai/rlbot_convert.py`, `rlbot_bot/` | Brücke zum echten Spiel |
| `rocketai/play.py` | Startet Matches in Rocket League (eine Bot-Datei je Auto, eindeutige Kennungen) |
| `rocketai/runtime.py` | Datesperre pro Run: verhindert zwei Trainingsprozesse auf demselben Run |
| `rocketai/benchmark.py` | Misst Schritte/s, Echtzeit-Faktor und Lernschritt |
| `tests/` | 100 gesammelte Tests (92 laufen hier durch, 8 werden übersprungen: 1× RLBot-Paket nicht installiert, 7× Lehrer nicht geladen) |
| `docs/PLAN.md` | Der Entwicklungsplan |
| `docs/WISSEN.md` | Diese Datei |

---

## 13. Häufige Fehler und Lösungen

| Symptom | Ursache / Lösung |
|---|---|
| Training zeigt „Unterbrochen“ | Der Prozess läuft nicht mehr (Absturz, Neustart, Herunterfahren). Übersicht → *Jetzt fortsetzen*, oder `python3 start.py --resume-interrupted`; es geht beim letzten `latest.pt` weiter. |
| „Neues Ziel muss über dem erreichten Stand liegen“ | Der Run ist schon so weit. Ein größeres Gesamtziel angeben (der Dialog schlägt eines vor); es zählt der Checkpoint-Stand, nicht der (evtl. ältere) Status. |
| Fortsetzen ändert nichts an Stufe/Lehrer | Über den Dialog auf der Training-Seite (oder `--stage`, `--teacher-opponent`, `--teacher-weight`) lässt sich das beim Start neu setzen; „Autopilot“ überschreibt die Stufe später trotzdem. |
| Tempo bricht ein, wenn der Lernschritt läuft | Das ist normal: Sammeln und Lernen laufen nacheinander, während des Lernens stehen die Simulationsprozesse. Die Karte „Tempo-Zerlegung“ zeigt den Anteil; hilft: `torch_threads`, Grafikkarte, weniger Epochen. |
| „Der Lehrer konnte nicht geladen werden“ | Kein Internet beim ersten Start. Einmal online gehen (`python3 -m rocketai teacher`), danach läuft alles offline. |
| „hat sich bei GitHub geändert … wird nicht geladen“ | Die Prüfsumme passt nicht. Das ist Absicht (Sicherheit). RocketAI aktualisieren. |
| Training läuft nicht los, „simulation process exited unexpectedly“ | Zu wenig Arbeitsspeicher oder ein doppelt gestartetes Training mit gleichem Namen. Prozesse im Formular reduzieren. |
| Training startet, aber Ballkontakte bleiben bei 0 | Belohnungsstufe 1 nötig, bzw. Lernrate zu hoch. Mit der Vorlage „Autopilot“ starten. |
| Sehr langsam (unter 500 Schritte/s) | Prozente im Taskmanager prüfen: andere Programme, oder `Simulations-Prozesse` zu hoch eingestellt. |
| Im echten Spiel trifft die KI nichts | Falsche Stufe: Erst gegen `Beginner` testen. Außerdem muss das Training mindestens Ballkontakt-Niveau erreicht haben. |
| RLBot findet Rocket League nicht | `python3 -m rocketai doctor` ausführen; im Spiel „ohne Easy Anti-Cheat“ starten. |
| Screenshots/3D bleibt schwarz | WebGL im Browser deaktiviert; auf 2D umschalten. |
| Live-Ansicht zeigt „kein Checkpoint“ | Training braucht mindestens einen Checkpoint (`Checkpoint alle`). |
| `NameError: __annotations__` in `rlgym/.../physics_object.py` | Python 3.14 ist mit der verwendeten RLGym-Version inkompatibel. Python 3.12 installieren, die alte `.venv` löschen und das Setup erneut starten (`python3.12 install.py`; Windows: `py -3.12 install.py`). |
| `python: command not found` (Ubuntu) | Auf Linux heißt der Befehl `python3` (siehe Abschnitt 12). |
| `ModuleNotFoundError: rocketai` | Im falschen Ordner oder ohne `.venv`: `python3 install.py`, dann `python3 start.py`. |

---

## 14. Glossar

| Begriff | Bedeutung |
|---|---|
| **RLGym** | Python-Bibliothek, in der Bots trainiert werden (Beobachtung, Aktionen, Belohnung) |
| **RocketSim** | Exakte Physik-Nachbildung von Rocket League, ohne Grafik, sehr schnell |
| **RLBot** | Offizielle Schnittstelle, um Bots im echten Spiel fahren zu lassen (offline) |
| **Nexto** | Der „Lehrer“: Community-Bot auf Grand-Champion-Niveau |
| **PPO** | Lernverfahren; verbessert Entscheidungen vorsichtig in die Richtung, die sich gelohnt hat |
| **Checkpoint** | Gespeicherter Trainingsstand (`.pt`) |
| **Reward** | Belohnung: die Zahl, die die KI maximieren will |
| **Episode** | Ein Spieldurchgang von Kickoff (oder Zufallsstart) bis Tor/Zeitlimit |
| **Curriculum** | Lernen in Stufen; hier: erst Ball treffen, dann Tore, dann ganzes Spiel |
| **Autopilot** | Schaltet die Stufe automatisch weiter |
| **Gegner-Pool** | Frühere eigene Versionen als Trainingsgegner |
| **Nachahmung / Destillation** | Die KI ahmt die Entscheidungen eines stärkeren Netzes nach |
| **Kreuzentropie / Abweichung** | Maß, wie weit die eigene Wahrscheinlichkeitsverteilung von der des Lehrers entfernt ist |
| **Siegquote** | Anteil gewonnener Spiele (unentschieden zählt halb) |
| **Checkpoint-Marke** | Senkrechte Linie im Diagramm: Hier wurde ein Stand gespeichert |

---

## 15. Regeln, die nie gebrochen werden

1. **Nur offline.** Bots laufen ausschließlich in lokalen Matches, die RLBot
   startet (Spiel ohne Easy Anti-Cheat). Kein Online, kein Ranked, keine
   privaten Online-Matches, keine Turniere mit echten Spielern.
2. **Keine Automation für Online-Spiele.** Weder in Rocket League noch in
   anderen Spielen. Auch keine Werkzeuge, die das ermöglichen könnten.
3. **Der Lehrer wird nie mitgeliefert.** Seine Dateien werden geprüft
   heruntergeladen und bleiben lokal; die Lizenz (GPL) wird respektiert.
4. **Ehrliche Zahlen.** Wenn etwas nicht funktioniert, steht es hier und in den
   Antworten — nicht in der Werbung.
