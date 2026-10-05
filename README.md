# Control Center

Ein lokales Control Center für eine Bibliothek digitaler Projekte und Ideen. Du kannst einen vorhandenen Desktop-Ordner mit Projektordnern verbinden; ein Projekt öffnet dann seinen echten Inhalt in einem neuen Browser-Tab. Dateien werden nicht kopiert oder verändert.

## Starten

Voraussetzung: **Python 3.9 oder neuer**; es werden keine externen Python-Pakete benötigt.

- **Windows:** `start.cmd` doppelklicken.
- **Ubuntu / WSL:** im Projektordner `python3 start.py` ausführen.

Standardadresse: <http://127.0.0.1:8765>. Ist der Port belegt, sucht sich Control Center automatisch den nächsten freien Port und zeigt ihn an. Mit `Ctrl+C` wird der Server beendet.

Weitere Startoptionen:

```bash
python3 start.py --port 9000          # anderen Port benutzen
python3 start.py --no-browser         # Browser nicht automatisch öffnen
python3 start.py --workspace ~/Code   # Projektordner direkt festlegen
python3 start.py --data-dir ~/.cc     # anderen Speicherort für die Bibliothek
python3 start.py --version            # Version anzeigen
```

## Projekte aus deinem Desktop-Ordner öffnen

1. Starte Control Center auf dem PC, auf dem der Projektordner liegt.
2. Klicke in der Bibliothek auf **Ordner wählen** und wähle den Hauptordner, der deine einzelnen Projektordner enthält. Alternativ kannst du den Pfad eintragen.
3. **Verbinden & einlesen** verknüpft passende Einträge und nimmt übrige vorhandene Unterordner in die Bibliothek auf. Es werden keine Projektdateien erstellt, kopiert oder verschoben.
4. Klicke bei einem verbundenen Projekt auf **Projekt öffnen**. Enthält sein Ordner eine `index.html`, wird diese geladen; andernfalls erscheint eine Dateiansicht, in der du die Ordnerstruktur durchgehen kannst.

Der eingebaute Ordnerbrowser funktioniert auf allen Systemen; ist ein Systemdialog verfügbar (Windows/macOS/mit tkinter), wird dieser zuerst benutzt.

## Bibliothek bedienen

| Funktion | So geht es |
|---|---|
| **Listenansicht** | Umschalter oben rechts — zeigt Status, Bereich, Ordner, letzte Schritte und „zuletzt geöffnet“ pro Zeile |
| **Sortieren** | Auswahl neben den Filtern: zuletzt geändert, Name, zuletzt geöffnet, neueste zuerst, Status |
| **Filtern** | Alle · Projekte · Ideen · Favoriten · Ordner fehlt |
| **Suchen** | Oben oder mit `/`; durchsucht Titel, Beschreibung, Bereich, Schlagworte, **Notizen und nächste Schritte** |
| **Favoriten** | Stern auf der Karte oder in der Zeile |
| **Bearbeiten** | Stift-Symbol: Notizen, nächste Schritte, Status, Bereich, Schlagworte, Symbol, Farbe, Favorit |
| **Ordner** | Im Bearbeiten-Dialog: Ordner wählen, Ordner anlegen, im Dateimanager öffnen, Verknüpfung lösen |
| **Löschen** | Mülleimer-Symbol — löscht **nur den Bibliothekseintrag**, dein Ordner auf der Festplatte bleibt unberührt |
| **Sicherungen** | **Sicherungen**-Knopf: jetzt sichern, alte Stände anzeigen und wiederherstellen |

Tastenkürzel: `Ctrl/⌘ + K` Schnellaktionen, `/` Suche, `Esc` Dialog schließen oder Suche leeren, `↑`/`↓` und `↵` in der Palette.

Bibliotheksdaten liegen lokal in `~/.control_center/projects.json`. Der verbundene Hauptordner wird in `~/.control_center/settings.json` gespeichert. Vor einem Import wird die aktuelle Bibliothek automatisch gesichert; die letzten zehn Sicherungen bleiben erhalten.

## Was der Scan kann

- Versteckte Ordner (`.git`, `.cache`, …) und Build-/Abhängigkeitsordner (`node_modules`, `venv`, `dist`, …) werden übersprungen.
- Projekte, die in einem Gruppierungsordner eine Ebene tiefer liegen, werden gefunden, ohne die Gruppe selbst als Projekt aufzunehmen (Gruppe ohne erkennbare Projekte bleibt ein Eintrag).
- **Umbenannte oder verschobene Ordner werden wiedererkannt** — über den Pfad, einen Fingerabdruck des Ordnerinhalts und als letzte Stufe über ähnliche Namen („Alpha“ → „Alpha2“). Es entstehen keine Duplikate, und es werden keine Dateien im Projektordner angelegt. Wird ein Ordner ganz anders benannt, bleibt der Eintrag erhalten und lässt sich im Bearbeiten-Dialog neu verknüpfen.
- Namen ohne lateinische Buchstaben (z. B. japanisch, kyrillisch) bleiben unterscheidbar.
- Ein Workspace auf Laufwerksebene wird abgelehnt, der Benutzerordner gibt eine Warnung.

## Handy im selben WLAN

Die Arena-Vorschau läuft in einer Sandbox und nicht auf deinem PC. Verwende dafür den von Arena angezeigten Live-Preview-Link — `localhost` auf dem Handy zeigt auf das Handy selbst.

Für eine Lesevorschau von einem Gerät im gleichen vertrauenswürdigen WLAN kann der Server auf dem PC so gestartet werden:

```bash
python3 start.py --host 0.0.0.0 --port 8765 --no-browser
```

Öffne danach auf dem Handy `http://<LAN-IP-deines-PCs>:8765`. Netzwerkgeräte erhalten eine **Leseansicht**; Erstellen, Ändern und Dateiaktionen bleiben auf dem Server-PC. Den Server nicht ungeschützt ins öffentliche Internet stellen. Bei WSL können zusätzliche Firewall- oder Netzwerkeinstellungen erforderlich sein.

## Probleme lösen

| Problem | Lösung |
|---|---|
| „Adresse wird bereits verwendet“ | Control Center weicht automatisch aus — die tatsächliche Adresse steht in der Konsole. Zwei Instanzen auf dieselbe Bibliothek werden mit einem Hinweis gemeldet. |
| „Ordner wählen“ meldet „System-Dateiauswahl nicht verfügbar“ | Kein Problem: danach öffnet sich automatisch der eingebaute Ordnerbrowser. |
| Ein Projekt zeigt „Ordner fehlt“ | **Verbinden & einlesen** erneut ausführen, oder im Bearbeiten-Dialog einen Ordner zuweisen. |
| Dateien eines Projekts lassen sich nicht öffnen | Projektdateien werden nur auf dem PC ausgeliefert, auf dem sie liegen (loopback-only). |
| Python fehlt (Windows) | `start.cmd` meldet das und bleibt offen: Python von <https://www.python.org/downloads/> installieren. |

## Tests

```bash
python3 -m unittest discover -s tests
```

Die Tests starten einen echten Server auf einem freien Port mit temporärem Datenverzeichnis — es werden keine Python-Pakete benötigt.

## Dokumentation

Siehe [project.md](project.md) für Zweck, aktuellen Umfang, Architektur, Datenschutz und Roadmap.
