# Control Center

Ein lokales Control Center für eine Bibliothek digitaler Projekte und Ideen. Du kannst einen vorhandenen Desktop-Ordner mit Projektordnern verbinden; ein Projekt öffnet dann seinen echten Inhalt in einem neuen Browser-Tab. Dateien werden nicht kopiert oder verändert.

## Starten

Voraussetzung: Python 3; es werden keine externen Python-Pakete benötigt.

- **Windows:** `start.cmd` doppelklicken.
- **Ubuntu / WSL:** im Projektordner `python3 start.py` ausführen.

Standardadresse: <http://127.0.0.1:8765>. Mit `Ctrl+C` wird der Server beendet.

## Projekte aus deinem Desktop-Ordner öffnen

1. Starte Control Center auf dem PC, auf dem der Projektordner liegt.
2. Klicke in der Bibliothek auf **Ordner wählen** und wähle den Hauptordner, der deine einzelnen Projektordner enthält. Alternativ kannst du den Pfad eintragen.
3. **Verbinden & einlesen** verknüpft passende Einträge und nimmt übrige vorhandene Unterordner in die Bibliothek auf. Es werden keine Projektdateien erstellt, kopiert oder verschoben.
4. Klicke bei einem verbundenen Projekt auf **Projekt öffnen**. Enthält sein Ordner eine `index.html`, wird diese geladen; andernfalls erscheint eine Dateiansicht, in der du die Ordnerstruktur durchgehen kannst.

Bibliotheksdaten liegen lokal in `~/.control_center/projects.json`. Der verbundene Hauptordner wird in `~/.control_center/settings.json` gespeichert. Mit `Ctrl+K` kannst du die Schnellaktionen-Palette für Suche, Ordnerverbindung, JSON-Export und -Import öffnen. Vor einem Import wird die aktuelle Bibliothek automatisch gesichert; die letzten zehn Sicherungen bleiben erhalten.

Die Arena-Vorschau läuft in einer Sandbox und hat keinen Zugriff auf Dateien auf deinem PC. Zum Verbinden und Öffnen des Desktop-Ordners musst du Control Center lokal auf diesem PC starten.

## Handy im selben WLAN

Die Arena-Vorschau läuft in einer Sandbox und nicht auf deinem PC. Verwende dafür den von Arena angezeigten Live-Preview-Link — `localhost` auf dem Handy zeigt auf das Handy selbst.

Für eine Lesevorschau von einem Gerät im gleichen vertrauenswürdigen WLAN kann der Server auf dem PC so gestartet werden:

```bash
python3 start.py --host 0.0.0.0 --port 8765 --no-browser
```

Öffne danach auf dem Handy `http://<LAN-IP-deines-PCs>:8765`. Netzwerkgeräte erhalten eine **Leseansicht**; Erstellen, Ändern und Dateiaktionen bleiben auf dem Server-PC. Den Server nicht ungeschützt ins öffentliche Internet stellen. Bei WSL können zusätzliche Firewall- oder Netzwerkeinstellungen erforderlich sein.

## Dokumentation

Siehe [project.md](project.md) für Zweck, aktuellen Umfang, Architektur, Datenschutz und Roadmap.
