# Control Center — Projektbeschreibung

**Technischer Projektname:** `Control_Center`
**Anzeigename:** Control Center
**Stand:** 5. Oktober 2026
**Typ:** lokales, modular erweiterbares Projekt-Hub
**Phase:** GUI plus lokale Projektbibliothek und Arbeitsordner-Funktionen

## Kurzfassung

Control Center ist eine persönliche Bibliothek für digitale Projekte, Spielideen und Experimente. Es gibt einen zentralen Ort, an dem der Nutzer Ideen sammeln, den Stand verfolgen und später einzelne Projekte weiterentwickeln kann.

Das Control Center entstand aus der sauberen grünen RocketAI-Oberfläche. Die Oberfläche wurde als Basis übernommen; RocketAI-Training, RocketSim und RLBot sind nicht mehr Teil des aktiven Control-Center-Arbeitsverzeichnisses.

## Warum es das Projekt gibt

Control Center bündelt vorhandene Projektordner und Ideen an einem einzigen Ort:

- Den Desktop-Hauptordner mit den vorhandenen Projektordnern verbinden.
- Projekte und Ideen gemeinsam in der Bibliothek finden.
- Beim Öffnen direkt den echten Projektinhalt statt einer Notiz-Detailseite sehen.
- Die Bibliothek sichern und auf einem anderen Computer wiederherstellen.
- Die gemeinsame Oberfläche modular erweitern, ohne automatisch Projektdateien anzulegen.

## Leitbild

> Eine übersichtliche, lokale Projektzentrale, in der digitale Ideen wachsen können, ohne dass jedes Projekt seine eigene Startseite und Navigation braucht.

Der Fokus liegt auf **digitalen Spielen, Simulationen und Softwareprojekten**. Das Control Center soll keine physischen Geräte oder realen Abläufe steuern.

## Grundprinzipien

1. **Lokal zuerst:** keine Konten, Cloud oder automatische Datenübertragung.
2. **Modular:** gemeinsames Control Center, getrennte Arbeitsbereiche je Projekt.
3. **Schrittweise:** erst Bibliothek und Projektverwaltung, dann nur die konkret gewünschten Projektfunktionen.
4. **Ehrliche Zustände:** eine Karte zeigt nur dann ein startbares Projekt, wenn tatsächlich ein Projektordner und eine Startintegration existieren.
5. **Sichere Dateizugriffe:** Projektordner werden nur ausdrücklich erstellt; Schreib- und Dateiaktionen sind auf dem Rechner mit dem Server erlaubt.
6. **Keine verbotene Spielautomation:** zukünftige Spielintegrationen müssen die Regeln des jeweiligen Spiels und Modus einhalten.

## Aktueller Funktionsumfang

### Bibliothek

- Eine zentrale Liste mit Projekten und Ideen; Suche, Ideen-/Projektfilter und Raster- oder Listenansicht.
- Den vorhandenen Hauptordner auswählen und direkte Unterordner automatisch mit Bibliothekseinträgen verknüpfen.
- Noch nicht gelistete Unterordner werden erst durch diese ausdrückliche Einlese-Aktion als Einträge übernommen; ihre Dateien werden nicht verändert.
- `Ctrl+K` öffnet die Schnellaktions-Palette.

### Projekt öffnen

- Ein Projekt öffnet sich in einem neuen Browser-Tab.
- Enthält der Projektordner eine `index.html`, wird das eigentliche Projekt geladen.
- Sonst öffnet sich eine Dateiansicht, in der Unterordner und Dateien des Projekts durchsucht und geöffnet werden können.
- Die Metadatenansicht mit Notizen ist nicht der Projekt-Öffnen-Ablauf.

### Sicherung

- Bibliothek als JSON-Datei exportieren.
- JSON-Sicherung importieren; der Import ersetzt die Bibliotheks-Metadaten nach Bestätigung.
- Vor dem Import wird eine datierte lokale Sicherung der aktuellen Bibliothek erstellt; maximal zehn Sicherungen bleiben erhalten.
- Ordnerpfade werden absichtlich nicht in portable Sicherungen übernommen; nach einem Import kann der Hauptordner erneut eingelesen werden.
- Import löscht keine bereits bestehenden Ordner auf der Festplatte.

### Responsive Weboberfläche

Die GUI passt sich an kleinere Bildschirme an. Sie ist eine Webseite, keine installierbare Handy-App. Im LAN erhalten andere Geräte eine read-only Ansicht; Bearbeitungen und Dateiaktionen bleiben auf dem Server-Rechner.

## Daten und Speicherorte

- Bibliotheksdatei: `~/.control_center/projects.json`.
- Der anfängliche Standardordner ist `~/Control_Center_Projects/`; in der GUI kann stattdessen der vorhandene Projekt-Hauptordner auf dem Desktop ausgewählt werden.
- Die ausgewählte Basis wird in `~/.control_center/settings.json` gespeichert und beim nächsten Start wiederverwendet.
- Mit `--data-dir` und `--workspace` können Speicherorte beim Start angepasst werden.
- Die Bibliothek speichert Projektnamen, Status und optionale Beschreibungen; der eigentliche Code und die Projektdateien bleiben in den jeweiligen Ordnern.
- Eine leere Installation startet ohne Demo-Projekte. Vorhandene Ordner werden erst nach der ausdrücklichen Aktion „Verbinden & einlesen“ aufgenommen.

Das Control Center selbst liefert die Projektdateien über einen zusätzlichen loopback-only Webserver auf einem eigenen Port aus. Netzwerkgeräte und Arenas Vorschau können diesen Port und den Desktop nicht erreichen; sie erhalten nur die read-only Bibliotheksansicht.

## Sicherheit und Grenzen des LAN-Zugriffs

- Standardmäßig bindet der Server nur an `127.0.0.1`.
- `--host 0.0.0.0` ist ein bewusstes Opt-in für Vorschau im vertrauenswürdigen LAN.
- Netzwerkgeräte können nur die freigegebene read-only Bibliotheksübersicht abrufen.
- Projektdateien werden von separaten loopback-only Servern auf zufälligen Ports ausgeliefert und nicht an Geräte im LAN oder an Arenas Vorschau freigegeben.
- Es gibt keinen Schutz für öffentliche Internet-Exposition; den lokalen Server nicht öffentlich freigeben.

Die Arena-Vorschau läuft in einer Sandbox und nicht auf dem Nutzer-PC. Ein Arena-Live-Preview-Link kann deshalb weder den Desktop-Ordner wählen noch auf die dortigen Projektdateien zugreifen.

## Technische Struktur

```text
index.html    Seitenstruktur und Einstiegspunkt
styles.css    dunkles Layout, limettengrüner Akzent, responsive GUI
app.js        Bibliothek, Ordnerverbindung, Projektstart und JSON-Transfer
start.py      Python-Standardbibliothek: lokale API und loopback-only Projektserver
start.cmd     Starthelfer für Windows
README.md     Kurzstart und Desktop-Ordnerverbindung
project.md    Projektvision, Funktionsumfang und Roadmap
```

Es werden keine externen Python-Pakete oder Frontend-CDNs benötigt. Die lokale JSON-API ist Bestandteil des kleinen Python-Startservers.

## Projekt-Datensatz

Ein Bibliothekseintrag verwendet Felder wie:

- `id`, `title`, `description`, `category`
- `status`, `statusLabel`, `tags`, `favorite`
- `notes`, `nextSteps`
- `icon`, `color`
- `createdAt`, `updatedAt`, `lastOpened`
- `folderPath` für den lokalen Arbeitsordner, sofern vorhanden

Beim JSON-Export wird `folderPath` ausgelassen beziehungsweise zurückgesetzt, damit eine Sicherung nicht so tut, als lägen die Dateien auch auf einem anderen Computer am gleichen Ort.

## Vorschau und Start

Voraussetzung ist Python 3.

- Windows: `start.cmd` starten.
- Ubuntu/WSL: `python3 start.py` im Projektordner ausführen.
- Standardadresse: `http://127.0.0.1:8765`.
- Optionales LAN-Read-only: `python3 start.py --host 0.0.0.0 --port 8765 --no-browser`.

Auf dem Handy im selben WLAN wird die LAN-IP des Rechners mit Port `8765` geöffnet. Eine Firewall oder WSL-Netzwerkkonfiguration kann zusätzlichen Zugriff erfordern.

## Roadmap

### Nächste Ausbaustufe — Alltagstauglichkeit

- Desktop-Ordnerauswahl und Bibliotheksscan auf Windows, macOS und Ubuntu/WSL prüfen.
- Sonderfälle bei Namen, doppelten Ordnern und großen Projektdateien testen.
- Ergänzen, wie ein Projekt ohne `index.html` seine passende Startseite mitteilt.

### Später — konkrete Projektintegration

- Projektabhängige Funktionen nur für ein konkret ausgewähltes Projekt ergänzen.
- Bei ausführbaren Projekten zuerst eine sichere und eindeutige Startmethode festlegen.
- Bei Spielen vorab prüfen, ob der gewünschte Modus und Bot-Einsatz erlaubt sind.
- Keine allgemeine Launcher-, Cloud- oder Plugin-Plattform ohne konkreten Bedarf bauen.

## Noch nicht enthalten

- Cloud- oder Geräte-Synchronisierung.
- Benutzerkonten oder Mehrbenutzer-Berechtigungen.
- Öffnen beliebiger externer Programme.
- Projektordner umbenennen oder löschen.
- Ein vollwertiges Plugin-System.
- RocketAI-Training, PPO, RocketSim oder RLBot.
- Eine installierbare Handy-App.
