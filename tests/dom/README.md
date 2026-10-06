# DOM-/Szenen-Tests (optional)

Die Python-Suite (`python3 -m unittest discover -s tests`) deckt Backend, API und
Szenen-Payloads ab. Da 3D und Diagramme aber im Browser laufen, prüfen diese
Node-Skripte zusätzlich das echte Frontend – headless mit **jsdom**, ohne Browser
oder GPU:

| Datei | Prüft |
| --- | --- |
| `smoke.mjs` | Lädt `web/index.html`, importiert `web/js/app.js` gegen einen laufenden Server, klickt alle acht Panels durch und prüft DOM-Verträge (Viewport, Buttons, Canvas-Charts, Slider). |
| `scene.mjs` | Baut alle sechs Karten sowie eine Custom-Map mit dem echten `three.module.min.js`, speist synthetische Arena-Frames ein (Agenten, Trails, Label) und prüft Geometrie, Instancing und `dispose()`. |

## Voraussetzungen

```bash
cd tests/dom
npm install            # jsdom (nur hier, nicht Teil der Python-Pakete)
```

`scene.mjs` und `smoke.mjs` finden das Three.js-Bundle in `web/vendor/`. Fehlt es,
lädt `install.py` es nach (`--force-assets` erneut laden).

## Ausführen

```bash
# Terminal 1 – Server (im Projektwurzelverzeichnis)
python3 start.py --no-browser

# Terminal 2
cd tests/dom
node smoke.mjs                  # erwartet http://127.0.0.1:8501
BASE_URL=http://127.0.0.1:8600 node smoke.mjs   # anderer Port
node scene.mjs                  # braucht keinen Server
```

Beide Skripte geben `PASS`/`FAIL` pro Prüfung aus, listen gefundene Probleme auf
und enden mit `ERGEBNIS: OK` bzw. einer Fehlerzahl (Exit-Code 1 bei Fehlern).

## Grenzen von jsdom

* WebGL fehlt: `scene.mjs` injiziert einen Renderer-Stub, `smoke.mjs` akzeptiert
  die Fehlermeldung „3D-Renderer Fehler" als erwartetes Ergebnis und notiert sie
  unter „Notizen".
* `canvas.getContext('2d')` liefert ohne das native `canvas`-Paket `null`; die
  Diagramme und Agenten-Labels müssen das überleben (sie fallen dann still aus).
* Maus-/Tastaturinteraktionen der 3D-Ansicht werden nicht simuliert, nur die
  Aufrufpfade (`setStatic`, `setFrame`, `setOverlay`, `dispose`).
