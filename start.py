#!/usr/bin/env python3
"""Run the local Control Center UI and its small local project-library API."""

from __future__ import annotations

import argparse
import contextlib
import html
import ipaddress
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
import webbrowser
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator
from urllib.parse import parse_qs, quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path.home() / ".control_center"
DEFAULT_WORKSPACE = Path.home() / "Control_Center_Projects"
MAX_BODY_BYTES = 10_000_000
PROJECT_CONTENT_LOCK = threading.Lock()
LIBRARY_LOCK = threading.RLock()
CONTENT_SERVER_IDLE_SECONDS = 30 * 60
MAX_SCAN_DIRECTORIES = 300

STATUS_LABELS = {
    "idea": "Idee",
    "project": "Projekt",
    "prototype": "Prototyp",
    "active": "In Arbeit",
    "paused": "Pausiert",
    "archived": "Archiviert",
    "reference": "GUI-Vorlage",
}

PROJECT_ICONS = {"rocket", "arena", "dungeon", "hub", "gamepad", "spark", "plus"}
PROJECT_COLORS = {"lime", "blue", "violet", "orange"}

# Folders that are never treated as projects when the workspace is scanned.
IGNORED_DIRECTORY_NAMES = {
    ".git", ".svn", ".hg", ".cache", ".config", ".local", ".venv", "venv", "env",
    "__pycache__", "node_modules", "site-packages", "vendor", "dist", "build",
    "out", "target", "bin", "obj", ".next", ".nuxt", ".turbo", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox", ".gradle", ".idea", ".vscode",
    "system volume information", "$recycle.bin", "recovery", "windows",
    "program files", "program files (x86)", "programdata", "appdata",
    "library", "applications", "snap", ".snapshots", "steamapps",
}

# Files that mark a folder as a real project (used by the nested scan).
PROJECT_MARKER_FILES = {
    "index.html", "package.json", "pyproject.toml", "requirements.txt",
    "cargo.toml", "pom.xml", "go.mod", "gemfile", "composer.json",
    "main.py", "main.js", "main.ts", "main.cpp", "app.py", "app.js",
    "readme.md", "readme.txt", "readme", "setup.py", "makefile",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-").lower()
    if len(slug) >= 3:
        return slug[:56].strip("-") or "projekt"
    # Names without latin letters would all collapse into the same fallback, so
    # they keep their original characters instead (Japanese, Cyrillic, ...).
    unicode_text = "".join(
        char.lower() if char.isalnum() else "-" for char in unicodedata.normalize("NFC", value)
    )
    slug = re.sub(r"-+", "-", unicode_text).strip("-")
    return slug[:56].strip("-") or "projekt"


def folder_signature(folder: Path) -> str:
    """Stable fingerprint of a folder's direct content.

    It is stored next to the folder path so a renamed or moved folder can be
    recognised again without writing anything into the user's project folder.
    """
    try:
        entries = sorted((child.name, child.is_dir()) for child in folder.iterdir())
    except OSError:
        return ""
    if not entries:
        # Empty folders all look the same, so they are matched by name only.
        return ""
    digest = sha256()
    for name, is_directory in entries[:400]:
        digest.update(f"{'d' if is_directory else 'f'}:{name}\n".encode("utf-8"))
    return digest.hexdigest()[:16]


def clean_text(value: object, limit: int, default: str = "") -> str:
    if not isinstance(value, str):
        return default
    return value.strip()[:limit]


def normalize_project(source: object, *, keep_folder: bool = True) -> dict[str, object] | None:
    if not isinstance(source, dict):
        return None
    title = clean_text(source.get("title"), 80)
    if not title:
        return None
    raw_status = source.get("status")
    status = raw_status if isinstance(raw_status, str) and raw_status in STATUS_LABELS else "idea"
    raw_tags = source.get("tags", [])
    tags = [clean_text(tag, 24) for tag in raw_tags[:10] if isinstance(tag, str)] if isinstance(raw_tags, list) else []
    raw_steps = source.get("nextSteps", [])
    steps = [clean_text(step, 180) for step in raw_steps[:30] if isinstance(step, str) and step.strip()] if isinstance(raw_steps, list) else []
    project_id = clean_text(source.get("id"), 64)
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", project_id):
        project_id = uuid.uuid4().hex
    raw_icon = source.get("icon")
    icon = raw_icon if isinstance(raw_icon, str) and raw_icon in PROJECT_ICONS else "hub"
    raw_color = source.get("color")
    color = raw_color if isinstance(raw_color, str) and raw_color in PROJECT_COLORS else "lime"
    folder_path = source.get("folderPath") if keep_folder and isinstance(source.get("folderPath"), str) else None
    raw_signature = source.get("folderSignature")
    signature = raw_signature if isinstance(raw_signature, str) and re.fullmatch(r"[a-f0-9]{16}", raw_signature) else None
    return {
        "id": project_id,
        "title": title,
        "description": clean_text(source.get("description"), 500, "Noch keine Beschreibung."),
        "category": clean_text(source.get("category"), 50, "Allgemein"),
        "status": status,
        "statusLabel": STATUS_LABELS[status],
        "tags": tags,
        "icon": icon,
        "color": color,
        "favorite": bool(source.get("favorite", False)),
        "updated": clean_text(source.get("updated"), 80, "Gerade hinzugefügt"),
        "notes": clean_text(source.get("notes"), 20_000),
        "nextSteps": steps,
        "folderPath": folder_path,
        "folderSignature": signature,
        "lastOpened": source.get("lastOpened") if isinstance(source.get("lastOpened"), (int, float)) else None,
        "isDemo": bool(source.get("isDemo", False)),
        "createdAt": clean_text(source.get("createdAt"), 40, now_iso()),
        "updatedAt": clean_text(source.get("updatedAt"), 40, now_iso()),
    }


class ProjectContentHandler(SimpleHTTPRequestHandler):
    """Read-only local server bound to one project's actual folder."""

    project_root: Path | None = None
    project_title = "Projekt"

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        try:
            size = path.stat().st_size
            with path.open("rb") as file:
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                if content_type.startswith("text/") or content_type in {"application/javascript", "application/json", "application/xml"}:
                    content_type += "; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(size))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                shutil.copyfileobj(file, self.wfile)
        except OSError:
            self.send_error(404, "Datei nicht gefunden.")

    @staticmethod
    def _href(parts: tuple[str, ...] | list[str], *, directory: bool = False) -> str:
        path = "/" + "/".join(quote(part, safe="") for part in parts)
        if directory and not path.endswith("/"):
            path += "/"
        return path or "/"

    def _directory_page(self, root: Path, current: Path) -> None:
        relative = current.relative_to(root)
        crumbs = [f'<a href="/">{html.escape(self.project_title)}</a>']
        accumulated: list[str] = []
        for part in relative.parts:
            accumulated.append(part)
            crumbs.append(f'<span>/</span><a href="{self._href(accumulated, directory=True)}">{html.escape(part)}</a>')
        rows: list[str] = []
        if relative.parts:
            rows.append(f'<a class="file-row parent" href="{self._href(list(relative.parts[:-1]), directory=True)}"><span>↰</span><b>Übergeordneter Ordner</b><small>—</small></a>')
        try:
            entries = sorted(current.iterdir(), key=lambda item: (not item.is_dir(), item.name.casefold()))
        except OSError:
            self.send_error(403, "Ordner kann nicht gelesen werden.")
            return
        for entry in entries:
            try:
                resolved = entry.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, ValueError):
                continue
            is_directory = entry.is_dir()
            item_parts = [*relative.parts, entry.name]
            href = self._href(item_parts, directory=is_directory)
            try:
                size = "Ordner" if is_directory else f"{entry.stat().st_size:,} Byte"
            except OSError:
                size = "—"
            rows.append(f'<a class="file-row" href="{href}"><span>{"📁" if is_directory else "📄"}</span><b>{html.escape(entry.name)}</b><small>{size}</small></a>')
        title = html.escape(self.project_title)
        body = f"""<!doctype html><html lang="de"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} — Projektdateien</title><style>
:root{{color-scheme:dark;--bg:#0b0d12;--surface:#11151d;--line:#28303b;--text:#eef1f5;--muted:#99a3b2;--accent:#c6f432}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(ellipse at 50% -15%,#202a13,transparent 50%),var(--bg);color:var(--text);font:14px/1.5 system-ui,sans-serif}}main{{width:min(920px,calc(100% - 32px));margin:48px auto}}.eyebrow{{color:#9fb873;font:10px monospace;letter-spacing:.14em}}h1{{margin:7px 0 4px;font-size:clamp(24px,5vw,34px);letter-spacing:-.04em}}.subtitle{{color:var(--muted);font-size:12px}}nav{{display:flex;flex-wrap:wrap;gap:8px;margin:24px 0 12px;color:var(--muted);font-size:11px}}nav a{{color:var(--accent);text-decoration:none}}.files{{overflow:hidden;border:1px solid var(--line);border-radius:14px;background:var(--surface)}}.file-row{{min-height:49px;display:grid;grid-template-columns:26px minmax(0,1fr) 100px;align-items:center;gap:10px;padding:8px 14px;border-bottom:1px solid var(--line);color:var(--text);text-decoration:none}}.file-row:last-child{{border-bottom:0}}.file-row:hover{{background:#1a2029}}.file-row b{{overflow:hidden;font-size:12px;font-weight:550;text-overflow:ellipsis;white-space:nowrap}}.file-row small{{color:var(--muted);text-align:right;font:10px monospace}}.parent{{color:var(--accent)}}.empty{{padding:28px;color:var(--muted);text-align:center;font-size:12px}}.note{{margin-top:16px;color:#748093;font-size:10px}}@media(max-width:560px){{main{{margin:26px auto}}.file-row{{grid-template-columns:22px minmax(0,1fr) 72px;padding:8px 10px}}}}
</style><main><div class="eyebrow">PROJEKTORDNER</div><h1>{title}</h1><p class="subtitle">Der echte Inhalt aus deinem lokalen Projektordner.</p><nav>{''.join(crumbs)}</nav><section class="files">{''.join(rows) if rows else '<div class="empty">Dieser Ordner ist noch leer.</div>'}</section><p class="note">Nur Dateien innerhalb dieses Projektordners werden angezeigt. Zum Control Center zurückkehren: diesen Tab schließen.</p></main></html>"""
        self._send_bytes(200, body.encode("utf-8"), "text/html; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        setattr(self.server, "last_access", time.monotonic())
        host = urlsplit(f"//{self.headers.get('Host', '')}").hostname
        try:
            local = bool(host and (host.lower() == "localhost" or ipaddress.ip_address(host).is_loopback))
        except ValueError:
            local = False
        if not local:
            self.send_error(403, "Projektdateien sind nur lokal verfügbar.")
            return
        if self.project_root is None:
            self.send_error(404, "Projektordner nicht gefunden.")
            return
        try:
            root = self.project_root.expanduser().resolve(strict=True)
            parts = [unquote(part) for part in urlsplit(self.path).path.split("/") if part]
            if any(part in {".", ".."} or "/" in part or "\\" in part or "\x00" in part for part in parts):
                raise ValueError("Ungültiger Pfad.")
            target = root.joinpath(*parts).resolve(strict=True)
            target.relative_to(root)
        except (OSError, ValueError):
            self.send_error(404, "Projektdatei nicht gefunden.")
            return
        if target.is_dir():
            index_file = target / "index.html"
            if index_file.is_file():
                try:
                    safe_index = index_file.resolve(strict=True)
                    safe_index.relative_to(root)
                except (OSError, ValueError):
                    self.send_error(404, "Projektdatei nicht gefunden.")
                    return
                self._send_file(safe_index)
            else:
                self._directory_page(root, target)
            return
        if not target.is_file():
            self.send_error(404, "Projektdatei nicht gefunden.")
            return
        self._send_file(target)

    def log_message(self, format: str, *args: object) -> None:
        super().log_message(format, *args)


class ControlCenterHandler(SimpleHTTPRequestHandler):
    data_dir = DEFAULT_DATA_DIR
    library_path = DEFAULT_DATA_DIR / "projects.json"
    workspace_root = DEFAULT_WORKSPACE
    workspace_configured = False
    project_content_servers: dict[str, ThreadingHTTPServer] = {}

    def _local_request(self) -> bool:
        # Browser writes are enabled only when the page is opened via localhost.
        # This keeps LAN/preview-host requests read-only, including reverse proxies
        # that may make remote clients appear to originate from loopback.
        host_header = self.headers.get("Host", "")
        host = urlsplit(f"//{host_header}").hostname
        if not host:
            return False
        if host.lower() == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def _send_json(self, status: int, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message})

    def _read_json(self) -> object:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Ungültige Content-Length.") from error
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ValueError("Die Anfrage ist leer oder zu groß.")
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Ungültiges JSON.") from error

    def _write_allowed(self) -> bool:
        if not self._local_request():
            self._send_error_json(403, "Änderungen und Ordneraktionen sind nur auf dem lokalen PC erlaubt.")
            return False
        origin = self.headers.get("Origin")
        host = self.headers.get("Host")
        if origin and host:
            origin_host = urlsplit(origin).netloc.lower()
            if origin_host and origin_host != host.lower():
                self._send_error_json(403, "Anfrage mit abweichendem Origin wurde blockiert.")
                return False
        return True

    def _load_projects(self) -> list[dict[str, object]]:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.library_path.exists():
            self._save_projects([])
        try:
            value = json.loads(self.library_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            value = []
        if not isinstance(value, list):
            value = []
        projects: list[dict[str, object]] = []
        seen: set[str] = set()
        for item in value:
            project = normalize_project(item)
            if project and project["id"] not in seen:
                projects.append(project)
                seen.add(str(project["id"]))
        return projects

    def _save_projects(self, projects: list[dict[str, object]]) -> None:
        """Write the library atomically.

        Every write uses its own temporary file: two requests that save at the
        same time must never share (and delete) the same temp file.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        temp_path = self.library_path.with_name(f"{self.library_path.stem}.{uuid.uuid4().hex}.tmp")
        temp_path.write_text(json.dumps(projects, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if os.name != "nt":
            self.data_dir.chmod(0o700)
            temp_path.chmod(0o600)
        temp_path.replace(self.library_path)
        self._cleanup_temp_files()

    def _cleanup_temp_files(self) -> None:
        """Remove leftovers of interrupted writes."""
        try:
            for leftover in self.data_dir.glob(f"{self.library_path.stem}.*.tmp"):
                if leftover.is_file():
                    leftover.unlink()
        except OSError:
            pass

    def _create_backup(self) -> str:
        """Copy the current library into a dated backup file (max. 10 kept)."""
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = self.data_dir / f"projects-backup-{stamp}-{uuid.uuid4().hex[:6]}.json"
        shutil.copy2(self.library_path, backup_path)
        if os.name != "nt":
            backup_path.chmod(0o600)
        old_backups = sorted(self.data_dir.glob("projects-backup-*.json"), reverse=True)
        for old_backup in old_backups[10:]:
            old_backup.unlink(missing_ok=True)
        return backup_path.name

    def _list_backups(self) -> list[dict[str, object]]:
        backups: list[dict[str, object]] = []
        if not self.data_dir.is_dir():
            return backups
        for path in sorted(self.data_dir.glob("projects-backup-*.json"), reverse=True)[:10]:
            try:
                stat = path.stat()
                count = len(json.loads(path.read_text(encoding="utf-8")) or [])
            except (OSError, ValueError):
                count = 0
            backups.append({
                "name": path.name,
                "created": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "size": stat.st_size,
                "entries": count if isinstance(count, int) else 0,
            })
        return backups

    @contextlib.contextmanager
    def _library(self) -> "Iterator[list[dict[str, object]]]":
        """Load, change and save the library without losing concurrent writes."""
        with LIBRARY_LOCK:
            projects = self._load_projects()
            yield projects
            self._save_projects(projects)

    def _project(self, project_id: str) -> tuple[list[dict[str, object]], dict[str, object] | None]:
        projects = self._load_projects()
        for project in projects:
            if project["id"] == project_id:
                return projects, project
        return projects, None

    def _workspace_path(self) -> Path:
        return self.workspace_root.expanduser().resolve()

    def _save_workspace_path(self, raw_path: object) -> tuple[Path, list[str]]:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("Bitte wähle den Hauptordner mit deinen Projekten aus.")
        workspace = Path(raw_path.strip()).expanduser().resolve(strict=True)
        if not workspace.is_dir():
            raise ValueError("Der ausgewählte Pfad ist kein Ordner.")
        warnings: list[str] = []
        try:
            if workspace == Path(workspace.anchor) or workspace.parent == workspace:
                raise ValueError("Bitte wähle einen Projektordner und nicht ein ganzes Laufwerk.")
        except OSError:
            pass
        if workspace in {ROOT, self.data_dir}:
            raise ValueError("Dieser Ordner gehört zum Control Center selbst und ist als Projektordner nicht geeignet.")
        if workspace == Path.home():
            warnings.append("Das ist dein Benutzerordner. Ein eigener Ordner nur für Projekte bleibt übersichtlicher.")
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        config_path = self.data_dir / "settings.json"
        temp_path = config_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps({"workspace": str(workspace)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if os.name != "nt":
            temp_path.chmod(0o600)
        temp_path.replace(config_path)
        old_workspace = self._workspace_path()
        type(self).workspace_root = workspace
        type(self).workspace_configured = True
        if old_workspace != workspace:
            with PROJECT_CONTENT_LOCK:
                for server in type(self).project_content_servers.values():
                    server.shutdown()
                    server.server_close()
                type(self).project_content_servers.clear()
        return workspace, warnings

    @staticmethod
    def _is_ignored_directory(directory: Path) -> bool:
        name = directory.name.lower()
        return name.startswith(".") or name in IGNORED_DIRECTORY_NAMES

    @staticmethod
    def _looks_like_project(directory: Path) -> bool:
        try:
            names = {child.name.lower() for child in directory.iterdir()}
        except OSError:
            return False
        return bool(names & PROJECT_MARKER_FILES)

    def _candidate_directories(self, workspace: Path) -> list[Path]:
        """Direct project folders plus one extra level for grouping folders."""
        direct: list[Path] = []
        nested: list[Path] = []
        for candidate in workspace.iterdir():
            try:
                if not candidate.is_dir() or candidate.is_symlink():
                    continue
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(workspace)
            except (OSError, ValueError):
                continue
            if self._is_ignored_directory(resolved):
                continue
            if self._looks_like_project(resolved):
                direct.append(resolved)
                continue
            # Grouping folder: use the real projects inside instead of the group.
            inner: list[Path] = []
            try:
                children = sorted(resolved.iterdir(), key=lambda item: item.name.casefold())
            except OSError:
                children = []
            for child in children:
                try:
                    if not child.is_dir() or child.is_symlink():
                        continue
                    nested_resolved = child.resolve(strict=True)
                    nested_resolved.relative_to(workspace)
                except (OSError, ValueError):
                    continue
                if self._is_ignored_directory(nested_resolved):
                    continue
                if self._looks_like_project(nested_resolved):
                    inner.append(nested_resolved)
            if inner:
                nested.extend(inner)
            else:
                direct.append(resolved)
        direct.sort(key=lambda path: path.name.casefold())
        nested.sort(key=lambda path: str(path).casefold())
        for directory in nested:
            if len(direct) >= MAX_SCAN_DIRECTORIES:
                break
            direct.append(directory)
        return direct[:MAX_SCAN_DIRECTORIES]

    def _known_folder(self, project: dict[str, object]) -> Path | None:
        """The stored folder even when it lives outside the current workspace."""
        raw = project.get("folderPath")
        if not isinstance(raw, str) or not raw:
            return None
        try:
            candidate = Path(raw).expanduser().resolve(strict=True)
        except OSError:
            return None
        return candidate if candidate.is_dir() else None

    def _scan_workspace_projects(self) -> tuple[list[dict[str, object]], int, int, list[str]]:
        workspace = self._workspace_path()
        if not workspace.is_dir():
            raise ValueError("Der verbundene Hauptordner ist nicht vorhanden. Wähle bitte den Ordner auf deinem Desktop erneut aus.")
        candidates = self._candidate_directories(workspace)
        signatures: dict[Path, str] = {}
        by_signature: dict[str, list[Path]] = {}
        by_name: dict[str, list[Path]] = {}
        by_slug: dict[str, list[Path]] = {}
        for directory in candidates:
            signature = folder_signature(directory)
            signatures[directory] = signature
            if signature:
                by_signature.setdefault(signature, []).append(directory)
            by_name.setdefault(directory.name.casefold(), []).append(directory)
            by_slug.setdefault(safe_slug(directory.name), []).append(directory)

        def take(mapping: dict[str, list[Path]], key: str, used: set[Path]) -> Path | None:
            for option in mapping.get(key) or []:
                if option not in used:
                    used.add(option)
                    return option
            return None

        warnings: list[str] = []
        projects = self._load_projects()
        used: set[Path] = set()
        linked = 0
        for project in projects:
            current = self._known_folder(project)
            matched: Path | None = None
            if current is not None and current in signatures:
                matched = current
                used.add(current)
            else:
                stored_signature = project.get("folderSignature")
                if current is not None:
                    # The folder was renamed or moved: recognise it by content.
                    stored_signature = folder_signature(current) or stored_signature
                if isinstance(stored_signature, str) and stored_signature:
                    matched = take(by_signature, stored_signature, used)
                if matched is None:
                    matched = take(by_name, str(project["title"]).strip().casefold(), used)
                if matched is None:
                    matched = take(by_slug, safe_slug(str(project["title"])), used)
            if matched is not None:
                changed = project.get("folderPath") != str(matched)
                project["folderPath"] = str(matched)
                project["folderSignature"] = signatures.get(matched) or folder_signature(matched)
                project["isDemo"] = False
                if changed:
                    linked += 1
            elif current is None and project.get("folderPath"):
                # Only drop links whose folder really disappeared; folders of a
                # different workspace stay saved and come back later.
                project["folderPath"] = None
                project["folderSignature"] = None

        added = 0
        for directory in candidates:
            if directory in used:
                continue
            title = clean_text(directory.name.replace("_", " ").replace("-", " "), 80)
            if not title:
                continue
            project = normalize_project({
                "id": uuid.uuid4().hex,
                "title": title,
                "description": "",
                "category": "Projektordner",
                "status": "project",
                "tags": [],
                "icon": "hub",
                "color": "lime",
                "folderPath": str(directory),
                "folderSignature": signatures.get(directory),
                "isDemo": False,
            })
            if project:
                project["folderPath"] = str(directory)
                projects.append(project)
                used.add(directory)
                added += 1
        if len(candidates) >= MAX_SCAN_DIRECTORIES:
            warnings.append(f"Es wurden nur die ersten {MAX_SCAN_DIRECTORIES} Ordner berücksichtigt.")
        self._save_projects(projects)
        return projects, added, linked, warnings

    def _start_project_content_server(self, project: dict[str, object], folder: Path) -> int:
        project_id = str(project["id"])
        resolved_folder = folder.resolve(strict=True)
        with PROJECT_CONTENT_LOCK:
            server = type(self).project_content_servers.get(project_id)
            if server is not None and getattr(server, "project_root", None) == resolved_folder:
                return int(server.server_address[1])
            if server is not None:
                server.shutdown()
                server.server_close()
            handler = type(
                f"ProjectContent_{safe_slug(str(project['title']))}_{project_id[:8]}",
                (ProjectContentHandler,),
                {"project_root": resolved_folder, "project_title": str(project["title"])},
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            server.daemon_threads = True
            server.project_root = resolved_folder
            server.last_access = time.monotonic()
            thread = threading.Thread(target=server.serve_forever, name=f"project-content-{project_id[:8]}", daemon=True)
            thread.start()
            type(self).project_content_servers[project_id] = server
            return int(server.server_address[1])

    def _serve_project_preview_redirect(self, path: str) -> None:
        parts = [unquote(part) for part in path.split("/") if part]
        if len(parts) < 2 or parts[0] != "project" or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", parts[1]):
            self.send_error(404, "Projekt nicht gefunden.")
            return
        _, project = self._project(parts[1])
        folder = self._safe_project_folder(project) if project else None
        if not project or folder is None:
            self.send_error(404, "Diesem Eintrag ist noch kein vorhandener Projektordner zugeordnet.")
            return
        try:
            port = self._start_project_content_server(project, folder)
        except OSError as error:
            self.send_error(503, f"Projekt-Webserver konnte nicht gestartet werden: {error}")
            return
        with LIBRARY_LOCK:
            projects, stored = self._project(parts[1])
            if stored is not None:
                stored["lastOpened"] = int(datetime.now().timestamp() * 1000)
                self._save_projects(projects)
        location = f"http://127.0.0.1:{port}/"
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _create_folder(self, project: dict[str, object]) -> Path:
        workspace = self._workspace_path()
        workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
        base = safe_slug(str(project["title"]))
        candidate = workspace / base
        index = 2
        while candidate.exists():
            candidate = workspace / f"{base}-{index}"
            index += 1
        candidate.mkdir(mode=0o700, parents=False, exist_ok=False)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(workspace)
        return resolved

    def _safe_project_folder(self, project: dict[str, object]) -> Path | None:
        raw = project.get("folderPath")
        if not isinstance(raw, str) or not raw:
            return None
        workspace = self._workspace_path()
        try:
            candidate = Path(raw).expanduser().resolve(strict=True)
            candidate.relative_to(workspace)
        except (OSError, ValueError):
            return None
        return candidate if candidate.is_dir() else None

    def _open_folder(self, folder: Path) -> str | None:
        try:
            if os.name == "nt":
                os.startfile(str(folder))  # type: ignore[attr-defined]
                return None
            if sys.platform == "darwin":
                command = ["open", str(folder)]
            elif os.environ.get("WSL_DISTRO_NAME"):
                opener = shutil.which("wslview")
                if opener:
                    command = [opener, str(folder)]
                else:
                    explorer = shutil.which("explorer.exe")
                    wslpath = shutil.which("wslpath")
                    if not explorer or not wslpath:
                        return "WSL-Datei-Manager nicht gefunden (wslview oder explorer.exe/wslpath)."
                    windows_path = subprocess.check_output([wslpath, "-w", str(folder)], text=True).strip()
                    command = [explorer, windows_path]
            else:
                opener = shutil.which("xdg-open")
                if not opener:
                    return "Kein Datei-Manager-Befehl gefunden (xdg-open)."
                command = [opener, str(folder)]
            subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            return None
        except (OSError, AttributeError) as error:
            return f"Ordner konnte nicht geöffnet werden: {error}"

    def _remote_project(self, project: dict[str, object]) -> dict[str, object]:
        hidden = {"notes", "nextSteps", "folderPath", "folderAvailable", "folderSignature"}
        public = {key: value for key, value in project.items() if key not in hidden}
        public["hasFolder"] = self._safe_project_folder(project) is not None
        return public

    def _folder_inside_workspace(self, raw_path: object) -> Path | None:
        workspace = self._workspace_path()
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None
        try:
            candidate = Path(raw_path.strip()).expanduser().resolve(strict=True)
            candidate.relative_to(workspace)
        except (OSError, ValueError):
            return None
        return candidate if candidate.is_dir() else None

    def _cross_site_request(self) -> bool:
        """True for requests that a foreign website triggered (DNS rebinding)."""
        return self.headers.get("Sec-Fetch-Site") in {"cross-site", "same-site"}

    def _send_browse_result(self, query: str) -> None:
        """List subfolders so the UI can pick a folder without a native dialog."""
        params = parse_qs(query)
        raw_path = (params.get("path") or [""])[0]
        try:
            current = Path(raw_path).expanduser().resolve(strict=True) if raw_path.strip() else Path.home().resolve()
        except OSError:
            self._send_error_json(400, "Dieser Ordner existiert nicht.")
            return
        if not current.is_dir():
            self._send_error_json(400, "Dieser Pfad ist kein Ordner.")
            return
        entries: list[dict[str, object]] = []
        try:
            children = sorted(current.iterdir(), key=lambda item: item.name.casefold())
        except OSError as error:
            self._send_error_json(403, f"Ordner kann nicht gelesen werden: {error}")
            return
        for child in children:
            try:
                if not child.is_dir() or child.is_symlink():
                    continue
                resolved = child.resolve(strict=True)
            except OSError:
                continue
            if self._is_ignored_directory(resolved):
                continue
            entries.append({"name": child.name, "path": str(resolved)})
        shortcut_candidates = (
            ("Benutzerordner", Path.home()),
            ("Desktop", Path.home() / "Desktop"),
            ("Desktop (OneDrive)", Path.home() / "OneDrive" / "Desktop"),
            ("Dokumente", Path.home() / "Documents"),
            ("Aktueller Projektordner", self._workspace_path() if self.workspace_configured else None),
        )
        shortcuts = [
            {"label": label, "path": str(path.resolve())}
            for label, path in shortcut_candidates
            if path is not None and path.is_dir()
        ]
        parent = current.parent if current.parent != current else None
        self._send_json(200, {
            "path": str(current),
            "parent": str(parent) if parent else None,
            "directories": entries[:400],
            "shortcuts": shortcuts,
        })

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler method name
        parsed = urlsplit(self.path)
        if parsed.path.startswith("/api/") and self._cross_site_request():
            self._send_error_json(403, "Anfrage von einer fremden Seite wurde blockiert.")
            return
        if parsed.path == "/api/health":
            local = self._local_request()
            self._send_json(200, {
                "ok": True,
                "readOnly": not local,
                "workspace": str(self._workspace_path()) if local else None,
                "workspaceName": self._workspace_path().name,
                "workspaceConfigured": bool(self.workspace_configured) if local else None,
                "libraryExists": self.library_path.exists() if local else None,
            })
            return
        if parsed.path == "/api/projects":
            projects = self._load_projects()
            if not self._local_request():
                projects = [self._remote_project(project) for project in projects]
            else:
                for project in projects:
                    folder = self._safe_project_folder(project)
                    project["folderPath"] = str(folder) if folder else None
                    outside = None if folder else self._known_folder(project)
                    project["folderAvailable"] = str(outside) if outside else None
            self._send_json(200, {"projects": projects})
            return
        if parsed.path == "/api/backups":
            if not self._local_request():
                self._send_error_json(403, "Sicherungen sind nur auf dem lokalen PC verfügbar.")
                return
            self._send_json(200, {"backups": self._list_backups()})
            return
        if parsed.path == "/api/browse":
            if not self._local_request():
                self._send_error_json(403, "Der Ordnerbrowser ist nur auf dem lokalen PC verfügbar.")
                return
            self._send_browse_result(urlsplit(self.path).query)
            return
        if parsed.path == "/api/settings":
            local = self._local_request()
            self._send_json(200, {
                "readOnly": not local,
                "workspace": str(self._workspace_path()) if local else None,
                "workspaceName": self._workspace_path().name,
                "dataStorage": "lokale Datei im Benutzerprofil",
            })
            return
        if parsed.path.startswith("/project/"):
            if not self._local_request():
                self.send_error(403, "Projektordner können nur lokal am PC geöffnet werden.")
                return
            self._serve_project_preview_redirect(parsed.path)
            return
        if parsed.path in {"/", "/index.html", "/styles.css", "/app.js"}:
            super().do_GET()
            return
        if parsed.path.startswith("/api/"):
            self._send_error_json(404, "API-Endpunkt nicht gefunden.")
        else:
            self.send_error(404, "Datei nicht gefunden.")

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler method name
        path = urlsplit(self.path).path
        if path in {"/", "/index.html", "/styles.css", "/app.js"}:
            super().do_HEAD()
        else:
            self.send_error(404, "Datei nicht gefunden.")

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler method name
        if not self._write_allowed():
            return
        parsed = urlsplit(self.path)
        try:
            payload = self._read_json()
        except ValueError as error:
            self._send_error_json(400, str(error))
            return

        if parsed.path == "/api/workspace":
            if not isinstance(payload, dict):
                self._send_error_json(400, "Ordnerpfad fehlt.")
                return
            try:
                workspace, warnings = self._save_workspace_path(payload.get("path"))
            except (OSError, ValueError) as error:
                self._send_error_json(400, f"Projektordner konnte nicht verbunden werden: {error}")
                return
            self._send_json(200, {"workspace": str(workspace), "warnings": warnings})
            return

        if parsed.path == "/api/workspace/select":
            dialog = None
            try:
                import tkinter as tk
                from tkinter import filedialog

                dialog = tk.Tk()
                dialog.withdraw()
                try:
                    dialog.attributes("-topmost", True)
                except tk.TclError:
                    pass
                desktop_options = [Path.home() / "Desktop", Path.home() / "OneDrive" / "Desktop"]
                desktop = next((path for path in desktop_options if path.is_dir()), Path.home())
                initial = self._workspace_path() if self.workspace_configured and self._workspace_path().is_dir() else desktop
                selected = filedialog.askdirectory(title="Desktop-Hauptordner mit deinen Projektordnern auswählen", initialdir=str(initial), mustexist=True, parent=dialog)
            except Exception as error:
                self._send_error_json(501, f"System-Dateiauswahl nicht verfügbar: {error}")
                return
            finally:
                if dialog is not None:
                    try:
                        dialog.destroy()
                    except Exception:
                        pass
            if not selected:
                self._send_json(200, {"cancelled": True})
                return
            try:
                workspace, warnings = self._save_workspace_path(selected)
            except (OSError, ValueError) as error:
                self._send_error_json(400, f"Projektordner konnte nicht verbunden werden: {error}")
                return
            self._send_json(200, {"workspace": str(workspace), "warnings": warnings})
            return

        if parsed.path == "/api/workspace/scan":
            try:
                with LIBRARY_LOCK:
                    projects, added, linked, warnings = self._scan_workspace_projects()
            except (OSError, ValueError) as error:
                self._send_error_json(400, str(error))
                return
            self._send_json(200, {"projects": projects, "added": added, "linked": linked, "warnings": warnings})
            return

        if parsed.path == "/api/projects":
            if not self.workspace_configured:
                self._send_error_json(409, "Verbinde zuerst den vorhandenen Desktop-Hauptordner mit deinen Projekten.")
                return
            if not isinstance(payload, dict):
                self._send_error_json(400, "Projekt-Daten fehlen.")
                return
            data = dict(payload)
            data.setdefault("id", uuid.uuid4().hex)
            data.setdefault("status", "idea")
            data.setdefault("statusLabel", STATUS_LABELS["idea"])
            data.setdefault("icon", "gamepad" if data.get("category") == "Spielidee" else "spark")
            data.setdefault("color", "blue" if data.get("category") == "Spielidee" else "lime")
            data.setdefault("tags", ["Neu"])
            project = normalize_project(data, keep_folder=False)
            if not project:
                self._send_error_json(400, "Bitte gib einen Projektnamen ein.")
                return
            with LIBRARY_LOCK:
                projects = self._load_projects()
                if any(item["id"] == project["id"] for item in projects):
                    project["id"] = uuid.uuid4().hex
                if bool(payload.get("createFolder")):
                    try:
                        folder = self._create_folder(project)
                    except (OSError, ValueError) as error:
                        self._send_error_json(500, f"Projektordner konnte nicht angelegt werden: {error}")
                        return
                    project["folderPath"] = str(folder)
                    project["folderSignature"] = folder_signature(folder)
                projects.insert(0, project)
                self._save_projects(projects)
            self._send_json(201, {"project": project})
            return

        if parsed.path == "/api/projects/import":
            if not isinstance(payload, dict) or payload.get("format") != "control-center-library":
                self._send_error_json(400, "Das ist keine gültige Control-Center-Bibliothek.")
                return
            incoming = payload.get("projects")
            if not isinstance(incoming, list) or len(incoming) > 500:
                self._send_error_json(400, "Die Datei enthält keine gültige Projektliste (maximal 500 Einträge).")
                return
            imported: list[dict[str, object]] = []
            seen: set[str] = set()
            for item in incoming:
                project = normalize_project(item, keep_folder=False)
                if not project:
                    continue
                project["folderPath"] = None
                project["isDemo"] = False
                if project["id"] in seen:
                    project["id"] = uuid.uuid4().hex
                seen.add(str(project["id"]))
                imported.append(project)
            backup_name = None
            if self.library_path.exists():
                try:
                    self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    backup_path = self.data_dir / f"projects-backup-{stamp}-{uuid.uuid4().hex[:6]}.json"
                    shutil.copy2(self.library_path, backup_path)
                    if os.name != "nt":
                        backup_path.chmod(0o600)
                    backup_name = backup_path.name
                    old_backups = sorted(self.data_dir.glob("projects-backup-*.json"), reverse=True)
                    for old_backup in old_backups[10:]:
                        old_backup.unlink(missing_ok=True)
                except OSError as error:
                    self._send_error_json(500, f"Sicherung vor dem Import fehlgeschlagen: {error}")
                    return
            self._save_projects(imported)
            self._send_json(200, {"imported": len(imported), "backupCreated": backup_name is not None, "backupFile": backup_name, "projects": imported})
            return

        parts = [unquote(part) for part in parsed.path.strip("/").split("/")]
        if len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] in {"folder", "open"}:
            try:
                if parts[3] == "folder":
                    project = self._ensure_project_folder(parts[2])
                    self._send_json(200, {"project": project, "folderPath": project.get("folderPath")})
                else:
                    project = self._open_project_folder(parts[2])
                    self._send_json(200, {"opened": True, "project": project, "folderPath": project.get("folderPath")})
            except LookupError:
                self._send_error_json(404, "Projekt nicht gefunden.")
            except ValueError as error:
                self._send_error_json(501, str(error))
            except OSError as error:
                self._send_error_json(500, f"Projektordner konnte nicht bearbeitet werden: {error}")
            return

        if parsed.path == "/api/backups":
            try:
                backup_name = self._create_backup() if self.library_path.exists() else None
            except OSError as error:
                self._send_error_json(500, f"Sicherung konnte nicht erstellt werden: {error}")
                return
            self._send_json(200, {
                "backupCreated": backup_name is not None,
                "backupFile": backup_name,
                "backups": self._list_backups(),
            })
            return

        if parsed.path == "/api/backups/restore":
            if not isinstance(payload, dict) or not isinstance(payload.get("backup"), str):
                self._send_error_json(400, "Der Name der Sicherung fehlt.")
                return
            try:
                restored = self._restore_backup(payload["backup"])
            except (OSError, ValueError) as error:
                self._send_error_json(400, f"Sicherung konnte nicht gelesen werden: {error}")
                return
            if restored is None:
                self._send_error_json(404, "Sicherung nicht gefunden.")
                return
            self._send_json(200, {"restored": payload["backup"], "projects": restored, "backups": self._list_backups()})
            return

        self._send_error_json(404, "API-Endpunkt nicht gefunden.")

    def do_PATCH(self) -> None:  # noqa: N802 - stdlib handler method name
        if not self._write_allowed():
            return
        parts = [unquote(part) for part in urlsplit(self.path).path.strip("/").split("/")]
        if len(parts) != 3 or parts[:2] != ["api", "projects"]:
            self._send_error_json(404, "API-Endpunkt nicht gefunden.")
            return
        try:
            payload = self._read_json()
        except ValueError as error:
            self._send_error_json(400, str(error))
            return
        if not isinstance(payload, dict):
            self._send_error_json(400, "Änderungsdaten fehlen.")
            return
        with LIBRARY_LOCK:
            projects, project = self._project(parts[2])
            if not project:
                self._send_error_json(404, "Projekt nicht gefunden.")
                return
            for key in ("title", "description", "category", "notes", "updated"):
                if key in payload:
                    max_length = 80 if key == "title" else 20_000 if key == "notes" else 500
                    value = clean_text(payload[key], max_length)
                    if key == "title" and not value:
                        self._send_error_json(400, "Der Projektname darf nicht leer sein.")
                        return
                    project[key] = value
            if "favorite" in payload:
                project["favorite"] = bool(payload["favorite"])
            if "lastOpened" in payload:
                last_opened = payload["lastOpened"]
                project["lastOpened"] = last_opened if isinstance(last_opened, (int, float)) else None
            if "nextSteps" in payload:
                steps = payload["nextSteps"]
                if not isinstance(steps, list):
                    self._send_error_json(400, "Nächste Schritte müssen als Liste gesendet werden.")
                    return
                project["nextSteps"] = [clean_text(step, 180) for step in steps[:30] if isinstance(step, str) and step.strip()]
            if "tags" in payload:
                tags = payload["tags"]
                if not isinstance(tags, list):
                    self._send_error_json(400, "Schlagworte müssen als Liste gesendet werden.")
                    return
                project["tags"] = [tag for tag in (clean_text(item, 24) for item in tags[:10]) if tag]
            if "icon" in payload:
                if payload["icon"] not in PROJECT_ICONS:
                    self._send_error_json(400, "Unbekanntes Symbol.")
                    return
                project["icon"] = payload["icon"]
            if "color" in payload:
                if payload["color"] not in PROJECT_COLORS:
                    self._send_error_json(400, "Unbekannte Farbe.")
                    return
                project["color"] = payload["color"]
            if "folderPath" in payload:
                raw_folder = payload["folderPath"]
                if raw_folder in (None, ""):
                    project["folderPath"] = None
                    project["folderSignature"] = None
                else:
                    folder = self._folder_inside_workspace(raw_folder)
                    if folder is None:
                        self._send_error_json(400, "Dieser Ordner liegt nicht im verbundenen Hauptordner.")
                        return
                    project["folderPath"] = str(folder)
                    project["folderSignature"] = folder_signature(folder)
            if "status" in payload:
                status = payload["status"]
                if status not in STATUS_LABELS:
                    self._send_error_json(400, "Unbekannter Projektstatus.")
                    return
                project["status"] = status
                project["statusLabel"] = STATUS_LABELS[status]
            project["updated"] = "Gerade aktualisiert"
            project["updatedAt"] = now_iso()
            saved = dict(project)
            self._save_projects(projects)
        self._send_json(200, {"project": saved})

    def do_DELETE(self) -> None:  # noqa: N802 - stdlib handler method name
        """Remove a library entry. Project folders on disk are never touched."""
        if not self._write_allowed():
            return
        parts = [unquote(part) for part in urlsplit(self.path).path.strip("/").split("/")]
        if len(parts) != 3 or parts[:2] != ["api", "projects"]:
            self._send_error_json(404, "API-Endpunkt nicht gefunden.")
            return
        with LIBRARY_LOCK:
            projects = self._load_projects()
            remaining = [item for item in projects if item.get("id") != parts[2]]
            if len(remaining) == len(projects):
                self._send_error_json(404, "Projekt nicht gefunden.")
                return
            self._save_projects(remaining)
        self._send_json(200, {"deleted": parts[2], "projects": remaining})

    def _ensure_project_folder(self, project_id: str) -> dict[str, object]:
        """Create the working folder of a project if it does not exist yet."""
        with LIBRARY_LOCK:
            projects, project = self._project(project_id)
            if project is None:
                raise LookupError(project_id)
            folder = self._safe_project_folder(project)
            if folder is None:
                folder = self._create_folder(project)
                project["folderPath"] = str(folder)
                project["folderSignature"] = folder_signature(folder)
                project["updated"] = "Ordner angelegt"
                project["updatedAt"] = now_iso()
                self._save_projects(projects)
            return dict(project)

    def _open_project_folder(self, project_id: str) -> dict[str, object]:
        """Open the project folder in the file manager of the server machine."""
        with LIBRARY_LOCK:
            projects, project = self._project(project_id)
            if project is None:
                raise LookupError(project_id)
            folder = self._safe_project_folder(project)
            if folder is None:
                raise ValueError("Für dieses Projekt gibt es noch keinen gültigen Ordner im verbundenen Hauptordner.")
            open_error = self._open_folder(folder)
            if open_error:
                raise ValueError(open_error)
            project["lastOpened"] = int(datetime.now().timestamp() * 1000)
            project["updated"] = "Ordner geöffnet"
            project["updatedAt"] = now_iso()
            self._save_projects(projects)
            return dict(project)

    def _restore_backup(self, name: str) -> list[dict[str, object]] | None:
        """Replace the library with an existing backup (a fresh backup is kept)."""
        path = self.data_dir / Path(name).name
        if not path.is_file() or not path.name.startswith("projects-backup-") or not path.name.endswith(".json"):
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            return None
        projects: list[dict[str, object]] = []
        seen: set[str] = set()
        for item in value:
            project = normalize_project(item)
            if project and project["id"] not in seen:
                projects.append(project)
                seen.add(str(project["id"]))
        with LIBRARY_LOCK:
            if self.library_path.exists():
                self._create_backup()
            self._save_projects(projects)
        return projects

    def log_message(self, format: str, *args: object) -> None:
        # Keep the preview log readable without suppressing request errors.
        super().log_message(format, *args)


VERSION = "1.1.0"


def other_instance_running(data_dir: Path) -> str | None:
    """Best-effort check for a second Control Center using the same library."""
    if os.name == "nt":
        return None
    lock_path = data_dir / "control-center.lock"
    try:
        data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if lock_path.exists():
            raw = lock_path.read_text(encoding="utf-8").strip()
            if raw.isdigit() and raw != str(os.getpid()):
                try:
                    os.kill(int(raw), 0)
                except OSError:
                    pass
                else:
                    return raw
        lock_path.write_text(str(os.getpid()), encoding="utf-8")
        if os.name != "nt":
            lock_path.chmod(0o600)
    except OSError:
        return None
    return None


def reap_idle_content_servers(interval: float = 60.0) -> None:
    """Stop project servers nobody used for a while."""
    while True:
        time.sleep(interval)
        with PROJECT_CONTENT_LOCK:
            for project_id, server in list(ControlCenterHandler.project_content_servers.items()):
                last_access = getattr(server, "last_access", 0.0)
                if time.monotonic() - last_access <= CONTENT_SERVER_IDLE_SECONDS:
                    continue
                try:
                    server.shutdown()
                    server.server_close()
                except Exception:  # noqa: BLE001 - shutdown must never kill the reaper
                    pass
                ControlCenterHandler.project_content_servers.pop(project_id, None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the local Control Center.")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind (default: localhost).")
    parser.add_argument("--port", type=int, default=8765, help="Port to serve on (default: 8765).")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically.")
    parser.add_argument("--version", action="version", version=f"Control Center {VERSION}")
    parser.add_argument("--workspace", type=Path, default=None, help="Root folder for project directories (overrides the saved local setting).")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Folder for the local library JSON file.")
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    workspace_configured = bool(args.workspace)
    if args.workspace:
        workspace_root = args.workspace.expanduser().resolve()
    else:
        workspace_root = DEFAULT_WORKSPACE
        try:
            saved = json.loads((data_dir / "settings.json").read_text(encoding="utf-8"))
            if isinstance(saved, dict) and isinstance(saved.get("workspace"), str):
                workspace_root = Path(saved["workspace"]).expanduser().resolve()
                workspace_configured = True
        except (OSError, json.JSONDecodeError):
            pass
    ControlCenterHandler.data_dir = data_dir
    ControlCenterHandler.library_path = data_dir / "projects.json"
    ControlCenterHandler.workspace_root = workspace_root
    ControlCenterHandler.workspace_configured = workspace_configured

    other = other_instance_running(data_dir)
    if other:
        print(f"Hinweis: Control Center läuft bereits (Prozess {other}). Änderungen werden sonst doppelt geschrieben.")

    handler = partial(ControlCenterHandler, directory=str(ROOT))
    ControlCenterHandler.protocol_version = "HTTP/1.1"
    server: ThreadingHTTPServer | None = None
    chosen_port = args.port
    for candidate in range(args.port, args.port + 20):
        try:
            server = ThreadingHTTPServer((args.host, candidate), handler)
        except OSError:
            continue
        chosen_port = candidate
        break
    if server is None:
        print(f"Kein freier Port gefunden (versucht: {args.port} bis {args.port + 19}).", file=sys.stderr)
        return 1
    if chosen_port != args.port:
        print(f"Port {args.port} ist belegt — Control Center nutzt Port {chosen_port}.")
    server.daemon_threads = True
    threading.Thread(target=reap_idle_content_servers, name="content-server-reaper", daemon=True).start()
    url_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{url_host}:{chosen_port}/"
    print(f"Control Center {VERSION} läuft unter {url}")
    print(f"Projektordner: {workspace_root}")
    print("Bibliothek: lokal im Benutzerprofil; Änderungen von Netzwerkgeräten sind gesperrt.")
    print("Zum Beenden Ctrl+C drücken.")

    if not args.no_browser and args.host in {"127.0.0.1", "localhost", "::1"}:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nControl Center beendet.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
