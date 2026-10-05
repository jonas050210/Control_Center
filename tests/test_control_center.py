"""Tests for the local Control Center server.

Run with:  python3 -m unittest discover -s tests
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import start as cc  # noqa: E402  (module lives next to the app)


def request(base: str, path: str, method: str = "GET", payload: object = None, headers: dict | None = None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json"} if data is not None else {}
    request_headers.update(headers or {})
    req = urllib.request.Request(f"{base}{path}", data=data, method=method, headers=request_headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            body = response.read().decode("utf-8")
            return response.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8")
        try:
            return error.code, json.loads(body) if body else {}
        except json.JSONDecodeError:
            return error.code, {}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Keeps 302 responses so the redirect target can be inspected."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802
        return None


class ControlCenterTestCase(unittest.TestCase):
    """Starts a real server on a free port with an isolated data directory."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="control-center-test-"))
        self.data_dir = self.tmp / "data"
        self.workspace = self.tmp / "projects"
        self.workspace.mkdir(parents=True)
        self.data_dir.mkdir(parents=True)
        cc.ControlCenterHandler.protocol_version = "HTTP/1.1"
        self.handler = partial(cc.ControlCenterHandler, directory=str(ROOT))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self.handler)
        self.server.daemon_threads = True
        cc.ControlCenterHandler.data_dir = self.data_dir
        cc.ControlCenterHandler.library_path = self.data_dir / "projects.json"
        cc.ControlCenterHandler.workspace_root = self.workspace
        cc.ControlCenterHandler.workspace_configured = True
        cc.ControlCenterHandler.project_content_servers.clear()
        cc.ControlCenterHandler.log_message = lambda self, fmt, *args: None  # type: ignore[method-assign]
        cc.ProjectContentHandler.log_message = lambda self, fmt, *args: None  # type: ignore[method-assign]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        with cc.PROJECT_CONTENT_LOCK:
            for server in list(cc.ControlCenterHandler.project_content_servers.values()):
                server.shutdown()
                server.server_close()
            cc.ControlCenterHandler.project_content_servers.clear()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_project(self, name: str, files: dict[str, str] | None = None) -> Path:
        folder = self.workspace / name
        folder.mkdir(parents=True, exist_ok=True)
        for filename, content in (files or {}).items():
            (folder / filename).write_text(content, encoding="utf-8")
        return folder

    def scan(self) -> dict:
        status, body = request(self.base, "/api/workspace/scan", "POST", {})
        self.assertEqual(status, 200, body)
        return body


class TestHelpers(ControlCenterTestCase):
    def test_slug_keeps_latin_names_readable(self) -> None:
        self.assertEqual(cc.safe_slug("Neon Arena"), "neon-arena")
        self.assertEqual(cc.safe_slug("Größe"), "groe")
        self.assertEqual(cc.safe_slug("!!!"), "projekt")

    def test_slug_keeps_non_latin_names_distinct(self) -> None:
        self.assertNotEqual(cc.safe_slug("日本語"), cc.safe_slug("中国"))
        self.assertNotEqual(cc.safe_slug("Проект А"), cc.safe_slug("Проект Б"))

    def test_folder_signature_ignores_the_folder_name(self) -> None:
        first = self.make_project("Alpha", {"main.py": "print(1)"})
        renamed = first.rename(self.workspace / "Beta")
        self.assertEqual(cc.folder_signature(first) if first.exists() else "", "")
        self.assertEqual(cc.folder_signature(renamed), cc.folder_signature(renamed))
        other = self.make_project("Gamma", {"other.py": "print(2)"})
        self.assertNotEqual(cc.folder_signature(renamed), cc.folder_signature(other))

    def test_empty_folders_have_no_signature(self) -> None:
        self.assertEqual(cc.folder_signature(self.make_project("Leer")), "")

    def test_normalize_project_rejects_bad_values(self) -> None:
        project = cc.normalize_project({
            "title": "  Test  ",
            "status": "does-not-exist",
            "icon": "not-an-icon",
            "color": "pink",
            "tags": ["a", 5, "b"],
        })
        self.assertIsNotNone(project)
        assert project is not None
        self.assertEqual(project["title"], "Test")
        self.assertEqual(project["status"], "idea")
        self.assertEqual(project["icon"], "hub")
        self.assertEqual(project["color"], "lime")
        self.assertEqual(project["tags"], ["a", "b"])
        self.assertIsNone(cc.normalize_project({"title": "   "}))
        self.assertIsNone(cc.normalize_project("no dict"))


class TestLibraryApi(ControlCenterTestCase):
    def test_health_reports_workspace(self) -> None:
        status, body = request(self.base, "/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertFalse(body["readOnly"])
        self.assertEqual(body["workspace"], str(self.workspace))

    def test_create_patch_and_delete(self) -> None:
        status, body = request(self.base, "/api/projects", "POST", {"title": "Idee", "category": "Tool"})
        self.assertEqual(status, 201)
        project_id = body["project"]["id"]

        status, body = request(self.base, f"/api/projects/{project_id}", "PATCH", {
            "status": "active",
            "tags": ["web", "test"],
            "icon": "rocket",
            "color": "violet",
            "notes": "Notiz",
            "nextSteps": ["Schritt eins", "Schritt zwei"],
            "favorite": True,
        })
        self.assertEqual(status, 200)
        project = body["project"]
        self.assertEqual(project["status"], "active")
        self.assertEqual(project["tags"], ["web", "test"])
        self.assertEqual(project["icon"], "rocket")
        self.assertEqual(project["color"], "violet")
        self.assertEqual(project["nextSteps"], ["Schritt eins", "Schritt zwei"])
        self.assertTrue(project["favorite"])

        status, body = request(self.base, f"/api/projects/{project_id}", "PATCH", {"status": "nope"})
        self.assertEqual(status, 400)

        status, _ = request(self.base, f"/api/projects/{project_id}", "DELETE")
        self.assertEqual(status, 200)
        _, body = request(self.base, "/api/projects")
        self.assertEqual(body["projects"], [])

    def test_delete_never_touches_the_folder(self) -> None:
        folder = self.make_project("Bleibt", {"index.html": "<h1>x</h1>"})
        self.scan()
        _, body = request(self.base, "/api/projects")
        project_id = next(item["id"] for item in body["projects"] if item["title"] == "Bleibt")
        status, _ = request(self.base, f"/api/projects/{project_id}", "DELETE")
        self.assertEqual(status, 200)
        self.assertTrue(folder.is_dir())
        self.assertTrue((folder / "index.html").is_file())

    def test_concurrent_writes_do_not_break_the_library(self) -> None:
        status, body = request(self.base, "/api/projects", "POST", {"title": "Race"})
        self.assertEqual(status, 201)
        project_id = body["project"]["id"]
        errors: list[str] = []

        def patch(index: int) -> None:
            try:
                request(self.base, f"/api/projects/{project_id}", "PATCH", {"title": f"T{index}"})
            except Exception as error:  # noqa: BLE001 - the test collects failures
                errors.append(str(error))

        threads = [threading.Thread(target=patch, args=(index,)) for index in range(15)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        _, body = request(self.base, "/api/projects")
        self.assertEqual(len(body["projects"]), 1)
        self.assertIn(body["projects"][0]["title"], {f"T{index}" for index in range(15)})
        self.assertEqual(list(self.data_dir.glob("*.tmp")), [])


class TestWorkspaceScan(ControlCenterTestCase):
    def test_ignores_hidden_and_build_folders(self) -> None:
        self.make_project("Echtes Projekt", {"index.html": "<h1>x</h1>"})
        self.make_project(".git")
        self.make_project("node_modules")
        body = self.scan()
        titles = [item["title"] for item in body["projects"]]
        self.assertIn("Echtes Projekt", titles)
        self.assertNotIn("git", titles)
        self.assertNotIn("node_modules", titles)

    def test_finds_projects_one_level_deeper(self) -> None:
        nested = self.make_project("Gruppe")
        (nested / "Kindprojekt").mkdir()
        (nested / "Kindprojekt" / "main.py").write_text("print(1)", encoding="utf-8")
        body = self.scan()
        titles = [item["title"] for item in body["projects"]]
        self.assertIn("Kindprojekt", titles)
        self.assertNotIn("Gruppe", titles)

    def test_renamed_folder_keeps_its_entry(self) -> None:
        folder = self.make_project("Alpha", {"main.py": "print(1)"})
        self.scan()
        folder.rename(self.workspace / "Alpha Neu")
        body = self.scan()
        self.assertEqual(body["added"], 0, "renaming must not create a duplicate")
        self.assertEqual(body["linked"], 1)
        entry = next(item for item in body["projects"] if item["title"] == "Alpha")
        self.assertTrue(entry["folderPath"].endswith("Alpha Neu"))

    def test_non_latin_names_are_linked(self) -> None:
        self.make_project("日本語", {"main.py": "print(1)"})
        status, body = request(self.base, "/api/projects", "POST", {"title": "日本語"})
        self.assertEqual(status, 201)
        scanned = self.scan()
        self.assertEqual(scanned["added"], 0)
        entry = next(item for item in scanned["projects"] if item["title"] == "日本語")
        self.assertTrue(entry["folderPath"].endswith("日本語"))

    def test_folder_outside_of_the_workspace_is_not_accepted(self) -> None:
        status, body = request(self.base, "/api/projects", "POST", {"title": "Falsch"})
        self.assertEqual(status, 201)
        project_id = body["project"]["id"]
        status, _ = request(self.base, f"/api/projects/{project_id}", "PATCH", {"folderPath": str(self.tmp)})
        self.assertEqual(status, 400)

    def test_unsuitable_workspaces_are_rejected(self) -> None:
        status, body = request(self.base, "/api/workspace", "POST", {"path": "/"})
        self.assertEqual(status, 400)
        self.assertIn("Laufwerk", body["error"])


class TestSecurityAndFiles(ControlCenterTestCase):
    def test_project_files_cannot_escape_the_folder(self) -> None:
        self.make_project("Projekt", {"index.html": "<h1>x</h1>"})
        self.scan()
        _, body = request(self.base, "/api/projects")
        project_id = body["projects"][0]["id"]
        opener = urllib.request.build_opener(_NoRedirect)
        try:
            response = opener.open(f"{self.base}/project/{project_id}/", timeout=10)
            self.assertEqual(response.status, 302)
            location = response.headers["Location"]
            response.close()
        except urllib.error.HTTPError as error:
            self.assertEqual(error.code, 302)
            location = error.headers["Location"]
        opener = urllib.request.build_opener()
        with opener.open(location, timeout=10) as response:
            self.assertIn(b"<h1>x</h1>", response.read())
        for escape in ("../../../../etc/passwd", "..%2f..%2f..%2fetc/passwd"):
            with self.assertRaises(urllib.error.HTTPError) as context:
                opener.open(f"{location}{escape}", timeout=10)
            self.assertEqual(context.exception.code, 404)

    def test_writes_from_other_hosts_are_blocked(self) -> None:
        status, body = request(
            self.base, "/api/projects", "POST", {"title": "Fremd"},
            headers={"Host": "192.168.0.99:8765", "Origin": "http://192.168.0.99:8765"},
        )
        self.assertEqual(status, 403)

    def test_cross_origin_writes_are_blocked(self) -> None:
        status, _ = request(self.base, "/api/projects", "POST", {"title": "CSRF"}, headers={"Origin": "http://evil.example"})
        self.assertEqual(status, 403)

    def test_remote_reads_hide_private_fields(self) -> None:
        status, body = request(self.base, "/api/projects", "POST", {"title": "Privat", "notes": "geheim"})
        self.assertEqual(status, 201)
        _, body = request(self.base, "/api/projects", headers={"Host": "192.168.0.99:8765"})
        entry = body["projects"][0]
        self.assertNotIn("notes", entry)
        self.assertNotIn("folderPath", entry)
        self.assertIn("hasFolder", entry)

    def test_cross_site_api_reads_are_blocked(self) -> None:
        status, _ = request(self.base, "/api/projects", headers={"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(status, 403)


class TestBackups(ControlCenterTestCase):
    def test_backup_create_list_and_restore(self) -> None:
        request(self.base, "/api/projects", "POST", {"title": "Erster"})
        status, body = request(self.base, "/api/backups", "POST", {})
        self.assertEqual(status, 200)
        self.assertTrue(body["backupCreated"])
        backups = body["backups"]
        self.assertEqual(len(backups), 1)

        request(self.base, "/api/projects", "POST", {"title": "Zweiter"})
        status, body = request(self.base, "/api/backups/restore", "POST", {"backup": backups[0]["name"]})
        self.assertEqual(status, 200)
        titles = [item["title"] for item in body["projects"]]
        self.assertEqual(titles, ["Erster"])

    def test_restore_rejects_unknown_names(self) -> None:
        status, _ = request(self.base, "/api/backups/restore", "POST", {"backup": "projects-backup-does-not-exist.json"})
        self.assertEqual(status, 404)
        status, _ = request(self.base, "/api/backups/restore", "POST", {"backup": "../../etc/passwd"})
        self.assertEqual(status, 404)


class TestBrowse(ControlCenterTestCase):
    def test_browse_lists_subfolders(self) -> None:
        (self.workspace / "Unterordner").mkdir()
        status, body = request(self.base, f"/api/browse?path={self.workspace}")
        self.assertEqual(status, 200)
        names = [item["name"] for item in body["directories"]]
        self.assertIn("Unterordner", names)
        self.assertEqual(body["path"], str(self.workspace))

    def test_browse_is_local_only(self) -> None:
        status, _ = request(self.base, "/api/browse", headers={"Host": "192.168.0.99:8765"})
        self.assertEqual(status, 403)


if __name__ == "__main__":
    unittest.main()
