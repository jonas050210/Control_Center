import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from sandboxai.run_control import RunControl, from_cli_paths


class RunControlTests(unittest.TestCase):
    def test_unmanaged_cli_has_no_control_object(self):
        self.assertIsNone(from_cli_paths(None, None, None))

    def test_status_is_published_atomically_with_runtime_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control = RunControl(root / "command.json", root / "status.json", root / "events.jsonl")
            control.start(training_type="ppo", total_training_steps=1000)
            control.running(timesteps=250, progress=0.25)
            status = json.loads((root / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "Running")
            self.assertEqual(status["timesteps"], 250)
            self.assertEqual(status["total_training_steps"], 1000)
            self.assertGreater(status["pid"], 0)

    def test_pause_resume_is_cooperative_and_stop_is_graceful(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command_path = root / "command.json"
            control = RunControl(
                command_path, root / "status.json", root / "events.jsonl", poll_interval=0.02
            )
            control.start()
            control.running()
            command_path.write_text(
                json.dumps({"command": "pause", "sequence": 1}), encoding="utf-8"
            )
            result: list[bool] = []
            worker = threading.Thread(target=lambda: result.append(control.checkpoint()))
            worker.start()
            deadline = time.time() + 2.0
            while control.state != "Paused" and time.time() < deadline:
                time.sleep(0.01)
            self.assertEqual(control.state, "Paused")
            command_path.write_text(
                json.dumps({"command": "resume", "sequence": 2}), encoding="utf-8"
            )
            worker.join(timeout=2.0)
            self.assertFalse(worker.is_alive())
            self.assertEqual(result, [True])
            self.assertEqual(control.state, "Running")

            command_path.write_text(
                json.dumps({"command": "stop", "sequence": 3}), encoding="utf-8"
            )
            self.assertFalse(control.checkpoint())
            self.assertTrue(control.stop_requested)
            self.assertEqual(control.state, "Stopping")
            control.finish(timesteps=250)
            self.assertEqual(control.state, "Finished")
            self.assertTrue(control.snapshot()["stopped"])

    def test_error_is_visible_to_the_dashboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control = RunControl(status_path=root / "status.json", event_path=root / "events.jsonl")
            control.fail(RuntimeError("CUDA unavailable"))
            status = json.loads((root / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "Error")
            self.assertIn("CUDA unavailable", status["error"])
            event = json.loads((root / "events.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(event["category"], "error")


if __name__ == "__main__":
    unittest.main()
