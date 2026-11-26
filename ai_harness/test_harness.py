"""Behavioral tests: run `py -m unittest discover -s ai_harness -p test_harness.py -v`."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest

from harness import HarnessError, Ledger


def task(task_id="LIT-001", dependencies=None, scopes=None):
    return {"id": task_id, "title": "Check paper evidence", "role": "literature",
            "description": "Read sources and report their scope.",
            "depends_on": dependencies or [],
            "write_paths": scopes or [f"ai_harness/artifacts/{task_id}"],
            "acceptance": ["Include source URLs and limitations."]}


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="research-harness-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "ai_harness").mkdir()
        self.ledger = Ledger(self.root)
        self.write_seed([task(), task("REASON-001", ["LIT-001"])])
        self.ledger.init()

    def write_seed(self, tasks):
        (self.root / "ai_harness/tasks.json").write_text(json.dumps({"version": 1, "tasks": tasks}), encoding="utf-8")

    def evidence(self, task_id="LIT-001", text="Source-backed report."):
        name = f"ai_harness/artifacts/{task_id}/report.md"
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return name

    def submit(self, task_id="LIT-001", agent="miner"):
        self.ledger.claim(task_id, agent, "literature")
        return self.ledger.mutate("submit", task_id, agent, artifacts=[self.evidence(task_id)], summary="Primary sources checked.")

    def test_dependencies_and_independent_review(self):
        with self.assertRaises(HarnessError):
            self.ledger.claim("REASON-001", "reasoner", "literature")
        self.submit()
        with self.assertRaises(HarnessError):
            self.ledger.claim("REASON-001", "reasoner", "literature")
        with self.assertRaises(HarnessError):
            self.ledger.mutate("review", "LIT-001", "miner", verdict="accept", note="Reviewed")
        self.ledger.mutate("review", "LIT-001", "critic", verdict="accept", note="Checked primary sources.")
        self.assertEqual(self.ledger.claim("REASON-001", "reasoner", "literature")["state"], "active")

    def test_revision_retains_owner_and_requires_resubmission(self):
        self.submit()
        self.ledger.mutate("review", "LIT-001", "critic", verdict="revise", note="Add missing limitations.")
        self.assertEqual(self.ledger.show("LIT-001")["owner"], "miner")
        self.evidence(text="Revised primary evidence with limitations.")
        self.ledger.mutate("submit", "LIT-001", "miner", artifacts=[self.evidence()], summary="Added limitations.")
        self.ledger.mutate("review", "LIT-001", "critic", verdict="accept", note="Limitations now included.")
        self.assertTrue(self.ledger.check()["ok"])

    def test_wrong_owner_role_and_state(self):
        with self.assertRaises(HarnessError):
            self.ledger.claim("LIT-001", "worker", "engineer")
        self.ledger.claim("LIT-001", "miner", "literature")
        for action in ("heartbeat", "block", "release", "submit"):
            with self.assertRaises(HarnessError):
                self.ledger.mutate(action, "LIT-001", "imposter", reason="reason", summary="summary", artifacts=[self.evidence()])
        with self.assertRaises(HarnessError):
            self.ledger.mutate("review", "LIT-001", "critic", verdict="accept", note="Not submitted.")
        self.assertEqual(self.ledger.show("LIT-001")["state"], "active")

    def test_competing_claims_are_atomic_with_separate_connections(self):
        barrier = threading.Barrier(8)

        def attempt(i):
            barrier.wait()
            try:
                Ledger(self.root).claim("LIT-001", f"miner-{i}", "literature")
                return True
            except HarnessError:
                return False

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(attempt, range(8)))
        self.assertEqual(sum(results), 1)
        events = self.ledger.show("LIT-001")["events"]
        self.assertEqual(sum(e["action"] == "claim" for e in events), 1)

    def test_parent_child_and_case_alias_scope_conflicts(self):
        self.ledger.add(task("PARENT", scopes=["Shared"]))
        self.ledger.add(task("CHILD", scopes=["shared/subdir"]))
        self.ledger.claim("PARENT", "writer-a", "literature")
        with self.assertRaises(HarnessError):
            self.ledger.claim("CHILD", "writer-b", "literature")
        self.ledger.mutate("block", "PARENT", "writer-a", reason="Awaiting source access.")
        with self.assertRaises(HarnessError):
            self.ledger.claim("CHILD", "writer-b", "literature")
        self.ledger.mutate("release", "PARENT", "writer-a", reason="Stopped work; release reservation.")
        self.ledger.claim("CHILD", "writer-b", "literature")

    def test_unsafe_paths_and_out_of_scope_artifact(self):
        for bad in ("../escape", "/absolute", "C:/absolute", "\\\\host\\share", "foo/../bar", "foo//bar", "foo./bar", "foo /bar"):
            with self.subTest(path=bad), self.assertRaises(HarnessError):
                self.ledger.safe_path(bad)
        self.ledger.claim("LIT-001", "miner", "literature")
        (self.root / "unrelated.md").write_text("Do not overwrite.", encoding="utf-8")
        before = len(self.ledger.show("LIT-001")["events"])
        with self.assertRaises(HarnessError):
            self.ledger.mutate("submit", "LIT-001", "miner", artifacts=["unrelated.md"], summary="Bad path")
        self.assertEqual(self.ledger.show("LIT-001")["state"], "active")
        self.assertEqual(len(self.ledger.show("LIT-001")["events"]), before)

    def test_symlink_escape_when_supported(self):
        with tempfile.TemporaryDirectory(prefix="outside-harness-") as other:
            try:
                (self.root / "escape-link").symlink_to(other, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("Symlink creation unavailable to this Windows account.")
            with self.assertRaises(HarnessError):
                self.ledger.safe_path("escape-link/report.md")

    def test_empty_missing_and_changed_evidence_rejected(self):
        self.ledger.claim("LIT-001", "miner", "literature")
        for name in ("ai_harness/artifacts/LIT-001/missing.md", self.evidence(text="")):
            with self.assertRaises(HarnessError):
                self.ledger.mutate("submit", "LIT-001", "miner", artifacts=[name], summary="Invalid evidence")
        name = self.evidence()
        self.ledger.mutate("submit", "LIT-001", "miner", artifacts=[name], summary="Evidence ready")
        self.evidence(text="Changed after submission.")
        with self.assertRaises(HarnessError):
            self.ledger.mutate("review", "LIT-001", "critic", verdict="accept", note="Cannot accept changed evidence.")
        self.assertEqual(self.ledger.show("LIT-001")["state"], "review")
        with self.assertRaises(HarnessError):
            self.ledger.check()

    def test_seed_idempotent_does_not_replace_work(self):
        self.submit()
        changed = task()
        changed["title"] = "Would overwrite existing title"
        self.write_seed([changed, task("REASON-001", ["LIT-001"])])
        self.assertEqual(self.ledger.init(), {"added": 0, "preserved": 2})
        item = self.ledger.show("LIT-001")
        self.assertEqual(item["state"], "review")
        self.assertEqual(item["spec"]["title"], "Check paper evidence")

    def test_invalid_graph_and_add_leave_ledger_unchanged(self):
        before = self.ledger.status()
        with self.assertRaises(HarnessError):
            self.ledger.add(task("UNKNOWN", ["MISSING"]))
        with self.assertRaises(HarnessError):
            self.ledger.add(task("LOOP", ["LOOP"]))
        with self.assertRaises(HarnessError):
            self.ledger.add(task())
        self.write_seed([task("CYCLE-A", ["CYCLE-B"]), task("CYCLE-B", ["CYCLE-A"])])
        with self.assertRaises(HarnessError):
            self.ledger.init()
        after = self.ledger.status()
        self.assertEqual(before["tasks"], after["tasks"])
        self.assertEqual(before["events"], after["events"])

    def test_stale_tasks_are_not_stolen_and_recovery_is_explicit(self):
        self.ledger.claim("LIT-001", "miner", "literature")
        stale = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        with closing(sqlite3.connect(self.ledger.db)) as con:
            with con:
                con.execute("UPDATE tasks SET updated=? WHERE id='LIT-001'", (stale,))
        self.assertTrue(self.ledger.show("LIT-001")["stale"])
        with self.assertRaises(HarnessError):
            self.ledger.claim("LIT-001", "replacement", "literature")
        self.ledger.mutate("recover", "LIT-001", "coordinator", reason="Confirmed original process stopped.")
        self.ledger.claim("LIT-001", "replacement", "literature")
        self.assertEqual(self.ledger.show("LIT-001")["owner"], "replacement")
        self.assertIn("Previous owner: miner", self.ledger.show("LIT-001")["events"][-2]["detail"])

    def test_snapshot_and_cli_are_portable(self):
        self.submit()
        self.ledger.snapshot()
        data = json.loads((self.root / "ai_harness/state.snapshot.json").read_text(encoding="utf-8"))
        self.assertEqual(data["tasks"][0]["state"], "review")
        self.assertIn("LIT-001", (self.root / "ai_harness/BOARD.md").read_text(encoding="utf-8"))
        script = Path(__file__).with_name("harness.py")
        result = subprocess.run([sys.executable, str(script), "--root", str(self.root), "status", "--format", "json"], text=True, encoding="utf-8", capture_output=True, cwd=self.temp.name)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(json.loads(result.stdout)["tasks"]), 2)
        result = subprocess.run([sys.executable, str(script), "--root", str(self.root), "packet", "LIT-001"], text=True, encoding="utf-8", capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ai_harness/roles/literature.md", result.stdout)

    def test_case_insensitive_task_ids_do_not_alias_artifact_directories(self):
        before = self.ledger.status()
        with self.assertRaises(HarnessError):
            self.ledger.add(task("lit-001"))
        after = self.ledger.status()
        self.assertEqual(before["tasks"], after["tasks"])
        self.assertEqual(before["events"], after["events"])

    def test_scopes_cannot_overwrite_coordination_state(self):
        for scope in ("ai_harness", "AI_HARNESS/state.sqlite3", "ai_harness/state.sqlite3-journal",
                      "ai_harness/state.snapshot.json", "ai_harness/BOARD.md"):
            with self.subTest(scope=scope), self.assertRaises(HarnessError):
                self.ledger.add(task("BAD-SCOPE", scopes=[scope]))
        self.assertEqual(len(self.ledger.status()["tasks"]), 2)


if __name__ == "__main__":
    unittest.main()
