"""Local, cooperative multi-agent task ledger. Python standard library only.

Coordinates agents already running in their own hosts; does not launch models.
SQLite locking is for one computer, not simultaneous OneDrive clients.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile

ROLES = {"literature", "reasoning", "engineer", "reviewer", "coordinator"}
STATES = {"pending", "active", "review", "done", "blocked"}
BUSY = {"active", "review", "blocked"}
ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,79}\Z")


class HarnessError(Exception):
    pass


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def identifier(value):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise HarnessError("Identifiers must start with a letter and contain only letters, digits, _ or - (max 80).")
    return value


def required_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise HarnessError(f"{name} must be nonempty text.")
    return value.strip()


def within(child, parent):
    c, p = str(child).replace("\\", "/").casefold(), str(parent).replace("\\", "/").casefold()
    return c == p or c.startswith(p.rstrip("/") + "/")


class Ledger:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.folder = self.safe_path("ai_harness")
        self.db = self.safe_path("ai_harness/state.sqlite3")

    def safe_path(self, name):
        required_text(name, "path")
        value = name.replace("\\", "/")
        parts = value.split("/")
        if (value.startswith("/") or ":" in value or "\x00" in value
                or any(p in {"", ".", ".."} or p.endswith((".", " ")) for p in parts)):
            raise HarnessError(f"Use a project-relative path without traversal or aliases: {name}")
        path = self.root.joinpath(*parts).resolve()
        if not within(path, self.root):
            raise HarnessError(f"Path escapes project (possibly through a symlink/junction): {name}")
        return path

    def validate_spec(self, spec):
        if not isinstance(spec, dict):
            raise HarnessError("Task specification must be an object.")
        identifier(spec.get("id"))
        for field in ("title", "description"):
            required_text(spec.get(field), field)
        if spec.get("role") not in ROLES:
            raise HarnessError("Unknown task role.")
        for field in ("depends_on", "write_paths", "acceptance"):
            values = spec.get(field)
            if not isinstance(values, list) or any(not isinstance(x, str) or not x.strip() for x in values):
                raise HarnessError(f"{field} must be a list of nonempty strings.")
            if len(values) != len(set(values)):
                raise HarnessError(f"Duplicate values in {field}.")
        if not spec["write_paths"] or not spec["acceptance"]:
            raise HarnessError("A task needs write_paths and acceptance criteria.")
        for dependency in spec["depends_on"]:
            identifier(dependency)
        for scope in spec["write_paths"]:
            path = self.safe_path(scope)
            protected = [self.safe_path("ai_harness/" + name) for name in
                         ("state.sqlite3", "state.snapshot.json", "BOARD.md")]
            if any(within(path, p) or within(p, path) for p in protected):
                raise HarnessError("Task scopes cannot include generated ledger or snapshot files.")
            if str(path).casefold().startswith(str(self.db).casefold() + "-"):
                raise HarnessError("Task scopes cannot include SQLite auxiliary files.")

    def validate_graph(self, specs):
        mapping = {}
        identifiers = set()
        for spec in specs:
            self.validate_spec(spec)
            if spec["id"].casefold() in identifiers:
                raise HarnessError(f"Duplicate task ID: {spec['id']}")
            identifiers.add(spec["id"].casefold())
            mapping[spec["id"]] = spec
        visiting, visited = set(), set()

        def visit(task_id):
            if task_id in visiting:
                raise HarnessError(f"Dependency cycle at {task_id}.")
            if task_id not in mapping:
                raise HarnessError(f"Unknown dependency: {task_id}")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dep in mapping[task_id]["depends_on"]:
                visit(dep)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in mapping:
            visit(task_id)

    @contextmanager
    def transaction(self, initialize=False):
        if not initialize and not self.db.is_file():
            raise HarnessError("No ledger. Run init first.")
        if initialize:
            self.folder.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.db), timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("BEGIN IMMEDIATE")
            if initialize:
                connection.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, spec TEXT NOT NULL, state TEXT NOT NULL, owner TEXT, artifacts TEXT NOT NULL DEFAULT '[]', summary TEXT NOT NULL DEFAULT '', created TEXT NOT NULL, updated TEXT NOT NULL)")
                connection.execute("CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, task TEXT NOT NULL, action TEXT NOT NULL, agent TEXT NOT NULL, detail TEXT NOT NULL, at TEXT NOT NULL)")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def event(connection, task, action, agent, detail):
        connection.execute("INSERT INTO events(task,action,agent,detail,at) VALUES(?,?,?,?,?)", (task, action, agent, detail, now()))

    @staticmethod
    def tasks(connection):
        result = []
        for row in connection.execute("SELECT * FROM tasks ORDER BY created,id"):
            item = dict(row)
            item["spec"] = json.loads(item["spec"])
            item["artifacts"] = json.loads(item["artifacts"])
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(item["updated"])).total_seconds()
            item["stale"] = item["state"] == "active" and age > 1800
            result.append(item)
        return result

    def get(self, connection, task_id):
        identifier(task_id)
        for item in self.tasks(connection):
            if item["id"] == task_id:
                return item
        raise HarnessError(f"Unknown task: {task_id}")

    @staticmethod
    def owner(item, agent, states):
        identifier(agent)
        if item["state"] not in states:
            raise HarnessError(f"Task is {item['state']}; expected one of {sorted(states)}.")
        if item["owner"] != agent:
            raise HarnessError(f"Task belongs to {item['owner']}, not {agent}.")

    def init(self, seed=None):
        seed = Path(seed) if seed else self.folder / "tasks.json"
        data = json.loads(seed.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("tasks"), list):
            raise HarnessError("Seed must contain version=1 and a tasks list.")
        self.validate_graph(data["tasks"])
        with self.transaction(initialize=True) as con:
            existing = self.tasks(con)
            known = {t["id"] for t in existing}
            additions = [s for s in data["tasks"] if s["id"] not in known]
            self.validate_graph([t["spec"] for t in existing] + additions)
            for spec in additions:
                stamp = now()
                con.execute("INSERT INTO tasks(id,spec,state,created,updated) VALUES(?,?,'pending',?,?)", (spec["id"], json.dumps(spec), stamp, stamp))
                self.event(con, spec["id"], "init", "coordinator", "Seeded pending task")
        return {"added": len(additions), "preserved": len(existing)}

    def add(self, spec):
        with self.transaction() as con:
            self.validate_graph([t["spec"] for t in self.tasks(con)] + [spec])
            stamp = now()
            con.execute("INSERT INTO tasks(id,spec,state,created,updated) VALUES(?,?,'pending',?,?)", (spec["id"], json.dumps(spec), stamp, stamp))
            self.event(con, spec["id"], "add", "coordinator", "Added task specification")
        return {"added": spec["id"]}

    def claim(self, task_id, agent, role):
        identifier(agent)
        with self.transaction() as con:
            item = self.get(con, task_id)
            if item["state"] != "pending":
                raise HarnessError("Task is not pending; it cannot be claimed.")
            if role != item["spec"]["role"]:
                raise HarnessError("Role does not match assignment.")
            all_tasks = self.tasks(con)
            done = {t["id"] for t in all_tasks if t["state"] == "done"}
            missing = set(item["spec"]["depends_on"]) - done
            if missing:
                raise HarnessError(f"Dependencies not accepted yet: {sorted(missing)}")
            scopes = [self.safe_path(p) for p in item["spec"]["write_paths"]]
            for other in all_tasks:
                if other["state"] not in BUSY:
                    continue
                for other_path in other["spec"]["write_paths"]:
                    p = self.safe_path(other_path)
                    if any(within(p, s) or within(s, p) for s in scopes):
                        raise HarnessError(f"Write scope overlaps reserved task {other['id']}.")
            con.execute("UPDATE tasks SET state='active',owner=?,artifacts='[]',summary='',updated=? WHERE id=?", (agent, now(), task_id))
            self.event(con, task_id, "claim", agent, role)
            artifact_dir = self.safe_path(f"ai_harness/artifacts/{task_id}")
            artifact_dir.mkdir(parents=True, exist_ok=True)
            return self.get(con, task_id)

    def artifact(self, spec, name):
        path = self.safe_path(name)
        if not any(within(path, self.safe_path(scope)) for scope in spec["write_paths"]):
            raise HarnessError(f"Artifact is outside assigned write scopes: {name}")
        if not path.is_file() or path.stat().st_size == 0:
            raise HarnessError(f"Artifact must be an existing nonempty file: {name}")
        content = path.read_bytes()
        return {"path": name.replace("\\", "/"), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}

    def mutate(self, action, task_id, agent, **options):
        identifier(agent)
        with self.transaction() as con:
            item = self.get(con, task_id)
            if action == "review":
                if item["state"] != "review" or item["owner"] == agent:
                    raise HarnessError("Review requires a submitted task and a different reviewer identity.")
                note = required_text(options.get("note"), "review note")
                if options.get("verdict") not in {"accept", "revise"}:
                    raise HarnessError("Review verdict must be accept or revise.")
                self.verify_artifacts(item)
                state = "done" if options["verdict"] == "accept" else "active"
                con.execute("UPDATE tasks SET state=?,updated=? WHERE id=?", (state, now(), task_id))
                self.event(con, task_id, "review-" + options["verdict"], agent, note)
            elif action == "recover":
                if item["state"] not in BUSY:
                    raise HarnessError("Only active, blocked, or review tasks can be recovered.")
                reason = required_text(options.get("reason"), "recovery reason")
                con.execute("UPDATE tasks SET state='pending',owner=NULL,artifacts='[]',summary='',updated=? WHERE id=?", (now(), task_id))
                self.event(con, task_id, action, agent, f"Previous owner: {item['owner']}. Coordinator confirms prior writer stopped. {reason}")
            else:
                self.owner(item, agent, {"active", "blocked"} if action == "release" else {"active"})
                if action == "submit":
                    summary = required_text(options.get("summary"), "summary")
                    names = options.get("artifacts", [])
                    if not names or len(names) != len(set(names)):
                        raise HarnessError("Provide at least one unique artifact.")
                    artifacts = [self.artifact(item["spec"], n) for n in names]
                    con.execute("UPDATE tasks SET state='review',artifacts=?,summary=?,updated=? WHERE id=?", (json.dumps(artifacts), summary, now(), task_id))
                    detail = summary
                elif action == "heartbeat":
                    con.execute("UPDATE tasks SET updated=? WHERE id=?", (now(), task_id))
                    detail = "Heartbeat"
                elif action in {"block", "release"}:
                    detail = required_text(options.get("reason"), "reason")
                    state = "blocked" if action == "block" else "pending"
                    owner = agent if action == "block" else None
                    con.execute("UPDATE tasks SET state=?,owner=?,updated=? WHERE id=?", (state, owner, now(), task_id))
                else:
                    raise HarnessError(f"Unknown mutation: {action}")
                self.event(con, task_id, action, agent, detail)
            return self.get(con, task_id)

    def verify_artifacts(self, item):
        if not item["artifacts"]:
            raise HarnessError(f"No submitted evidence for {item['id']}.")
        for expected in item["artifacts"]:
            actual = self.artifact(item["spec"], expected["path"])
            if actual != expected:
                raise HarnessError(f"Artifact changed since submission: {expected['path']}. Recover/resubmit or request revision before editing.")

    def state(self, con):
        return {"version": 1, "generated_at": now(), "project_root": str(self.root),
                "tasks": self.tasks(con), "events": [dict(r) for r in con.execute("SELECT * FROM events ORDER BY seq")]}

    def status(self):
        with self.transaction() as con:
            return self.state(con)

    def show(self, task_id):
        with self.transaction() as con:
            item = self.get(con, task_id)
            item["events"] = [dict(r) for r in con.execute("SELECT * FROM events WHERE task=? ORDER BY seq", (task_id,))]
            return item

    def packet(self, task_id):
        item = self.show(task_id)
        spec = item["spec"]
        return "\n".join([
            f"# Task {task_id}: {spec['title']}", f"Project root: {self.root}",
            f"Role: {spec['role']}; status: {item['state']}; owner: {item['owner'] or 'unclaimed'}", "",
            "Read AGENTS.md, projectIdea.md, ai_harness/README.md, ai_harness/DECISIONS.md,",
            f"and ai_harness/roles/{spec['role']}.md. Current user instructions take precedence.",
            "Inspect current ledger with status/show; BOARD.md is only a snapshot.",
            "Claim this task with your unique agent identity before editing. Do not claim a role you were not assigned.",
            "", spec["description"], "", "Dependencies: " + (", ".join(spec["depends_on"]) or "none"),
            "Read accepted dependency reports and show outputs before using their conclusions.",
            "Exclusive permitted write paths:", *[f"- {p}" for p in spec["write_paths"]], "",
            "Acceptance criteria:", *[f"- {a}" for a in spec["acceptance"]], "",
            "Record evidence, limitations, unresolved questions, and next action. Submit all output files for independent review.",
            "Submitting is not acceptance. Only an independent reviewer may accept the task. No automatic training, model launches, or spending is implied.",
        ])

    def check(self):
        with self.transaction() as con:
            tasks = self.tasks(con)
            self.validate_graph([t["spec"] for t in tasks])
            done = {t["id"] for t in tasks if t["state"] == "done"}
            for t in tasks:
                if t["state"] not in STATES:
                    raise HarnessError(f"Invalid state: {t['state']}")
                if t["state"] != "pending":
                    identifier(t["owner"])
                    if set(t["spec"]["depends_on"]) - done:
                        raise HarnessError(f"Unaccepted dependency for {t['id']}.")
                if t["state"] in {"review", "done"}:
                    self.verify_artifacts(t)
            for i, left in enumerate(tasks):
                if left["state"] not in BUSY:
                    continue
                for right in tasks[i + 1:]:
                    if right["state"] in BUSY:
                        for a in left["spec"]["write_paths"]:
                            for b in right["spec"]["write_paths"]:
                                x, y = self.safe_path(a), self.safe_path(b)
                                if within(x, y) or within(y, x):
                                    raise HarnessError(f"Conflicting reservations: {left['id']} and {right['id']}")
            return {"ok": True, "tasks": len(tasks), "stale": [t["id"] for t in tasks if t["stale"]]}

    def snapshot(self):
        # Hold the ledger write lock through both atomic replacements so snapshots
        # from simultaneous processes cannot interleave. A crash can still leave
        # one older snapshot; the ledger remains authoritative. Rerun snapshot.
        with self.transaction() as con:
            data = self.state(con)
            for name, content in (("state.snapshot.json", json.dumps(data, indent=2, ensure_ascii=False) + "\n"),
                                  ("BOARD.md", markdown(data))):
                target = self.safe_path("ai_harness/" + name)
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=self.folder, delete=False) as f:
                    f.write(content)
                    temporary = f.name
                try:
                    os.replace(temporary, target)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
        return {"written": ["ai_harness/BOARD.md", "ai_harness/state.snapshot.json"]}


def markdown(data):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    lines = ["# Research task board", "", f"Generated: {data['generated_at']}", "",
             "Snapshot only. Run `py ai_harness/harness.py status` for current state.", "",
             "| Task | Role | State | Owner | Dependencies | Title |", "| --- | --- | --- | --- | --- | --- |"]
    for t in data["tasks"]:
        s = t["spec"]
        lines.append("| " + " | ".join(map(cell, [t["id"], s["role"], t["state"] + (" (stale)" if t["stale"] else ""), t["owner"] or "—", ", ".join(s["depends_on"]) or "—", s["title"]])) + " |")
    lines.extend(["", "## Submitted evidence", ""])
    for t in data["tasks"]:
        if t["artifacts"]:
            lines.append(f"- **{t['id']} ({t['state']})**: " + ", ".join(f"`{a['path']}`" for a in t["artifacts"]))
    return "\n".join(lines) + "\n"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent, help="Project root (defaults to Final Project)")
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("init").add_argument("--seed", type=Path)
    commands.add_parser("status").add_argument("--format", choices=["text", "json", "markdown"], default="text")
    for name in ("show", "packet"):
        commands.add_parser(name).add_argument("task")
    commands.add_parser("add").add_argument("--spec", type=Path, required=True)
    commands.add_parser("snapshot")
    commands.add_parser("check")
    for name in ("claim", "heartbeat", "submit", "review", "block", "release", "recover"):
        command = commands.add_parser(name)
        command.add_argument("task")
        command.add_argument("--agent", required=True)
        if name == "claim":
            command.add_argument("--role", choices=sorted(ROLES), required=True)
        if name == "submit":
            command.add_argument("--artifact", action="append", required=True)
            command.add_argument("--summary", required=True)
        if name == "review":
            command.add_argument("--verdict", choices=["accept", "revise"], required=True)
            command.add_argument("--note", required=True)
        if name in {"block", "release", "recover"}:
            command.add_argument("--reason", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        ledger = Ledger(args.root)
        if args.command == "init":
            result = ledger.init(args.seed)
        elif args.command == "add":
            result = ledger.add(json.loads(args.spec.read_text(encoding="utf-8-sig")))
        elif args.command == "claim":
            result = ledger.claim(args.task, args.agent, args.role)
        elif args.command in {"show", "packet"}:
            result = getattr(ledger, args.command)(args.task)
        elif args.command in {"status", "snapshot", "check"}:
            result = getattr(ledger, args.command)()
            if args.command == "status" and args.format != "json":
                result = markdown(result) if args.format == "markdown" else "\n".join(
                    f"{t['id']:16} {t['state']:8} {t['spec']['role']:12} {t['owner'] or '-':20} {t['spec']['title']}" + (" [stale; recovery is explicit]" if t["stale"] else "") for t in result["tasks"])
        else:
            if args.command == "recover":
                print("Recovery asserts that the coordinator has stopped the prior writer; no process is terminated by this tool.", file=sys.stderr)
            result = ledger.mutate(args.command, args.task, args.agent,
                                   artifacts=getattr(args, "artifact", None), summary=getattr(args, "summary", None),
                                   verdict=getattr(args, "verdict", None), note=getattr(args, "note", None), reason=getattr(args, "reason", None))
        print(result if isinstance(result, str) else json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (HarnessError, OSError, ValueError, sqlite3.Error, RecursionError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
