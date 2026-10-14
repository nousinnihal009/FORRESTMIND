"""Small transactional SQLite store for local single-user deployments."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS regions (
                    id TEXT PRIMARY KEY, metadata TEXT NOT NULL, observations TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS analyses (
                    region_id TEXT PRIMARY KEY REFERENCES regions(id), result TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY, region_id TEXT NOT NULL REFERENCES regions(id), payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS scenarios (
                    id TEXT PRIMARY KEY, region_id TEXT NOT NULL REFERENCES regions(id), payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS model_runs (
                    id TEXT PRIMARY KEY, region_id TEXT NOT NULL REFERENCES regions(id), payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS model_runs_region ON model_runs(region_id);
            """)

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(str(self.path), timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _json(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)

    def list_regions(self) -> list[dict]:
        with self.connection() as connection:
            return [json.loads(row["metadata"]) for row in connection.execute("SELECT metadata FROM regions ORDER BY rowid")]

    def region(self, region_id: str) -> tuple[dict, list[dict]] | None:
        with self.connection() as connection:
            row = connection.execute("SELECT metadata, observations FROM regions WHERE id=?", (region_id,)).fetchone()
        return (json.loads(row["metadata"]), json.loads(row["observations"])) if row else None

    def put_region(self, region: dict, observations: list[dict], analysis: dict):
        with self.connection() as connection:
            connection.execute("INSERT INTO regions (id,metadata,observations) VALUES (?,?,?)",
                               (region["id"], self._json(region), self._json(observations)))
            self._save_analysis(connection, analysis)

    def _save_analysis(self, connection, analysis):
        connection.execute("INSERT OR REPLACE INTO analyses (region_id,result) VALUES (?,?)",
                           (analysis["id"], self._json(analysis)))
        connection.execute("DELETE FROM alerts WHERE region_id=?", (analysis["id"],))
        connection.executemany("INSERT INTO alerts (id,region_id,payload) VALUES (?,?,?)", [
            (alert["id"], analysis["id"], self._json(alert)) for alert in analysis["alerts"]
        ])

    def save_analysis(self, analysis: dict, job: dict):
        with self.connection() as connection:
            self._save_analysis(connection, analysis)
            connection.execute("INSERT OR REPLACE INTO jobs (id,payload) VALUES (?,?)", (job["id"], self._json(job)))

    def save_job(self, job: dict):
        with self.connection() as connection:
            connection.execute("INSERT OR REPLACE INTO jobs (id,payload) VALUES (?,?)", (job["id"], self._json(job)))

    def recover_jobs(self, timestamp: str):
        """Single-process service: incomplete jobs cannot survive a process restart."""
        with self.connection() as connection:
            rows = connection.execute("SELECT id,payload FROM jobs").fetchall()
            for row in rows:
                job = json.loads(row["payload"])
                if job["status"] in {"pending", "running"}:
                    job.update({"status": "failed", "completed_at": timestamp,
                                "error": "The local service restarted before this analysis completed. Run the analysis again."})
                    connection.execute("UPDATE jobs SET payload=? WHERE id=?", (self._json(job), row["id"]))

    def analysis(self, region_id: str) -> dict | None:
        with self.connection() as connection:
            row = connection.execute("SELECT result FROM analyses WHERE region_id=?", (region_id,)).fetchone()
        return json.loads(row["result"]) if row else None

    def job(self, job_id: str) -> dict | None:
        with self.connection() as connection:
            row = connection.execute("SELECT payload FROM jobs WHERE id=?", (job_id,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def save_scenario(self, scenario: dict):
        with self.connection() as connection:
            connection.execute("INSERT INTO scenarios (id,region_id,payload) VALUES (?,?,?)",
                               (scenario["id"], scenario["region_id"], self._json(scenario)))

    def scenario(self, scenario_id: str) -> dict | None:
        with self.connection() as connection:
            row = connection.execute("SELECT payload FROM scenarios WHERE id=?", (scenario_id,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def list_alerts(self) -> list[dict]:
        with self.connection() as connection:
            return [{**json.loads(row["payload"]), "region_id": row["region_id"]}
                    for row in connection.execute("SELECT region_id,payload FROM alerts ORDER BY rowid")]

    def save_model_run(self, region_id: str, run: dict):
        with self.connection() as connection:
            connection.execute("INSERT INTO model_runs (id,region_id,payload) VALUES (?,?,?)",
                               (run["id"], region_id, self._json(run)))

    def list_model_runs(self, region_id: str) -> list[dict]:
        with self.connection() as connection:
            return [json.loads(row["payload"]) for row in connection.execute(
                "SELECT payload FROM model_runs WHERE region_id=? ORDER BY rowid DESC", (region_id,))]

    def model_run(self, region_id: str, run_id: str) -> dict | None:
        with self.connection() as connection:
            row = connection.execute("SELECT payload FROM model_runs WHERE region_id=? AND id=?",
                                     (region_id, run_id)).fetchone()
        return json.loads(row["payload"]) if row else None
