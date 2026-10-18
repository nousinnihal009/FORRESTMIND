"""Offline workspace backup/restore including immutable source artifacts."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


@contextmanager
def offline_lease(database):
    """Use the worker's OS lock. Backups never race with scheduled acquisition."""
    root = database.parent / (database.stem + "-acquisition")
    root.mkdir(parents=True, exist_ok=True)
    path = root / "worker.lock"
    stream = path.open("a+b")
    try:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError("Stop the local service before full workspace backup or restore") from None
        yield
    finally:
        stream.close()


def backup_workspace(database, destination):
    database, destination = Path(database).resolve(), Path(destination).resolve()
    if not database.is_file():
        raise ValueError("Database does not exist")
    owned = [database.parent / (database.stem + suffix) for suffix in ("-acquisition", "-workflow")]
    if destination == database.parent or any(destination == root or root in destination.parents for root in owned):
        raise ValueError("Backup destination must be outside the acquisition and workflow directories")
    # Require a fresh destination; never replace a previous backup.
    destination.mkdir(parents=True, exist_ok=False)
    with offline_lease(database):
        files = []
        for root in owned:
            if root.exists():
                for path in root.rglob("*"):
                    if path.is_symlink():
                        raise ValueError("Workspace backup does not follow symbolic links")
                    if path.is_file() and path.name != "worker.lock":
                        files.append(path)
        if sum(path.stat().st_size for path in files) + database.stat().st_size > 3 * 1024**3:
            raise ValueError("Workspace exceeds the 3 GiB backup limit; archive a reviewed study separately")
        db_copy = destination / database.name
        with sqlite3.connect(database) as source, sqlite3.connect(db_copy) as target:
            source.backup(target)
            if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Database backup integrity check failed")
        for path in files:
            copy = destination / path.relative_to(database.parent)
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, copy)
        entries = [{"path": path.relative_to(destination).as_posix(), "size": path.stat().st_size, "sha256": sha256(path)}
                   for path in sorted(destination.rglob("*")) if path.is_file()]
        manifest = {"kind": "forestmind_workspace_backup", "schema_version": 1,
                    "created_at": datetime.now(timezone.utc).isoformat(), "database": str(database),
                    "files": entries, "restore_policy": "Restore to the original database path; absolute provenance references remain valid."}
        (destination / "backup-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_backup(source):
    source = Path(source).resolve()
    manifest_path = source / "backup-manifest.json"
    if manifest_path.stat().st_size > 4 * 1024**2:
        raise ValueError("Backup manifest too large")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "forestmind_workspace_backup" or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported backup manifest")
    database = Path(manifest["database"])
    allowed = {database.name, database.stem + "-acquisition", database.stem + "-workflow"}
    seen = set()
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        path = (source / relative).resolve()
        if (relative.is_absolute() or not relative.parts or relative.parts[0] not in allowed or ".." in relative.parts
                or source not in path.parents or entry["path"] in seen or path.is_symlink()):
            raise ValueError("Unsafe or duplicated backup path")
        seen.add(entry["path"])
        if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
            raise ValueError("Backup file integrity mismatch: " + entry["path"])
    if database.name not in seen:
        raise ValueError("Backup is missing its database")
    with sqlite3.connect(f"file:{(source / database.name).as_posix()}?mode=ro", uri=True) as db:
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Backup database failed integrity check")
    return manifest


def restore_workspace(source, database):
    source, database = Path(source).resolve(), Path(database).resolve()
    manifest = verify_backup(source)
    if Path(manifest["database"]).resolve() != database:
        raise ValueError("Restore must use the original database path to preserve absolute acquisition provenance")
    if database.exists():
        raise FileExistsError("Move the current workspace to a safe recovery location before restoring; existing data is never overwritten")
    with offline_lease(database):
        # Validate all targets before copying anything.
        for entry in manifest["files"]:
            target = database.parent / entry["path"]
            if target.exists():
                raise FileExistsError("Restore target exists: " + str(target))
        for entry in manifest["files"]:
            target = database.parent / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / entry["path"], target)
    return manifest
