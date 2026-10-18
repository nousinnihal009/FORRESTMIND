import json
from pathlib import Path

import pytest

from forestmind.api.store import Store
from forestmind.monitoring.backup import backup_workspace, restore_workspace, verify_backup


def test_full_workspace_restore_and_hash_failure(tmp_path):
    database = tmp_path / "study.db"
    store = Store(database)
    store.save_job({"id": "job-preserved", "status": "completed"})
    acquisition = tmp_path / "study-acquisition"
    acquisition.mkdir()
    raster = acquisition / "source.npz"
    raster.write_bytes(b"test-only-cache")
    workflow = tmp_path / "study-workflow"
    workflow.mkdir()
    (workflow / "source.json").write_text(json.dumps({"source": str(raster)}))
    backup = tmp_path / "safe-backup"
    manifest = backup_workspace(database, backup)
    assert len(manifest["files"]) == 3
    assert verify_backup(backup) == manifest
    with pytest.raises(FileExistsError):
        restore_workspace(backup, database)
    # Preserve the generated original workspace before simulating recovery.
    database.rename(tmp_path / "original.db")
    acquisition.rename(tmp_path / "original-acquisition")
    workflow.rename(tmp_path / "original-workflow")
    restore_workspace(backup, database)
    assert Store(database).job("job-preserved")["status"] == "completed"
    assert json.loads((workflow / "source.json").read_text())["source"] == str(raster)
    assert raster.read_bytes() == b"test-only-cache"
    (backup / "study-acquisition" / "source.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_backup(backup)


def test_restore_rejects_traversal_and_new_location(tmp_path):
    database = tmp_path / "study.db"
    Store(database)
    destination = tmp_path / "backup"
    manifest = backup_workspace(database, destination)
    with pytest.raises(ValueError, match="original database path"):
        restore_workspace(destination, tmp_path / "other.db")
    manifest["files"].append({"path": "../outside", "sha256": "0" * 64, "size": 0})
    (destination / "backup-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsafe"):
        verify_backup(destination)
