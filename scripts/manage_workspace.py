"""Offline backup/restore without overwriting live data."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forestmind.monitoring.backup import backup_workspace, restore_workspace, verify_backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    backup = commands.add_parser("backup")
    backup.add_argument("--db", default="data/forestmind.db")
    backup.add_argument("--destination", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--source", required=True)
    restore = commands.add_parser("restore")
    restore.add_argument("--source", required=True)
    restore.add_argument("--db", required=True)
    args = parser.parse_args()
    try:
        result = backup_workspace(args.db, args.destination) if args.command == "backup" else (
            restore_workspace(args.source, args.db) if args.command == "restore" else verify_backup(args.source))
    except (OSError, ValueError) as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps({"status": "verified", "command": args.command, "files": len(result["files"])}))


if __name__ == "__main__":
    main()
