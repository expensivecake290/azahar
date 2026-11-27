#!/usr/bin/env python3
"""Copy the tracked working tree and initialized submodules without Git metadata."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    paths = subprocess.check_output(["git", "ls-files", "--recurse-submodules", "-z"]).split(b"\0")
    for name in filter(None, paths):
        source = Path(os.fsdecode(name))
        target = args.destination / source
        if source.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_symlink():
                target.symlink_to(source.readlink())
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    print(f"Snapshot created at {args.destination}")


if __name__ == "__main__":
    main()
