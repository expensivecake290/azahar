#!/usr/bin/env python3
"""Capture baseline file hashes and compare the Revision 11 ownership inventory."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def git(*args):
    return subprocess.check_output(["git", *args])


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec_extract", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spec = "\n".join(path.read_text() for path in sorted(args.spec_extract.glob("spec-*.txt")))
    # Some hashed entries cross PDF text windows or page boundaries.
    flat_spec = re.sub(r"<PARSED TEXT FOR PAGE: [^>]+>.*?\n", "", spec)
    flat_spec = re.sub(r"^METALIRIS / NATIVE METAL 4[^\n]*", "", flat_spec, flags=re.MULTILINE)
    flat_spec = re.sub(
        r"(SHA-256 [a-f0-9]{60,63})\n +([a-f0-9]{1,4})(?=\n)",
        r"\1\2", flat_spec,
    )
    references = re.findall(r"^ +([^\s|]+) \| (\d+) lines \| SHA-256\s+([a-f0-9]{64})\b",
                            flat_spec, flags=re.MULTILINE)
    reference_count = len(re.findall(r"^ +[^\s|]+ \| \d+ lines \| SHA-256", spec, re.MULTILINE))
    if len(references) != reference_count or not references:
        raise RuntimeError(f"extracted {len(references)} of {reference_count} reference hashes")

    details = re.findall(r"^ Phase (\d{3}) \| ([^\n]+)\n Exclusive production owner: (\S+)",
                         spec, re.MULTILINE)
    index = re.findall(r"^ (\d{3}) (\S+)\s*$", spec, re.MULTILINE)
    actual_index = [(phase, file) for phase, _, file in details]
    if [phase for phase, _ in index] != [f"{i:03}" for i in range(1, 92)] or index != actual_index:
        raise RuntimeError("the ordered phase index and detail owners do not match")
    if len({path for _, path in index}) != len(index):
        raise RuntimeError("a file has multiple phase owners")

    tracked = git("ls-files", "--recurse-submodules", "-z").decode().split("\0")
    with (args.output / "source-sha256.tsv").open("w") as manifest:
        manifest.write("sha256\tkind\tpath\n")
        for name in dict.fromkeys(filter(None, tracked)):
            path = Path(name)
            if path.is_symlink():
                checksum, kind = sha256(bytes(path.readlink())), "symlink-target"
            elif path.is_file():
                checksum, kind = sha256(path.read_bytes()), "file"
            else:
                checksum, kind = "-", "missing"
            manifest.write(f"{checksum}\t{kind}\t{name}\n")

    files = []
    for name, expected_lines, expected_hash in sorted(set(references)):
        path = Path(name)
        data = path.read_bytes() if path.is_file() else None
        actual_hash = sha256(data) if data is not None else None
        files.append({
            "path": name,
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "expected_lines": int(expected_lines),
            "actual_lines": len(data.splitlines()) if data is not None else None,
            "matches_sha256": expected_hash == actual_hash,
        })
    write_json(args.output / "revision11-fingerprints.json", files)
    write_json(args.output / "ownership.json", [
        {"phase": phase, "file": file, "exists_in_checkout": Path(file).is_file()}
        for phase, file in index
    ])
    (args.output / "submodules.txt").write_bytes(git("submodule", "status", "--recursive"))
    (args.output / "initial-diff.txt").write_bytes(git("status", "--short", "--untracked-files=no"))
    (args.output / "baseline-head.txt").write_bytes(git("log", "-1", "--format=%H%n%s"))
    print(f"{len(index)} unique owners, {len(files)} unique fingerprints, "
          f"{sum(not item['matches_sha256'] for item in files)} mismatches, "
          f"{len(set(filter(None, tracked)))} tracked paths")


if __name__ == "__main__":
    main()
