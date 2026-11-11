#!/usr/bin/env python3
"""Rewrite compile_commands.json to use relative paths.

Replaces the remote project root with relative paths so the file
works on the local machine (via sshfs).

Usage: python3 script/rel_compile_commands.py [remote_root]
  remote_root defaults to /home/boyang/CDIxx
"""
import json
import os
import sys


def main():
    remote_root = sys.argv[1] if len(sys.argv) > 1 else "/home/boyang/CDIxx"
    path = os.path.join("build", "compile_commands.json")

    if not os.path.exists(path):
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    if not data:
        return

    remote_build = remote_root + "/build"

    for entry in data:
        entry["directory"] = "."
        # file: make relative to build/
        entry["file"] = entry["file"].replace(remote_root + "/", "../")
        # command: replace remote root with ../ (relative from build/)
        entry["command"] = entry["command"].replace(remote_root + "/", "../")
        # output field if present
        if "output" in entry:
            entry["output"] = entry["output"].replace(remote_build, ".")

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    print(f"OK: replaced '{remote_root}/' with '../' in {path}")


if __name__ == "__main__":
    main()
