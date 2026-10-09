# -*- coding: utf-8 -*-
"""Augment 04_Benchmark/results/main/environment.txt with SHA-256 hashes for
EVERY executed source file of all three tools, so that the "pinned by
SHA-256" provenance claim is true. Run once after run_benchmark.py; it does
not re-run the benchmark.

Usage:
    python add_source_hashes.py                       # default: results/main
    python add_source_hashes.py 04_Benchmark/results/results_<label>
"""
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", ".."))
SOFT = os.path.join(PKG, "02_Software")
FOUNTAIN_DIR = os.path.join(SOFT, "DNA-Fountain", "dna-fountain-master")
TOOLKIT_DIR = os.path.join(SOFT, "DNA-Storage-Toolkit", "DNAStorageToolkit-main")
RES = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 \
    else os.path.join(PKG, "04_Benchmark", "results", "main")
ENV_PATH = os.path.join(RES, "environment.txt")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


with open(ENV_PATH, encoding="utf-8") as f:
    env = json.load(f)

prov = env.get("tool_provenance", {})
source_hashes = {}


def collect(paths):
    out = []
    for p in paths:
        if os.path.exists(p):
            rel = os.path.relpath(p, PKG).replace(os.sep, "/")
            out.append({"file": rel, "sha256": sha256(p)})
    return out


for tool, meta in prov.items():
    if tool == "DNA-Fountain":
        # Python 3 sources (.py) and Cython sources (.pyx) executed by the
        # tool; generated .c/.so build artifacts are deliberately excluded.
        paths = sorted(glob.glob(os.path.join(FOUNTAIN_DIR, "*.py")) +
                       glob.glob(os.path.join(FOUNTAIN_DIR, "*.pyx")))
    elif tool == "DNA Storage Toolkit":
        codec = os.path.join(TOOLKIT_DIR, "1-encoding-decoding")
        paths = sorted(glob.glob(os.path.join(codec, "*.cpp")) +
                       glob.glob(os.path.join(codec, "*.h")) +
                       glob.glob(os.path.join(codec, "*.hpp")))
    else:
        paths = [os.path.join(PKG, f.replace("/", os.sep))
                 for f in meta.get("entry", [])]
    hashed = collect(paths)
    meta["entry_hashes"] = hashed
    for fh in hashed:
        source_hashes[f"{tool}:{fh['file']}"] = fh["sha256"]

env["source_file_hashes"] = source_hashes

with open(ENV_PATH, "w", encoding="utf-8") as f:
    json.dump(env, f, indent=2, ensure_ascii=False)

print(f"hashed {len(source_hashes)} source files -> {ENV_PATH}")
for k, v in sorted(source_hashes.items()):
    print(f"  {v[:16]}...  {k}")
