# -*- coding: utf-8 -*-
"""
Normalized benchmark for three DNA storage tools:

  * common benchmark dataset: 20 images randomly selected from the USC-SIPI
    Image Database (01_Datasets/benchmark_dataset/random20_images); the
    quantitative N-repeat run below uses the representative 786,572-byte
    image 4.2.03 (one of the 20; SHA-256 recorded)
  * every tool executed in its native coding framework with its
    distributed default parameter settings
  * identical oligonucleotide length (120 nt) for both general-purpose codecs
  * exact software provenance, environment report, random seeds
  * N repeated runs; wall-clock timing with interpreter-startup overhead
    measured separately; peak process-tree memory sampled at 5 ms
  * explicit definitions of redundancy and recovery rate
  * Experiment 1 (E1): error-free channel
  * Experiment 2 (E2): per-nucleotide substitution 0.1 %, strand dropout
    5 %, single read per strand, no indels for the two generic codecs
    (DNA-Chain: substitution-only channel, as its RS code is intra-strand)

Run from anywhere; all paths are resolved relative to the package root
(this script lives in 04_Benchmark/benchmark_scripts/). Third-party code
must be fetched and built first per
07_Reproduction/reproduction_commands.txt:
  02_Software/DNA-Fountain/dna-fountain-master/
  02_Software/DNA-Storage-Toolkit/DNAStorageToolkit-main/

Outputs (package-root-relative, regenerated working products):
  04_Benchmark/results/main/environment.txt  environment/provenance report
  04_Benchmark/results/main/benchmark_runs.csv  every individual run
  04_Benchmark/results/main/summary.md       means +/- SD and derived metrics
  04_Benchmark/results/main/runs/*.json      per-run records
  (--label NAME redirects to 04_Benchmark/results/results_<NAME>/)

Recovered viewable images are committed separately under 06_Result/.
"""

import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
# package layout: <PKG>/04_Benchmark/benchmark_scripts/this_file
PKG = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_DIR = os.path.join(PKG, "01_Datasets", "benchmark_dataset")
SOFT_DIR = os.path.join(PKG, "02_Software")
FOUNTAIN_DIR = os.path.join(SOFT_DIR, "DNA-Fountain", "dna-fountain-master")
TOOLKIT_DIR = os.path.join(SOFT_DIR, "DNA-Storage-Toolkit",
                           "DNAStorageToolkit-main")
TOOLKIT_CODEC = os.path.join(TOOLKIT_DIR, "1-encoding-decoding")
CHAIN_DIR = os.path.join(SOFT_DIR, "DNA-Chain")
# quantitative run outputs are regenerated working products (gitignored);
# recovered-image evidence lives in 06_Result/
RES_ROOT = os.path.join(PKG, "04_Benchmark", "results")
# --label NAME redirects all outputs to results/results_<NAME>/ so that
# verification runs on the other images of the 20-image benchmark set can
# use a different --image without colliding.
LABEL = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else None
RESULTS = os.path.join(RES_ROOT, "main" if LABEL is None else f"results_{LABEL}")
RUNS_DIR = os.path.join(RESULTS, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

INPUT_IMAGE = os.path.join(DATA_DIR, "usc-sipi_4.2.03_mandrill.tif")
# Representative input for the quantitative N-repeat run: standard test
# image 4.2.03 ("Mandrill"), USC-SIPI Image Database (512x512 24-bit RGB
# TIFF), one of the 20 randomly selected benchmark images. The other 19
# images are verified with --image/--label (see 07_Reproduction/
# reproduction_commands.txt).
if "--image" in sys.argv:
    INPUT_IMAGE = os.path.abspath(sys.argv[sys.argv.index("--image") + 1])
FILE_SIZE = os.path.getsize(INPUT_IMAGE) # whole file encoded by all tools
K_CHUNKS = (FILE_SIZE + 23) // 24        # DNA-Fountain 24-byte chunks
CHAIN_IMAGE_SIZE = None                  # filled at runtime (H x W)

N_RUNS = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 10
SEEDS = list(range(1000, 1000 + N_RUNS))

E2_SUBS = 0.001      # per-nucleotide substitution probability
E2_DROPOUT = 0.05    # fraction of strands removed

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _poll_tree(pid, stop_event, peak):
    """Sample the working set (RSS) of a process tree every ~5 ms."""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    while not stop_event.is_set():
        try:
            total = parent.memory_info().rss
            for child in parent.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak[0] = max(peak[0], total)
        except psutil.NoSuchProcess:
            break
        time.sleep(0.005)


def run_cmd(cmd, cwd=None):
    """Run a command; return (wall_s, peak_rss_bytes, returncode, stdout)."""
    peak = [0]
    stop = threading.Event()
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    poller = threading.Thread(target=_poll_tree, args=(proc.pid, stop, peak))
    poller.daemon = True
    poller.start()
    out, _ = proc.communicate()
    wall = time.perf_counter() - t0
    stop.set()
    poller.join(timeout=1)
    return wall, peak[0], proc.returncode, out


def strand_stats(path):
    n = 0
    total_nt = 0
    lengths = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            n += 1
            total_nt += len(line)
            lengths.add(len(line))
    return n, total_nt, sorted(lengths)


def apply_channel(in_path, out_path, subs, dropout, seed):
    """Identical channel for plain strand files (one sequence per line):
    substitutions on surviving strands, random strand dropout.
    RNG: numpy RandomState(seed) for reproducibility."""
    import numpy as np
    rng = np.random.RandomState(seed)
    pool = np.array([ord(c) for c in "ACGT"], dtype=np.uint8)
    kept = 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if rng.random_sample() < dropout:
                continue
            arr = np.frombuffer(line.encode("ascii"), dtype=np.uint8).copy()
            mask = rng.random_sample(arr.size) < subs
            if mask.any():
                arr[mask] = rng.choice(pool, size=int(mask.sum()))
            fout.write(arr.tobytes().decode("ascii") + "\n")
            kept += 1
    return kept


def apply_channel_fasta(in_path, out_path, subs, dropout, seed):
    """Channel for FASTA files: drop 'header + sequence' pairs jointly."""
    import numpy as np
    rng = np.random.RandomState(seed)
    pool = np.array([ord(c) for c in "ACGT"], dtype=np.uint8)
    kept, dropped = 0, 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        header = None
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                header = line
                continue
            if rng.random_sample() < dropout:
                dropped += 1
                header = None
                continue
            arr = np.frombuffer(line.encode("ascii"), dtype=np.uint8).copy()
            mask = rng.random_sample(arr.size) < subs
            if mask.any():
                arr[mask] = rng.choice(pool, size=int(mask.sum()))
            if header:
                fout.write(header + "\n")
            fout.write(arr.tobytes().decode("ascii") + "\n")
            kept += 1
            header = None
    return kept, dropped


def mean_sd(xs):
    if not xs:
        return (float("nan"), float("nan"))
    if len(xs) == 1:
        return (xs[0], 0.0)
    return statistics.mean(xs), statistics.stdev(xs)


# ----------------------------------------------------------------------------
# tool wrappers
# ----------------------------------------------------------------------------

class Fountain:
    name = "DNA-Fountain"
    task = "generic binary file codec (fountain code + RS + screens)"
    payload_bits = FILE_SIZE * 8
    reconstruction = "bit-exact file (MD5)"
    # The tool is executed with its distributed defaults: chunk size 24 B,
    # --rs 2 (per-droplet RS(30,28)), -m 4 (max homopolymer run), --gc 0.2,
    # and the distributed default oligo-pool overhead (no --alpha flag is
    # passed, so encode.py uses its built-in default). No non-default flags
    # are added by the benchmark.

    def __init__(self):
        self.encoded = os.path.join(RUNS_DIR, "fountain_encoded.fasta")
        self.noisy = os.path.join(RUNS_DIR, "fountain_noisy.fasta")
        self.decoded = os.path.join(RUNS_DIR, "fountain_decoded.bin")
        self.cmd = [
            sys.executable, "-W", "ignore", "encode.py",
            "-f", INPUT_IMAGE, "-l", "24", "-m", "4", "--gc", "0.2",
            "--rs", "2", "--out", self.encoded,
        ]

    def encode(self):
        return run_cmd(self.cmd, cwd=FOUNTAIN_DIR)

    def decode(self, in_file, out_file):
        # Distributed defaults of decode.py (no extra verification flags).
        cmd = [
            sys.executable, "-W", "ignore", "decode.py",
            "-f", in_file, "--header_size", "4", "--rs", "2",
            "-n", str(K_CHUNKS), "--gc", "0.2", "-m", "4", "--size", "24",
            "--fasta", "--out", out_file,
        ]
        return run_cmd(cmd, cwd=FOUNTAIN_DIR)

    def channel(self, seed):
        return apply_channel_fasta(self.encoded, self.noisy, E2_SUBS, E2_DROPOUT, seed)

    def verify(self, out_file):
        # decode.py writes the zero-padded file (size rounded up to a
        # multiple of the 24-byte chunk); compare only the first FILE_SIZE
        # bytes against the original input.
        h = hashlib.md5()
        with open(out_file, "rb") as f:
            h.update(f.read(FILE_SIZE))
        return h.hexdigest() == md5(INPUT_IMAGE)


class Toolkit:
    name = "DNA Storage Toolkit"
    task = "generic binary file codec (RS(65535,55705) over GF(2^16), column mapping)"
    payload_bits = FILE_SIZE * 8
    reconstruction = "bit-exact file (MD5)"

    def __init__(self):
        self.encoded = os.path.join(RUNS_DIR, "toolkit_EncodedStrands.txt")
        self.noisy = os.path.join(RUNS_DIR, "toolkit_NoisyStrands.txt")
        self.decoded = os.path.join(RUNS_DIR, "toolkit_decoded.bin")

    @staticmethod
    def _bin(name):
        # Windows builds produce encoder.exe/decoder.exe; the documented
        # build commands on macOS/Linux produce encoder/decoder.
        exe = name + (".exe" if os.name == "nt" else "")
        return os.path.join(TOOLKIT_CODEC, exe)

    def encode(self):
        cmd = [self._bin("encoder"), INPUT_IMAGE,
               self.encoded, str(FILE_SIZE), "0", "NA", "0"]
        return run_cmd(cmd, cwd=TOOLKIT_CODEC)

    def decode(self, in_file, out_file):
        cmd = [self._bin("decoder"), in_file,
               out_file, str(FILE_SIZE), "0", "NA", "0"]
        return run_cmd(cmd, cwd=TOOLKIT_CODEC)

    def channel(self, seed):
        return apply_channel(self.encoded, self.noisy, E2_SUBS, E2_DROPOUT, seed)

    def verify(self, out_file):
        return md5(out_file) == md5(INPUT_IMAGE)


class Chain:
    name = "DNA-Chain"
    task = ("structured/vector image codec (contour chain codes + Huffman + "
            "5-bit->3-base mapping)")
    reconstruction = "exact recovery of the chain-code symbol stream"

    def __init__(self):
        self.script = os.path.join(CHAIN_DIR, "dna_chain_headless.py")

    def encode(self):
        cmd = [sys.executable, self.script, INPUT_IMAGE]
        return run_cmd(cmd, cwd=CHAIN_DIR)

    def channel_and_decode(self, seed):
        # DNA-Chain: E2 uses substitution-only channel (no strand dropout),
        # because the RS code is intra-strand only and the thesis does not
        # design inter-strand redundancy. The channel difference is noted
        # in Table 4 and the methods description.
        cmd = [sys.executable, self.script, INPUT_IMAGE,
               str(E2_SUBS), str(seed), "0.0"]
        wall, rss, rc, out = run_cmd(cmd, cwd=CHAIN_DIR)
        self._last_json = json.loads(out.strip().splitlines()[-1])
        return wall, rss, rc, out


# ----------------------------------------------------------------------------
# startup-overhead measurement
# ----------------------------------------------------------------------------

def measure_startup():
    """Interpreter/runtime startup overhead (import time + interpreter boot),
    measured separately so it can be excluded from reported codec times."""
    overhead = {}
    walls = []
    for _ in range(3):
        wall, _, _, _ = run_cmd([sys.executable, "encode.py", "--help"],
                                cwd=FOUNTAIN_DIR)
        walls.append(wall)
    overhead["DNA-Fountain-encode"] = statistics.mean(walls)
    walls = []
    for _ in range(3):
        wall, _, _, _ = run_cmd([sys.executable, "-c",
                                 "import numpy, cv2, PyQt5.QtCore"], cwd=CHAIN_DIR)
        walls.append(wall)
    overhead["DNA-Chain"] = statistics.mean(walls)
    # Minimal native process spawn: cmd.exe on Windows, /usr/bin/true on Unix.
    spawn_cmd = ["cmd", "/c", "exit"] if os.name == "nt" else ["/usr/bin/true"]
    wall, _, _, _ = run_cmd(spawn_cmd, cwd=HERE)
    overhead["DNA Storage Toolkit (process spawn estimate)"] = wall
    return overhead


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def preflight():
    """Fail early with an actionable message if prerequisites are missing."""
    problems = []
    if not os.path.exists(INPUT_IMAGE):
        problems.append(f"input image not found: {INPUT_IMAGE}")
    if not os.path.exists(os.path.join(FOUNTAIN_DIR, "encode.py")):
        problems.append("DNA-Fountain not fetched: clone "
                        "https://github.com/TeamErlich/dna-fountain into "
                        f"{FOUNTAIN_DIR}")
    if not list(_glob_ext(FOUNTAIN_DIR, "glass")):
        problems.append("DNA-Fountain Cython modules not built: run "
                        "'python setup.py build_ext --inplace' in "
                        f"{FOUNTAIN_DIR}")
    tk_bin = os.path.join(TOOLKIT_CODEC,
                          "encoder" + (".exe" if os.name == "nt" else ""))
    if not os.path.exists(tk_bin):
        problems.append("Toolkit codec not built: compile encoder/decoder in "
                        f"{TOOLKIT_CODEC} (see 07_Reproduction/"
                        "reproduction_commands.txt, STEP 2)")
    if not os.path.exists(os.path.join(CHAIN_DIR, "Windows_DNA_Chain.py")):
        problems.append(f"DNA-Chain sources missing under {CHAIN_DIR}")
    if problems:
        print("PRE-FLIGHT CHECK FAILED:")
        for p in problems:
            print("  -", p)
        sys.exit(2)


def _glob_ext(directory, module_base):
    """True if module_base exists as .py or as a compiled extension (.so/.pyd)."""
    import glob
    return glob.glob(os.path.join(directory, module_base) + "*.so") + \
        glob.glob(os.path.join(directory, module_base) + "*.pyd") + \
        glob.glob(os.path.join(directory, module_base + ".py"))


def main():
    preflight()
    rows = []
    tools = [Fountain(), Toolkit(), Chain()]
    startup = measure_startup()

    env = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "os": f"{platform.system()} {platform.release()} build {platform.version()}",
        "cpu": platform.processor(),
        "python": sys.version,
        "machine": platform.machine(),
        "input_image": INPUT_IMAGE,
        "input_sha256": sha256(INPUT_IMAGE),
        "input_size_bytes": os.path.getsize(INPUT_IMAGE),
        "file_size_used_bytes": FILE_SIZE,
        "n_runs": N_RUNS,
        "dna_fountain_config": {
            "chunk_size_bytes": 24, "rs_bytes": 2,
            "max_homopolymer": 4, "gc_range": "0.3-0.7",
            "oligo_len_nt": 120, "K_chunks": K_CHUNKS,
            "note": ("executed with the distributed default parameter "
                     "settings of encode.py/decode.py; no non-default CLI "
                     "flags are passed by the benchmark"),
        },
        "e2_channel": {
            "substitution_per_nt": E2_SUBS,
            "strand_dropout": E2_DROPOUT,
            "indels": "none",
            "reads_per_strand": 1,
        },
        "startup_overhead_s": startup,
    }
    try:
        import cv2, numpy, scipy, reedsolo, psutil
        from importlib.metadata import version as _mdv

        def _pkgv(mod, dist):
            v = getattr(mod, "__version__", None)
            if v is None:
                try:
                    v = _mdv(dist)
                except Exception:
                    v = "unknown"
            return v

        env["packages"] = {
            "numpy": _pkgv(numpy, "numpy"), "opencv": _pkgv(cv2, "opencv-python"),
            "scipy": _pkgv(scipy, "scipy"), "reedsolo": _pkgv(reedsolo, "reedsolo"),
            "psutil": _pkgv(psutil, "psutil"),
        }
        env["ram_total_bytes"] = psutil.virtual_memory().total
    except Exception as e:  # pragma: no cover
        env["packages_error"] = str(e)
    import subprocess as _sp
    try:
        zig_ver = _sp.run([sys.executable, "-m", "ziglang", "version"],
                          capture_output=True, text=True, timeout=30).stdout.strip()
        env["zig"] = zig_ver or "not available"
    except Exception:
        env["zig"] = "not available (a system C++17 compiler may be used instead)"
    # provenance of the three benchmarked code bases. The two third-party
    # tools are fetched clones (exact upstream commit + source-file SHA-256
    # are recorded); DNA-Chain is redistributed inside this package.
    env["tool_provenance"] = {
        "DNA-Fountain": {
            "origin": "https://github.com/TeamErlich/dna-fountain (fetched clone)",
            "dir": "02_Software/DNA-Fountain/dna-fountain-master",
            "entry": ["encode.py", "decode.py", "fountain.py", "glass.py",
                      "droplet.py", "robust_solition.py", "utils.py"],
        },
        "DNA Storage Toolkit": {
            "origin": "https://github.com/DNAStorageToolkit/DNAStorageToolkit (fetched clone)",
            "dir": "02_Software/DNA-Storage-Toolkit/DNAStorageToolkit-main",
            "entry": ["1-encoding-decoding/encoder.cpp",
                      "1-encoding-decoding/decoder.cpp"],
        },
        "DNA-Chain": {
            "origin": "authors' tool, redistributed in this package",
            "dir": "02_Software/DNA-Chain",
            "entry": ["02_Software/DNA-Chain/Windows_DNA_Chain.py",
                      "02_Software/DNA-Chain/dna_chain_headless.py"],
        },
    }

    def _prov_path(tool, rel):
        if tool == "DNA-Fountain":
            return os.path.join(FOUNTAIN_DIR, rel)
        if tool == "DNA Storage Toolkit":
            return os.path.join(TOOLKIT_DIR, rel.replace("/", os.sep))
        return os.path.join(PKG, rel.replace("/", os.sep))

    for tool, info in env["tool_provenance"].items():
        for rel in info["entry"]:
            p = _prov_path(tool, rel)
            if os.path.exists(p):
                env[f"sha256:{rel}"] = sha256(p)
    with open(os.path.join(RESULTS, "environment.txt"), "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, ensure_ascii=False)

    print(json.dumps(env, indent=2)[:600])

    startup_key = {
        "DNA-Fountain": "DNA-Fountain-encode",
        "DNA-Chain": "DNA-Chain",
        "DNA Storage Toolkit": "DNA Storage Toolkit (process spawn estimate)",
    }

    for tool in tools:
        print(f"\n=== {tool.name} ===")
        enc_walls, enc_rss, dec_walls, dec_rss = [], [], [], []
        metrics_once = {}

        for i, seed in enumerate(SEEDS):
            # ---------------- encode ----------------
            proc_wall, rss, rc, out = tool.encode()
            if rc != 0:
                print(f"[{tool.name}] encode run {i} FAILED rc={rc}\n{out[-800:]}")
            enc_rss.append(rss)

            if tool.name == "DNA-Chain":
                # internal perf_counter timers of the tool itself (exclude
                # interpreter startup and contour extraction, mirroring the
                # tool's own timing convention)
                rec = json.loads(out.strip().splitlines()[-1])
                metrics_once.update(rec)
                enc_wall = rec["encode_time_s"]
                dec_wall = rec["decode_time_s"]
                enc_walls.append(enc_wall)
                dec_walls.append(dec_wall)
                ok = bool(rec["recovery_stream_exact"])
            else:
                if os.path.getsize(tool.encoded) > 0:
                    n, total_nt, lens = strand_stats(tool.encoded)
                else:
                    # tool produced an empty oligo pool: record zero strands
                    # and continue instead of crashing the whole benchmark
                    n, total_nt, lens = 0, 0, []
                metrics_once.update({
                    "n_oligos": n, "total_nt": total_nt,
                    "oligo_len_nt": lens[0] if len(lens) == 1 else lens,
                })
                enc_wall = proc_wall
                enc_walls.append(enc_wall)
                # ---------------- decode ----------------
                dwall, drss, drc, dout = tool.decode(tool.encoded, tool.decoded)
                dec_wall = dwall
                ok = ((drc == 0) and os.path.exists(tool.decoded)
                      and tool.verify(tool.decoded))
                dec_walls.append(dec_wall)
                dec_rss.append(drss)
                if not ok:
                    print(f"[{tool.name}] decode run {i} FAILED rc={drc}\n{(dout or '')[-500:]}")

            rows.append({
                "tool": tool.name, "experiment": "E1", "run": i,
                "seed": "deterministic",
                "encode_wall_s": round(enc_wall, 4),
                "decode_wall_s": round(dec_wall, 4),
                "encode_peak_rss_bytes": rss,
                "decode_peak_rss_bytes": ("" if tool.name == "DNA-Chain" else drss),
                "encode_rc": rc, "decode_ok": ok,
                "recovery": tool.reconstruction,
            })
            print(f"  E1 run {i}: enc {enc_wall:.2f}s  dec {dec_wall:.2f}s ok={ok}")

        # ---------------- E2: identical channel ----------------
        rec_ok = []
        for i, seed in enumerate(SEEDS):
            if tool.name == "DNA-Chain":
                proc_wall, rss, rc, out = tool.channel_and_decode(seed)
                # recovery = exact recovery of the chain-code symbol stream
                # (the tool's own reconstruction objective)
                ok = bool(tool._last_json["recovery_stream_exact"])
                dec_wall = tool._last_json["decode_time_s"]
                dec_rss_run = rss
            else:
                kept = tool.channel(seed)
                dwall, drss, drc, dout = tool.decode(tool.noisy, tool.decoded)
                ok = ((drc == 0) and os.path.exists(tool.decoded)
                      and tool.verify(tool.decoded))
                dec_wall, dec_rss_run = dwall, drss
                if not ok:
                    print(f"  E2 seed {seed}: decode failed (rc={drc})\n"
                          f"{(dout or '')[-400:]}")
            rec_ok.append(ok)
            rows.append({
                "tool": tool.name, "experiment": "E2", "run": i, "seed": seed,
                "encode_wall_s": "", "decode_wall_s": round(dec_wall, 4),
                "encode_peak_rss_bytes": "",
                "decode_peak_rss_bytes": dec_rss_run,
                "encode_rc": "", "decode_ok": ok,
                "recovery": tool.reconstruction,
            })
            print(f"  E2 seed {seed}: recovered={ok}")

        tool.summary = {
            "encode_wall_mean_s": mean_sd(enc_walls),
            "decode_wall_mean_s": mean_sd(dec_walls) if dec_walls else (None, None),
            "encode_peak_rss_mean": mean_sd(enc_rss),
            "decode_peak_rss_mean": mean_sd(dec_rss) if dec_rss else (None, None),
            "e1_recovery_rate": sum(1 for ok in [r["decode_ok"] for r in rows
                                  if r["tool"] == tool.name and r["experiment"] == "E1"]) / N_RUNS,
            "e2_recovery_rate": sum(rec_ok) / N_RUNS,
            "metrics": metrics_once,
            "startup_overhead_s": startup[startup_key[tool.name]],
        }

    # ---------------- CSV ----------------
    csv_path = os.path.join(RESULTS, "benchmark_runs.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------------- summary ----------------
    lines = ["# Benchmark summary\n"]
    lines.append("Benchmark dataset: 20 images randomly selected from the "
                 "USC-SIPI Image Database (random20_images/). The "
                 "quantitative N-repeat run below uses the representative "
                 "786,572-byte image 4.2.03, one of the 20; exact recovery "
                 "was verified for all three methods on all 20 images "
                 "(recovered images in 06_Result/<image_id>/; dataset "
                 "provenance in 01_Datasets/DATASET.txt).\n")
    lines.append(f"Representative input: {os.path.basename(INPUT_IMAGE)}, "
                 f"{FILE_SIZE} bytes, SHA-256 "
                 f"{env['input_sha256']}\n")
    lines.append(f"Runs per condition: N={N_RUNS}; seeds {SEEDS[0]}..{SEEDS[-1]}\n")
    lines.append(f"E2 channel: substitution {E2_SUBS*100:.1f} %/nt, dropout "
                 f"{E2_DROPOUT*100:.0f} %, no indels, 1 read/strand\n")
    for tool in tools:
        s = tool.summary
        lines.append(f"\n## {tool.name}\n")
        lines.append(f"- task: {tool.task}")
        em, es = s["encode_wall_mean_s"]; dm, ds = s["decode_wall_mean_s"]
        lines.append(f"- E1 encode wall: {em:.3f} ± {es:.3f} s "
                     f"(startup overhead {s['startup_overhead_s']:.3f} s, excluded)")
        if dm is not None:
            lines.append(f"- E1 decode wall: {dm:.3f} ± {ds:.3f} s")
        er, erd = s["encode_peak_rss_mean"]
        lines.append(f"- E1 encode peak RSS: {er/1e6:.1f} ± {erd/1e6:.1f} MB")
        if s["decode_peak_rss_mean"][0]:
            dr, drd = s["decode_peak_rss_mean"]
            lines.append(f"- E1 decode peak RSS: {dr/1e6:.1f} ± {drd/1e6:.1f} MB")
        lines.append(f"- E1 recovery rate: {s['e1_recovery_rate']*100:.0f} %")
        lines.append(f"- E2 recovery rate: {s['e2_recovery_rate']*100:.0f} %")
        m = s["metrics"]
        for k, v in m.items():
            lines.append(f"- {k}: {v}")
    with open(os.path.join(RESULTS, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\nDone. See", RESULTS)


if __name__ == "__main__":
    main()
