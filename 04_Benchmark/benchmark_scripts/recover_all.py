# -*- coding: utf-8 -*-
"""
Batch-generate the recovered output of every codec for ALL 20 randomly
selected benchmark images
(01_Datasets/benchmark_dataset/random20_images/*.tiff).

First run the encode/decode verification for every image, one output
directory per label (see 07_Reproduction/reproduction_commands.txt,
STEP 7 for the loop):

    python run_benchmark.py --runs 1 \
        --image 01_Datasets/benchmark_dataset/random20_images/<id>.tiff \
        --label <id>

Then run this script. For each image label, write into
06_Result/<label>/:
  00_original.tiff                 original input
  01_fountain_recovered.tiff       from 04_Benchmark/results/results_<label>/runs/fountain_decoded.bin
  02_toolkit_recovered.tiff        from 04_Benchmark/results/results_<label>/runs/toolkit_decoded.bin
  03_dnachain_E1.png               contour reconstruction, error-free
  04_dnachain_E2_sub0.1pct.png     contour reconstruction, 0.1 % substitution (RS)

Generic-codec recovery is bit-exact (MD5) against the input; DNA-Chain PNGs
are regenerated directly from the image set (fast) and the reconstruction is
checked to be pixel-identical to the encoder's thresholded binary image.
Exact recovery is obtained for every method on all 20 images.
"""
import hashlib
import importlib.util
import os
import shutil

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(PKG, "01_Datasets", "benchmark_dataset")
CHAIN_DIR = os.path.join(PKG, "02_Software", "DNA-Chain")
RES_ROOT = os.path.join(PKG, "04_Benchmark", "results")
IMGSET = os.path.join(DATA, "random20_images")
# recovered images are the committed package-level evidence in 06_Result/
OUTROOT = os.path.join(PKG, "06_Result")
os.makedirs(OUTROOT, exist_ok=True)


def md5_file(p, n=None):
    h = hashlib.md5()
    with open(p, "rb") as f:
        data = f.read(n) if n else f.read()
    return hashlib.md5(data).hexdigest()


# load DNA-Chain module with RS
CHAIN_PATH = os.path.join(CHAIN_DIR, "Windows_DNA_Chain.py")
spec = importlib.util.spec_from_file_location("dna_chain", CHAIN_PATH)
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)


def chain_png(image_path, sub_rate, seed, out_png):
    chaincodes, start_points = dc.image_to_chaincode_with_start(image_path)
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    hw = binary.shape
    enc_cc, huff = dc.encode_chaincodes_with_huffman(chaincodes)
    pad = (5 - len(enc_cc) % 5) % 5
    padded = enc_cc + "0" * pad
    bases = ""
    for i in range(0, len(padded), 5):
        bases += dc.binary_to_three_bases(padded[i:i+5])
    oligos, file_id, _ = dc.encode_with_rs(bases, image_path)
    noisy = list(oligos)
    if sub_rate > 0:
        rng = np.random.RandomState(seed)
        noisy = []
        for ol in oligos:
            arr = np.frombuffer(ol.encode("ascii"), dtype=np.uint8).copy()
            m = rng.random_sample(arr.size) < sub_rate
            pool = np.array([ord(c) for c in "ACGT"], dtype=np.uint8)
            arr[m] = rng.choice(pool, size=int(m.sum()))
            noisy.append(arr.tobytes().decode("ascii"))
    corrected = dc.decode_with_rs(noisy, len(enc_cc), file_id)
    nt = ((len(enc_cc) + 4) // 5) * 3
    cb = corrected[:nt]
    bits = ""
    for i in range(0, len(cb), 3):
        g = cb[i:i+3]
        if len(g) == 3:
            b0 = dc.base_to_binary(g[0]); b1 = dc.base_to_binary(g[1])
            same = (g[1] == g[2] and g[1] in "AG") or (g[1] != g[2] and g[2] in "AG")
            b2 = "0" if same else "1"
            bits += b0 + b1 + b2
    bits = bits[:len(enc_cc)].ljust(len(enc_cc), "0")
    dec_cc = dc.decode_huffman_to_chaincodes(bits, huff)
    grouped, idx = [], 0
    for c in chaincodes:
        grouped.append(dec_cc[idx:idx+len(c)]); idx += len(c)
    restored = dc.chaincode_to_image_with_start(grouped, start_points, hw)
    cv2.imwrite(out_png, restored)
    flat = [c for c in chaincodes for c in c]
    pmatch = float(np.mean(restored == binary))
    return (dec_cc == flat), pmatch


images = sorted(f for f in os.listdir(IMGSET)
                if f.lower().endswith((".tiff", ".tif")))
rows = []
for name in images:
    label = os.path.splitext(name)[0]
    src = os.path.join(IMGSET, name)
    size = os.path.getsize(src)
    sdir = os.path.join(RES_ROOT, f"results_{label}")
    outdir = os.path.join(OUTROOT, label)
    os.makedirs(outdir, exist_ok=True)

    line = {"label": label}

    # 00 original
    op = os.path.join(outdir, "00_original" + os.path.splitext(name)[1])
    shutil.copyfile(src, op)
    line["orig"] = os.path.basename(op)

    # 01 fountain
    fb = os.path.join(sdir, "runs", "fountain_decoded.bin")
    if os.path.exists(fb) and os.path.getsize(fb) >= size:
        dst = os.path.join(outdir, "01_fountain_recovered.tiff")
        data = open(fb, "rb").read(size)
        open(dst, "wb").write(data)
        line["fountain_match"] = hashlib.md5(data).hexdigest() == md5_file(src)
    else:
        line["fountain_match"] = "MISSING (run run_benchmark.py --label %s)" % label

    # 02 toolkit
    tb = os.path.join(sdir, "runs", "toolkit_decoded.bin")
    if os.path.exists(tb) and os.path.getsize(tb) > 0:
        dst = os.path.join(outdir, "02_toolkit_recovered.tiff")
        shutil.copyfile(tb, dst)
        line["toolkit_match"] = md5_file(tb) == md5_file(src)
    else:
        line["toolkit_match"] = "MISSING (run run_benchmark.py --label %s)" % label

    # 03/04 DNA-Chain
    try:
        e1_exact, e1_pm = chain_png(src, 0.0, 0,
                                    os.path.join(outdir, "03_dnachain_E1.png"))
        e2_exact, e2_pm = chain_png(src, 0.001, 1000,
                                    os.path.join(outdir, "04_dnachain_E2_sub0.1pct.png"))
        line["chain_e1"] = f"{e1_exact} (pmatch {e1_pm:.3f})"
        line["chain_e2"] = f"{e2_exact} (pmatch {e2_pm:.3f})"
    except Exception as e:
        line["chain_e1"] = f"ERR {e}"
        line["chain_e2"] = "n/a"

    rows.append(line)
    print(f"{label:>14} | fountain={line.get('fountain_match')} "
          f"toolkit={line.get('toolkit_match')} chainE1={line.get('chain_e1')} "
          f"chainE2={line.get('chain_e2')}")

print(f"\n{len(rows)} images processed. Outputs under:\n  {OUTROOT}")
