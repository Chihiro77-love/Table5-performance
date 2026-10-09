# -*- coding: utf-8 -*-
"""
Recover the decoded output of each codec into a viewable image file and
verify bit-exactness against the original input.

Inputs:  04_Benchmark/results/main/runs/ produced by run_benchmark.py
Outputs (04_Benchmark/results/recovered_mandrill/, regenerated; the
committed 20-image evidence is in 06_Result/, see recover_all.py):
  00_original_mandrill.tif        original input (for comparison)
  01_DNA-Fountain_recovered.tif   decoded file from fountain_decoded.bin
  02_DNA-Storage-Toolkit_recovered.tif  decoded file from toolkit_decoded.bin
  03_DNA-Chain_recovered_E1.png   reconstructed binary contour image (E1)
  04_DNA-Chain_recovered_E2.png   reconstructed binary contour image (E2, sub-only)
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
RUNS = os.path.join(PKG, "04_Benchmark", "results", "main", "runs")
OUT = os.path.join(PKG, "04_Benchmark", "results", "recovered_mandrill")
os.makedirs(OUT, exist_ok=True)

IMG = os.path.join(DATA, "usc-sipi_4.2.03_mandrill.tif")
FILE_SIZE = os.path.getsize(IMG)


def md5_bytes(b):
    return hashlib.md5(b).hexdigest()


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


orig_md5 = md5_file(IMG)
print("original MD5:", orig_md5, " size:", FILE_SIZE)

# 0. copy original for side-by-side comparison
shutil.copyfile(IMG, os.path.join(OUT, "00_original_mandrill.tif"))

# 1. DNA-Fountain: decoded file is zero-padded to a multiple of 24 B; take first FILE_SIZE bytes
fb = os.path.join(RUNS, "fountain_decoded.bin")
with open(fb, "rb") as f:
    fdata = f.read(FILE_SIZE)
fout = os.path.join(OUT, "01_DNA-Fountain_recovered.tif")
with open(fout, "wb") as f:
    f.write(fdata)
print("DNA-Fountain recovered MD5:", md5_bytes(fdata),
      " match:", md5_bytes(fdata) == orig_md5)

# 2. DNA Storage Toolkit: decoded file is exactly the original size
tb = os.path.join(RUNS, "toolkit_decoded.bin")
with open(tb, "rb") as f:
    tdata = f.read()
tout = os.path.join(OUT, "02_DNA-Storage-Toolkit_recovered.tif")
with open(tout, "wb") as f:
    f.write(tdata)
print("Toolkit recovered MD5:   ", md5_bytes(tdata),
      " match:", md5_bytes(tdata) == orig_md5)

# 3. DNA-Chain: reconstruct the binary contour image through the full RS pipeline
CHAIN_PATH = os.path.join(CHAIN_DIR, "Windows_DNA_Chain.py")
spec = importlib.util.spec_from_file_location("dna_chain", CHAIN_PATH)
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)


def chain_recover(sub_rate, seed, out_png):
    chaincodes, start_points = dc.image_to_chaincode_with_start(IMG)
    gray = cv2.imread(IMG, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    image_size = binary_image.shape

    # encode
    encoded_cc, huff = dc.encode_chaincodes_with_huffman(chaincodes)
    pad = (5 - len(encoded_cc) % 5) % 5
    padded = encoded_cc + "0" * pad
    bases = ""
    for i in range(0, len(padded), 5):
        bases += dc.binary_to_three_bases(padded[i:i+5])
    oligos, file_id, _ = dc.encode_with_rs(bases, IMG)

    # channel
    noisy = list(oligos)
    if sub_rate > 0:
        rng = np.random.RandomState(seed)
        noisy = []
        for ol in oligos:
            arr = np.frombuffer(ol.encode("ascii"), dtype=np.uint8).copy()
            mask = rng.random_sample(arr.size) < sub_rate
            pool = np.array([ord(c) for c in "ACGT"], dtype=np.uint8)
            arr[mask] = rng.choice(pool, size=int(mask.sum()))
            noisy.append(arr.tobytes().decode("ascii"))

    # decode
    corrected = dc.decode_with_rs(noisy, len(encoded_cc), file_id)
    payload_nt = ((len(encoded_cc) + 4) // 5) * 3
    cb = corrected[:payload_nt]
    bits = ""
    for i in range(0, len(cb), 3):
        g = cb[i:i+3]
        if len(g) == 3:
            b0 = dc.base_to_binary(g[0]); b1 = dc.base_to_binary(g[1])
            b2 = "0" if (g[1] == g[2] and g[1] in "AG") or (g[1] != g[2] and g[2] in "AG") else "1"
            bits += b0 + b1 + b2
    bits = bits[:len(encoded_cc)].ljust(len(encoded_cc), "0")
    dec_cc = dc.decode_huffman_to_chaincodes(bits, huff)

    grouped, idx = [], 0
    for cc_ in chaincodes:
        grouped.append(dec_cc[idx:idx+len(cc_)])
        idx += len(cc_)
    restored = dc.chaincode_to_image_with_start(grouped, start_points, image_size)
    cv2.imwrite(out_png, restored)
    match = float(np.mean(restored == binary_image))
    flat_orig = [c for cc_ in chaincodes for c in cc_]
    exact = (dec_cc == flat_orig)
    print(f"DNA-Chain {os.path.basename(out_png)}: stream_exact={exact} "
          f"pixel_match={match:.4f} saved={os.path.getsize(out_png)} B")
    return exact


chain_recover(0.0, 0, os.path.join(OUT, "03_DNA-Chain_recovered_E1.png"))
chain_recover(0.001, 1000, os.path.join(OUT, "04_DNA-Chain_recovered_E2_sub0.1pct.png"))

print("\nAll recovered files:")
for fn in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, fn)
    print(f"  {os.path.getsize(p):>10,} B  {p}")
