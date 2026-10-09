# -*- coding: utf-8 -*-
"""Verify the benchmark images: readability, shape, file size and SHA-256.

Covers the representative image
(01_Datasets/benchmark_dataset/usc-sipi_4.2.03_mandrill.tif) and the 20
randomly selected benchmark images
(01_Datasets/benchmark_dataset/random20_images/*.tiff).

Run from anywhere:
    python 03_Preprocessing/preprocessing_scripts/verify_images.py

For a pure-hash check without Python dependencies:
    cd 01_Datasets/benchmark_dataset && shasum -a 256 -c SHA256SUMS.txt
"""
import glob
import hashlib
import os

import cv2

NAMES = {"4.2.01": "Splash", "4.2.03": "Mandrill (a.k.a. Baboon)",
         "4.2.05": "Airplane (F-16)", "4.2.06": "Sailboat on lake",
         "4.2.07": "Peppers"}

HERE = os.path.dirname(os.path.abspath(__file__))
# 03_Preprocessing/preprocessing_scripts/ -> package root
PKG = os.path.abspath(os.path.join(HERE, "..", ".."))
DATASET = os.path.join(PKG, "01_Datasets", "benchmark_dataset")
patterns = [
    os.path.join(DATASET, "usc-sipi_*.tif"),
    os.path.join(DATASET, "random20_images", "*.tiff"),
]

paths = []
for pattern in patterns:
    paths.extend(sorted(glob.glob(pattern)))

for p in paths:
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"{os.path.basename(p):32s} UNREADABLE")
        continue
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    key = os.path.basename(p).replace("usc-sipi_", "").replace(".tif", "") \
        .replace(".tiff", "")
    print(f"{os.path.relpath(p, PKG):58s} {os.path.getsize(p):>8d} B  "
          f"shape={img.shape}  {NAMES.get(key, '?'):22s}  sha256={h}")
