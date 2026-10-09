# -*- coding: utf-8 -*-
"""
Random sampling record for the 20-image benchmark subset.

20 test images are randomly drawn from the 39-image USC-SIPI misc volume
(https://sipi.usc.edu/database/misc.zip) using a deterministic pseudo-random
number generator with a fixed seed, so the selection is fully reproducible.

Sampling algorithm:
  - PRNG:   numpy.random.RandomState (Mersenne Twister, MT19937)
  - Method: choice(39, 20, replace=False)  -- random subset without
            replacement, each image selected at most once
  - Seed:   42  -- fixed for exact reproducibility

Prerequisite (full 39-image pool; see 01_Datasets/DATASET.txt):
  download https://sipi.usc.edu/database/misc.zip and unpack it so that the
  39 .tiff files are at
      01_Datasets/benchmark_dataset/misc_raw/misc/*.tiff

Usage:
    python sample_selected20.py            # verify and print summary
    python sample_selected20.py --copy DIR # copy the 20 files into DIR

The script additionally verifies that the selected set is exactly the 20
images redistributed in 01_Datasets/benchmark_dataset/random20_images/.
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", ".."))
DATASET = os.path.join(PKG, "01_Datasets", "benchmark_dataset")
MISC = os.path.join(DATASET, "misc_raw", "misc")
SELECTED_DIR = os.path.join(DATASET, "random20_images")

# ---------------------------------------------------------------------------
# Step 1: enumerate the 39 candidate images (sorted for stable indexing)
# ---------------------------------------------------------------------------
if not os.path.isdir(MISC):
    print(f"The 39-image pool is not present at:\n  {MISC}\n")
    print("Download https://sipi.usc.edu/database/misc.zip and unpack the")
    print("inner 'misc' folder to that path (see 01_Datasets/DATASET.txt).")
    print("The already-selected 20 images are available in:")
    print(f"  {SELECTED_DIR}")
    sys.exit(2)

ALL_IMGS = sorted(f for f in os.listdir(MISC) if f.endswith(".tiff"))
assert len(ALL_IMGS) == 39, f"expected 39 candidates, found {len(ALL_IMGS)}"
print(f"Candidate pool: {len(ALL_IMGS)} images (USC-SIPI misc volume)")

# ---------------------------------------------------------------------------
# Step 2: random sampling with a fixed seed (reproducible)
# ---------------------------------------------------------------------------
import numpy as np

SEED = 42
rng = np.random.RandomState(SEED)
idx = sorted(rng.choice(len(ALL_IMGS), 20, replace=False))
SELECTED = [ALL_IMGS[i] for i in idx]

print(f"PRNG:          numpy.random.RandomState (MT19937)")
print(f"Seed:          {SEED}")
print(f"Method:        choice({len(ALL_IMGS)}, 20, replace=False)")
print(f"Selected:      {len(SELECTED)} images\n")

# ---------------------------------------------------------------------------
# Step 3: print index map and metadata table
# ---------------------------------------------------------------------------
try:
    from PIL import Image
    has_pil = True
except ImportError:
    has_pil = False

print(f"{'Pool#':>4s} {'Image ID':12s} {'Pixels':>14s} {'Mode':>5s} {'Bytes':>10s}")
print("-" * 55)
for k, img in enumerate(SELECTED):
    pool_idx = ALL_IMGS.index(img) + 1
    path = os.path.join(MISC, img)
    sz = os.path.getsize(path)
    if has_pil:
        im = Image.open(path)
        dims = f"{im.size[0]}x{im.size[1]}"
        mode = im.mode
    else:
        dims = "?"
        mode = "?"
    print(f"{pool_idx:4d} {os.path.splitext(img)[0]:12s} {dims:>14s} {mode:>5s} {sz:>10d}")

# ---------------------------------------------------------------------------
# Step 4: size and resolution distribution summary
# ---------------------------------------------------------------------------
sizes = [os.path.getsize(os.path.join(MISC, f)) for f in SELECTED]
print(f"\nFile size range: {min(sizes):,} - {max(sizes):,} bytes")
print(f"  ~64 KB tier  (< 100 KB):  {sum(1 for s in sizes if s < 100000)} images")
print(f"  ~192 KB tier (100-200 KB): {sum(1 for s in sizes if 100000 <= s < 200000)} images")
print(f"  ~256 KB tier (200-500 KB): {sum(1 for s in sizes if 200000 <= s < 500000)} images")
print(f"  ~768 KB tier (500-900 KB): {sum(1 for s in sizes if 500000 <= s < 900000)} images")
print(f"  ~1 MB tier   (>= 900 KB):  {sum(1 for s in sizes if s >= 900000)} images")

# ---------------------------------------------------------------------------
# Step 5: verify the redistributed 20-image subset matches the selection
# ---------------------------------------------------------------------------
if os.path.isdir(SELECTED_DIR):
    redistributed = set(f for f in os.listdir(SELECTED_DIR)
                        if f.lower().endswith((".tiff", ".tif")))
    selected_set = set(SELECTED)
    missing = selected_set - redistributed
    extra = redistributed - selected_set
    print("\nRedistributed subset check (random20_images/):")
    if not missing and not extra:
        print(f"  OK - exactly the {len(redistributed)} selected images")
    else:
        if missing:
            print(f"  MISSING: {sorted(missing)}")
        if extra:
            print(f"  EXTRA:   {sorted(extra)}")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Step 6: optional copy to target directory
# ---------------------------------------------------------------------------
if "--copy" in sys.argv:
    target = sys.argv[sys.argv.index("--copy") + 1]
    os.makedirs(target, exist_ok=True)
    for img in SELECTED:
        src = os.path.join(MISC, img)
        dst = os.path.join(target, img)
        shutil.copy2(src, dst)
    print(f"\nCopied {len(SELECTED)} files to: {target}")
