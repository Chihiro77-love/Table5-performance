# -*- coding: utf-8 -*-
"""
Record and verification of the 20-image benchmark subset.

The 20 test images were picked at random without replacement from the
39-image USC-SIPI misc volume (https://sipi.usc.edu/database/misc.zip).
The random draw was performed manually without a PRNG or a recorded seed,
so it is not algorithmically replayable; its outcome is fixed exactly by
the 20 redistributed files in random20_images/ together with their
SHA-256 hashes in SHA256SUMS.txt.

This script therefore does NOT perform any sampling. It:
  1. verifies the SHA-256 hashes of the 20 images against SHA256SUMS.txt
  2. prints the image ID / dimensions / size of every selected image
  3. if the full 39-image pool is downloaded, asserts it contains exactly
     39 .tiff files and that all 20 selected images are members of it

Optional prerequisite (full 39-image pool; see 01_Datasets/DATASET.txt):
  download https://sipi.usc.edu/database/misc.zip and unpack it so that
  the 39 .tiff files are at
      01_Datasets/benchmark_dataset/misc_raw/misc/*.tiff

Usage:
    python sample_selected20.py
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", ".."))
DATASET = os.path.join(PKG, "01_Datasets", "benchmark_dataset")
MISC = os.path.join(DATASET, "misc_raw", "misc")
SELECTED_DIR = os.path.join(DATASET, "random20_images")
MANIFEST = os.path.join(DATASET, "SHA256SUMS.txt")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Step 1: enumerate the recorded 20-image set
# ---------------------------------------------------------------------------
if not os.path.isdir(SELECTED_DIR):
    sys.exit(f"Selected-image folder not found:\n  {SELECTED_DIR}")

SELECTED = sorted(f for f in os.listdir(SELECTED_DIR)
                  if f.lower().endswith((".tiff", ".tif")))
assert len(SELECTED) == 20, f"expected 20 selected images, found {len(SELECTED)}"
print(f"Recorded benchmark subset: {len(SELECTED)} images "
      f"(randomly picked, without replacement, from the 39-image pool)")
print("No PRNG seed was recorded for the draw; the set is fixed by the "
      "redistributed files and SHA256SUMS.txt\n")

# ---------------------------------------------------------------------------
# Step 2: verify SHA-256 hashes against the manifest
# ---------------------------------------------------------------------------
expected = {}
with open(MANIFEST) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, relpath = line.split(None, 1)
        expected[os.path.basename(relpath.strip())] = digest.lower()

print("Hash verification against SHA256SUMS.txt:")
bad = 0
for img in SELECTED:
    actual = sha256_of(os.path.join(SELECTED_DIR, img))
    want = expected.get(img)
    ok = want is not None and actual == want
    bad += not ok
    print(f"  {'OK  ' if ok else 'FAIL'} {img}")
if bad:
    sys.exit(f"{bad} image(s) failed hash verification")
print("  all 20 hashes match\n")

# ---------------------------------------------------------------------------
# Step 3: metadata table
# ---------------------------------------------------------------------------
try:
    from PIL import Image
    has_pil = True
except ImportError:
    has_pil = False

print(f"{'Image ID':12s} {'Pixels':>14s} {'Mode':>5s} {'Bytes':>10s}")
print("-" * 48)
sizes = []
for img in SELECTED:
    path = os.path.join(SELECTED_DIR, img)
    sz = os.path.getsize(path)
    sizes.append(sz)
    if has_pil:
        im = Image.open(path)
        dims = f"{im.size[0]}x{im.size[1]}"
        mode = im.mode
    else:
        dims, mode = "?", "?"
    print(f"{os.path.splitext(img)[0]:12s} {dims:>14s} {mode:>5s} {sz:>10d}")

print(f"\nFile size range: {min(sizes):,} - {max(sizes):,} bytes")

# ---------------------------------------------------------------------------
# Step 4: optional pool-membership check (requires the downloaded pool)
# ---------------------------------------------------------------------------
if os.path.isdir(MISC):
    all_imgs = sorted(f for f in os.listdir(MISC) if f.endswith(".tiff"))
    assert len(all_imgs) == 39, \
        f"expected 39 images in the pool, found {len(all_imgs)}"
    pool = set(all_imgs)
    missing = [img for img in SELECTED if img not in pool]
    print(f"\nPool-membership check against the 39-image misc volume "
          f"({MISC}):")
    if missing:
        sys.exit(f"  FAIL - not found in the pool: {missing}")
    print(f"  OK - all 20 selected images are members of the 39-image pool")
else:
    print(f"\nPool-membership check skipped (full pool not present).")
    print("Download https://sipi.usc.edu/database/misc.zip and unpack the")
    print(f"inner 'misc' folder to:\n  {MISC}")
