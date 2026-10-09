# Headless benchmark wrapper for DNA-Chain (Windows_DNA_Chain.py).
#
# Faithfully reproduces the timing conventions of
# Windows_DNA_Chain.process_image_wrapper():
#   - contour extraction is performed BEFORE the encode timer starts
#   - encode timer  : Huffman coding of chain codes + 5-bit -> 3-base mapping
#   - decode timer  : Huffman decoding + regrouping + chain-code -> image
#                     restoration + 3-base -> binary decoding
# Only the Qt QImage conversion (GUI display) is omitted; it is not part of
# the storage codec. The image size used for restoration is the actual size
# of the loaded grayscale image (the GUI hard-codes 500x500, which would
# crop images of other sizes; we restore to the true dimensions).
#
# Additionally reports side information that DNA-Chain requires for
# reconstruction but stores outside the base string (Huffman table,
# contour start points, contour lengths, image dimensions) so that
# redundancy can be accounted honestly.

import importlib.util
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Windows_DNA_Chain.py resides in this same directory (02_Software/DNA-Chain)
CHAIN_PATH = os.path.join(HERE, "Windows_DNA_Chain.py")

spec = importlib.util.spec_from_file_location("dna_chain", CHAIN_PATH)
dna_chain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dna_chain)


def side_info_bytes(n_contours, huffman_codes, dims):
    """Canonical serialization size of DNA-Chain side information:
    - image dims: 2 x uint32
    - per contour: start point (2 x int32) + chain length (uint32)
    - Huffman table: per symbol: 1 byte symbol + 1 byte code length +
      code bits padded to whole bytes
    - total payload bit length: uint64
    """
    dims_b = 8
    contours_b = n_contours * 12
    table_b = 0
    for sym, code in huffman_codes.items():
        table_b += 2 + (len(code) + 7) // 8
    total_b = dims_b + contours_b + table_b + 8
    return total_b


def run(image_path, sub_error_rate=0.0, rng_seed=0, dropout_rate=0.0):
    image_size = None
    # --- outside encode timer (as in the released tool) ---
    chaincodes, start_points = dna_chain.image_to_chaincode_with_start(image_path)
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    image_size = binary_image.shape

    # ---------------- encode (timed) ----------------
    t0 = time.perf_counter()
    # 1. Huffman 编码
    encoded_chaincodes, huffman_codes = dna_chain.encode_chaincodes_with_huffman(chaincodes)
    # 2. 5-bit → 3-base 映射（先补齐到5的倍数，避免末尾bit丢失）
    pad_bits = (5 - len(encoded_chaincodes) % 5) % 5
    padded_bits = encoded_chaincodes + '0' * pad_bits
    encoded_bases = ""
    for i in range(0, len(padded_bits), 5):
        group = padded_bits[i:i + 5]
        encoded_bases += dna_chain.binary_to_three_bases(group)
    # 3. RS 纠错码: 分块 + 索引 + 文件ID + RS校验
    oligos, file_id, n_segments = dna_chain.encode_with_rs(encoded_bases, image_path)
    encode_time = time.perf_counter() - t0
    # ---------------- end encode ----------------

    payload_bits = len(encoded_chaincodes)
    tail_bits_dropped = payload_bits % 5
    # 实际寡核苷酸数和总碱基数
    n_oligos = len(oligos)
    total_nt = sum(len(o) for o in oligos)

    side_b = side_info_bytes(len(chaincodes), huffman_codes, image_size)

    # ---------------- decode (timed) ----------------
    t0 = time.perf_counter()
    # 对寡核苷酸施加信道错误
    noisy_oligos = list(oligos)  # 先复制一份
    if sub_error_rate > 0.0 or dropout_rate > 0.0:
        rng = np.random.RandomState(rng_seed)
        # 1. 替换错误
        if sub_error_rate > 0.0:
            corrupted = []
            for ol in noisy_oligos:
                arr = np.frombuffer(ol.encode("ascii"), dtype=np.uint8).copy()
                mask = rng.random_sample(arr.size) < sub_error_rate
                pool = np.array([ord(c) for c in "ACGT"], dtype=np.uint8)
                arr[mask] = rng.choice(pool, size=int(mask.sum()))
                corrupted.append(arr.tobytes().decode("ascii"))
            noisy_oligos = corrupted
        # 2. 链丢失
        if dropout_rate > 0.0:
            n_keep = int(len(noisy_oligos) * (1.0 - dropout_rate) + 0.5)
            keep_idx = rng.choice(len(noisy_oligos), size=n_keep, replace=False)
            noisy_oligos = [noisy_oligos[i] for i in sorted(keep_idx)]

    # RS 纠错解码
    corrected_bases = dna_chain.decode_with_rs(noisy_oligos, payload_bits, file_id)

    # 从纠错后的碱基序列恢复链码
    # 截取到原始 payload 对应的碱基数（含编码时补齐的padding）
    original_payload_nt = ((payload_bits + 4) // 5) * 3
    corrected_payload_bases = corrected_bases[:original_payload_nt]

    # 碱基 → 比特 → 链码
    bits = ""
    for i in range(0, len(corrected_payload_bases), 3):
        group = corrected_payload_bases[i:i + 3]
        if len(group) == 3:
            b0 = dna_chain.base_to_binary(group[0])
            b1 = dna_chain.base_to_binary(group[1])
            if group[1] == group[2]:
                b2 = "0" if group[1] in ["A", "G"] else "1"
            else:
                b2 = "0" if group[2] in ["A", "G"] else "1"
            bits += b0 + b1 + b2
    bits = bits[:payload_bits]
    # 补齐5b→3b映射可能丢弃的末尾位（payload_bits % 5 != 0时）
    if len(bits) < payload_bits:
        bits = bits + '0' * (payload_bits - len(bits))
    decoded_chaincodes = dna_chain.decode_huffman_to_chaincodes(bits, huffman_codes)

    # 分组链码
    grouped = []
    idx = 0
    for chaincode in chaincodes:
        grouped.append(decoded_chaincodes[idx:idx + len(chaincode)])
        idx += len(chaincode)
    restored = dna_chain.chaincode_to_image_with_start(grouped, start_points, image_size)
    decode_time = time.perf_counter() - t0
    # ---------------- end decode ----------------

    pixel_match = float(np.mean(restored == binary_image))
    stream_match = (decoded_chaincodes == [
        c for chaincode in chaincodes for c in chaincode])

    result = {
        "image_path": os.path.abspath(image_path),
        "image_hw": [int(image_size[0]), int(image_size[1])],
        "n_contours": len(chaincodes),
        "n_chaincode_symbols": sum(len(c) for c in chaincodes),
        "payload_bits": payload_bits,
        "tail_bits_dropped": int(tail_bits_dropped),
        "encoded_nt": total_nt,
        "n_oligos": n_oligos,
        "virtual_oligos_120nt": n_oligos,  # actual oligos now
        "oligo_length_nt": len(oligos[0]) if oligos else 0,
        "side_info_bytes": int(side_b),
        "encode_time_s": encode_time,
        "decode_time_s": decode_time,
        "recovery_stream_exact": bool(stream_match),
        "pixel_match_fraction": pixel_match,
        "sub_error_rate": sub_error_rate,
        "rs_nsym": dna_chain.RS_NSYM,
        "rs_correctable_errors": dna_chain.RS_NSYM // 2,
        "info_block_nt": dna_chain.INFO_BLOCK_NT,
    }
    return result


if __name__ == "__main__":
    image_path = sys.argv[1]
    sub_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    drop = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    res = run(image_path, sub_rate, seed, drop)
    print(json.dumps(res))
