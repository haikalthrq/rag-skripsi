"""
Verifikasi kualitas mapping untuk query tertentu:
- Q041-Q045: data numerik impor (sering ada di tabel)
- Q036-Q040: statistik pendidikan
- Q019, Q020, Q024, Q025, Q030, Q034, Q035: daftar isi / daftar tabel
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data/ground_truth/retrieval_ground_truth.csv"

rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))

TARGET_QUERIES = [
    "Q019", "Q020",          # IUV - daftar isi & daftar tabel
    "Q025", "Q030", "Q034", "Q035",  # daftar tabel/isi lainnya
    "Q036", "Q037", "Q038", "Q039", "Q040",  # pendidikan
    "Q041", "Q042", "Q043", "Q044", "Q045",  # impor numerik
]

DOC_MAP = {
    "DOC01_BIK":    "data/chunked/{method}/benchmark-indeks-konstruksi--2016-100---2018---2023_chunks.json",
    "DOC02_BSK":    "data/chunked/{method}/benchmark-statistik-konstruksi--2018---2023_chunks.json",
    "DOC03_CERDAS": "data/chunked/{method}/cerita-data-statistik-untuk-indonesia---mismatch-pendidikan---pekerjaan-pemuda-indonesia--implikasi-bagi-bonus-demografi_chunks.json",
    "DOC04_IUV":    "data/chunked/{method}/indeks-unit-value-ekspor-impor---agustus-2025_chunks.json",
    "DOC05_LNPRT":  "data/chunked/{method}/neraca-lembaga-non-profit-yang-melayani-rumahtangga--2022-2024_chunks.json",
    "DOC06_NPU":    "data/chunked/{method}/neraca-pemerintahan-umum-indonesia-2019-2024_chunks.json",
    "DOC07_NRT":    "data/chunked/{method}/neraca-rumah-tangga-indonesia--2022-2024_chunks.json",
    "DOC08_PEND":   "data/chunked/{method}/statistik-pendidikan-2025_chunks.json",
    "DOC09_IMPOR":  "data/chunked/{method}/statistik-perdagangan-luar-negeri-bulanan-impor--agustus-2025_chunks.json",
    "DOC10_MODA":   "data/chunked/{method}/statistik-perdagangan-luar-negeri-menurut-moda-transportasi--2023-dan-2024_chunks.json",
}

METHOD_DIR = {
    "element": "element_based",
    "maxmin_semantic": "maxmin_semantic",
    "recursive": "recursive",
}

def get_chunk_text(doc_id, method_key, chunk_id):
    template = DOC_MAP.get(doc_id)
    if not template:
        return "[unknown doc]"
    method_dir = METHOD_DIR.get(method_key, method_key)
    path = ROOT / template.replace("{method}", method_dir)
    if not path.exists():
        return "[file not found]"
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    for c in chunks:
        if str(c.get("chunk_id")) == str(chunk_id):
            return c.get("text", "")[:300]
    return "[chunk not found]"

print("=== Verifikasi Label-2 untuk Query Kritis ===\n")

flagged = []

for qid in TARGET_QUERIES:
    q_rows = [r for r in rows if r["query_id"] == qid and r["label"] == "2"]
    if not q_rows:
        print(f"[{qid}] ⚠ TIDAK ADA label-2!")
        flagged.append(qid)
        continue

    print(f"\n[{qid}]")
    for r in q_rows:
        method = r["method"]
        cid = r["chunk_id"]
        conf = r["confidence"]
        rat = r["rationale"]
        text = get_chunk_text(r["doc_id"], method, cid)
        print(f"  method={method} | chunk={cid} | conf={conf}")
        print(f"  rationale : {rat}")
        print(f"  chunk text: {text.strip()[:200].replace(chr(10), ' ')}")
        print()

print(f"\n=== Summary ===")
print(f"Queries kritis yang dicek: {len(TARGET_QUERIES)}")
if flagged:
    print(f"⚠ Tidak ada label-2: {flagged}")
else:
    print("✓ Semua query kritis memiliki minimal 1 label-2")
