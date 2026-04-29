"""Cari chunk yang memuat angka/teks spesifik untuk validasi manual."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PEND_FILES = {
    "element":         ROOT / "data/chunked/element_based/statistik-pendidikan-2025_chunks.json",
    "maxmin_semantic": ROOT / "data/chunked/maxmin_semantic/statistik-pendidikan-2025_chunks.json",
    "recursive":       ROOT / "data/chunked/recursive/statistik-pendidikan-2025_chunks.json",
}

# Angka/teks spesifik yang dicari
SEARCH_TERMS = {
    "Q038": ["39,68", "ruang kelas", "kondisi baik"],
    "Q039": ["34,15", "prasekolah", "0-6 tahun", "0–6 tahun"],
    "Q040": ["APS", "APK", "2025", "meningkat"],
    "Q037": ["jumlah sekolah", "SD", "SMP", "SMA", "SMK", "2024/2025"],
}

def search_chunks(chunks, terms):
    results = []
    for c in chunks:
        text = c.get("text", "").lower()
        hit_count = sum(1 for t in terms if t.lower() in text)
        if hit_count >= 2:
            pages = c.get("metadata", {}).get("page_numbers", "?")
            results.append((hit_count, c["chunk_id"], pages, c["text"][:200]))
    results.sort(reverse=True)
    return results

for qid, terms in SEARCH_TERMS.items():
    print(f"\n=== {qid} | terms: {terms} ===")
    for method, fpath in PEND_FILES.items():
        with open(fpath, encoding="utf-8") as f:
            chunks = json.load(f)
        found = search_chunks(chunks, terms)
        print(f"\n  [{method}] top matches:")
        if not found:
            print("    (none)")
        for hits, cid, pages, text in found[:3]:
            print(f"    chunk={cid} hits={hits}/{len(terms)} pages={pages}")
            print(f"    {text.strip()[:150].replace(chr(10), ' ')}")
