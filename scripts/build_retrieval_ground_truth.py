"""
Build retrieval ground truth CSV dari qa_gold (xlsx) + chunk JSON files.

Output: data/ground_truth/retrieval_ground_truth.csv
Kolom : query_id, doc_id, method, chunk_id, label, rationale, confidence, notes

Label:
  2 = chunk memuat bukti utama / jawaban lengkap
  1 = chunk memuat sebagian bukti / konteks pendukung

Metode matching (per method):
  element_based  : page_match + keyword_score + anchor_score
  maxmin_semantic: keyword_score + anchor_score  (tidak ada page info)
  recursive      : keyword_score + anchor_score  (tidak ada page info)
"""

import csv
import json
import re
import sys
from pathlib import Path

import openpyxl

# ─── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
XLSX_PATH  = ROOT / "data/ground_truth/gold_standard_rag_bps_draft.xlsx"
CHUNK_DIR  = ROOT / "data/chunked"
OUT_CSV    = ROOT / "data/ground_truth/retrieval_ground_truth.csv"

# ─── doc_id → chunk filename stem ──────────────────────────────────────────────
DOC_MAP = {
    "DOC01_BIK":    "benchmark-indeks-konstruksi--2016-100---2018---2023_chunks",
    "DOC02_BSK":    "benchmark-statistik-konstruksi--2018---2023_chunks",
    "DOC03_CERDAS": "cerita-data-statistik-untuk-indonesia---mismatch-pendidikan---pekerjaan-pemuda-indonesia--implikasi-bagi-bonus-demografi_chunks",
    "DOC04_IUV":    "indeks-unit-value-ekspor-impor---agustus-2025_chunks",
    "DOC05_LNPRT":  "neraca-lembaga-non-profit-yang-melayani-rumahtangga--2022-2024_chunks",
    "DOC06_NPU":    "neraca-pemerintahan-umum-indonesia-2019-2024_chunks",
    "DOC07_NRT":    "neraca-rumah-tangga-indonesia--2022-2024_chunks",
    "DOC08_PEND":   "statistik-pendidikan-2025_chunks",
    "DOC09_IMPOR":  "statistik-perdagangan-luar-negeri-bulanan-impor--agustus-2025_chunks",
    "DOC10_MODA":   "statistik-perdagangan-luar-negeri-menurut-moda-transportasi--2023-dan-2024_chunks",
}

METHOD_DIR = {
    "element":          "element_based",
    "maxmin_semantic":  "maxmin_semantic",
    "recursive":        "recursive",
}

# ─── Thresholds ────────────────────────────────────────────────────────────────
LABEL2_THRESHOLD  = 0.55   # skor >= ini → label 2
LABEL1_THRESHOLD  = 0.25   # skor >= ini → label 1
NOT_FOUND_MAX     = 0.15   # skor max < ini → NOT_FOUND

# Max label-2 per query×method (pilih yang terbaik)
MAX_LABEL2 = 3
MAX_LABEL1 = 5

# ─── Helpers ───────────────────────────────────────────────────────────────────

def parse_page(raw: str) -> int | None:
    """Extract angka halaman dari string seperti 'PDF page 7'."""
    if not raw:
        return None
    m = re.search(r"\d+", str(raw))
    return int(m.group()) if m else None


def parse_keywords(raw: str) -> list[str]:
    """Split 'kw1; kw2; kw3' → ['kw1', 'kw2', 'kw3'], lowercase."""
    if not raw:
        return []
    return [k.strip().lower() for k in str(raw).split(";") if k.strip()]


# Pola TOC (Daftar Isi / Daftar Tabel / Daftar Gambar)
_TOC_PATTERNS = re.compile(
    r"(\.\.\s*\d+|daftar isi|daftar tabel|daftar gambar|table of contents|list of tables|\.\.\.\.+)",
    re.IGNORECASE,
)

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def is_toc_chunk(text: str) -> bool:
    """True jika chunk adalah Daftar Isi / Daftar Tabel / TOC."""
    matches = _TOC_PATTERNS.findall(text)
    # Jika >4 pola TOC atau chunk pendek & banyak '...' → ini adalah TOC
    return len(matches) >= 4 or (len(text) < 1500 and text.count("...") >= 3)


def keyword_score(text_norm: str, keywords: list[str]) -> tuple[float, int]:
    """
    Hitung rasio keyword yang ditemukan dalam teks.
    Mendukung word-level matching: 'anak 0-6 tahun' match 'anak usia 0-6 tahun'
    jika semua kata-kata keyword hadir dalam teks.
    """
    if not keywords:
        return 0.0, 0

    hits = 0
    for kw in keywords:
        # Exact substring match
        if kw in text_norm:
            hits += 1
            continue
        # Word-level match: semua kata dari keyword ada di teks
        kw_words = [w for w in kw.split() if len(w) >= 3]
        if kw_words and all(w in text_norm for w in kw_words):
            hits += 1
    return hits / len(keywords), hits


def anchor_score(text_norm: str, anchor: str) -> float:
    """1.0 jika evidence_anchor ditemukan dalam teks chunk, 0 otherwise."""
    if not anchor:
        return 0.0
    return 1.0 if normalize(anchor) in text_norm else 0.0


def page_match(chunk_pages: list[int] | None, target_page: int | None) -> float:
    """1.0 jika halaman target ada di chunk pages."""
    if not chunk_pages or target_page is None:
        return 0.0
    return 1.0 if target_page in chunk_pages else 0.0


def score_chunk(
    chunk_text: str,
    chunk_pages: list[int] | None,
    keywords: list[str],
    anchor: str,
    evidence_summary: str,
    target_page: int | None,
    has_page_info: bool,
    is_toc: bool = False,
) -> float:
    """
    Hitung skor relevansi chunk terhadap query.

    Bobot:
      - page_match   : 0.35 (element_based only, else 0)
      - keyword_score: 0.40 (atau 0.60 jika tidak ada page info)
      - anchor_score : 0.20 (atau 0.30 jika tidak ada page info)
      - summary_hit  : 0.05 (atau 0.10 jika tidak ada page info)
    """
    tn = normalize(chunk_text)
    ks, _hits = keyword_score(tn, keywords)
    as_ = anchor_score(tn, anchor)
    pm  = page_match(chunk_pages, target_page) if has_page_info else 0.0

    # summary_hit: cek apakah fragmen evidence_summary ada di chunk
    ev_norm = normalize(evidence_summary) if evidence_summary else ""
    ev_words = [w for w in ev_norm.split() if len(w) > 4]
    ev_hit = 0.0
    if ev_words:
        ev_hit = sum(1 for w in ev_words if w in tn) / len(ev_words)

    if has_page_info:
        total = 0.35 * pm + 0.40 * ks + 0.20 * as_ + 0.05 * ev_hit
    else:
        total = 0.60 * ks + 0.30 * as_ + 0.10 * ev_hit

    # Penalti berat untuk chunk TOC / Daftar Isi
    if is_toc:
        total *= 0.25

    return round(total, 4)


# ─── Load qa_gold ──────────────────────────────────────────────────────────────

def load_qa_gold(xlsx_path: Path) -> list[dict]:
    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True, data_only=True)
    ws = wb["qa_gold"]
    headers = None
    rows = []
    for row in ws.iter_rows(values_only=True):
        if headers is None:
            headers = list(row)
            continue
        if not row[0]:
            continue
        rows.append(dict(zip(headers, row)))
    return rows


# ─── Load chunks ───────────────────────────────────────────────────────────────

def load_chunks(doc_id: str, method_key: str) -> list[dict] | None:
    stem     = DOC_MAP.get(doc_id)
    dir_name = METHOD_DIR.get(method_key)
    if not stem or not dir_name:
        return None
    path = CHUNK_DIR / dir_name / f"{stem}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─── Score all chunks for one query × method ──────────────────────────────────

def find_relevant_chunks(
    qa: dict,
    method_key: str,
) -> list[dict]:
    """
    Return list of dicts: {chunk_id, label, score, rationale, notes}
    Diurutkan skor tertinggi dulu.
    """
    chunks = load_chunks(qa["doc_id"], method_key)
    if chunks is None:
        return [{
            "chunk_id": "NOT_FOUND",
            "label": 0,
            "score": 0.0,
            "rationale": f"File chunk tidak ditemukan untuk {qa['doc_id']}/{method_key}",
            "notes": "file_missing",
        }]

    has_page = method_key == "element"
    keywords     = parse_keywords(qa.get("evidence_search_terms", ""))
    anchor       = str(qa.get("evidence_anchor", "") or "")
    ev_summary   = str(qa.get("evidence_summary", "") or "")
    target_page  = parse_page(qa.get("evidence_page_pdf", ""))
    gold_answer  = normalize(str(qa.get("gold_answer", "") or ""))

    scored = []
    for chunk in chunks:
        cid  = chunk.get("chunk_id")
        text = chunk.get("text", "")
        meta = chunk.get("metadata", {})
        pages = meta.get("page_numbers") if has_page else None

        toc = is_toc_chunk(text)
        s = score_chunk(
            chunk_text=text,
            chunk_pages=pages,
            keywords=keywords,
            anchor=anchor,
            evidence_summary=ev_summary,
            target_page=target_page,
            has_page_info=has_page,
            is_toc=toc,
        )
        scored.append((s, cid, text, pages))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    n2, n1 = 0, 0

    for s, cid, text, pages in scored:
        if s < LABEL1_THRESHOLD:
            break

        tn = normalize(text)
        kw_score, kw_hits = keyword_score(tn, keywords)
        pm = page_match(pages, target_page) if has_page else None
        an = anchor_score(tn, anchor)

        # Build rationale string
        parts = [f"score={s:.3f}"]
        parts.append(f"kw={kw_hits}/{len(keywords)}")
        if has_page and pages:
            parts.append(f"pages={pages}")
            parts.append(f"page_match={'Y' if pm else 'N'}")
        if an:
            parts.append(f"anchor_found=Y")

        rationale = " | ".join(parts)

        if s >= LABEL2_THRESHOLD and n2 < MAX_LABEL2:
            label = 2
            n2 += 1
        elif s >= LABEL1_THRESHOLD and n1 < MAX_LABEL1:
            label = 1
            n1 += 1
        else:
            continue

        results.append({
            "chunk_id": cid,
            "label": label,
            "score": s,
            "rationale": rationale,
            "notes": "auto",
        })

    # Jika tidak ada label 2 sama sekali tapi ada label 1 → promote yang terbaik
    has_label2 = any(r["label"] == 2 for r in results)
    if not has_label2 and results:
        results[0]["label"] = 2
        results[0]["notes"] = "auto_promoted_best"

    # NOT_FOUND jika skor max < threshold
    if not results or (scored and scored[0][0] < NOT_FOUND_MAX):
        return [{
            "chunk_id": "NOT_FOUND",
            "label": 0,
            "score": scored[0][0] if scored else 0.0,
            "rationale": f"Skor max={scored[0][0]:.3f} < threshold {NOT_FOUND_MAX}. Evidence kemungkinan di tabel/gambar atau tidak ter-chunk.",
            "notes": "not_found",
        }]

    return results


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading qa_gold from {XLSX_PATH}...")
    qa_items = load_qa_gold(XLSX_PATH)
    print(f"  {len(qa_items)} query items loaded.")

    methods = list(METHOD_DIR.keys())
    rows_out    = []
    ambiguous   = []
    not_found   = []

    for qa in qa_items:
        qid    = qa["query_id"]
        doc_id = qa["doc_id"]

        for method in methods:
            results = find_relevant_chunks(qa, method)

            for r in results:
                confidence = "high" if r["score"] >= 0.55 else ("medium" if r["score"] >= 0.30 else "low")

                rows_out.append({
                    "query_id":   qid,
                    "doc_id":     doc_id,
                    "method":     method,
                    "chunk_id":   r["chunk_id"],
                    "label":      r["label"] if r["chunk_id"] != "NOT_FOUND" else 0,
                    "rationale":  r["rationale"],
                    "confidence": confidence if r["chunk_id"] != "NOT_FOUND" else "none",
                    "notes":      r["notes"],
                })

                if r["chunk_id"] == "NOT_FOUND":
                    not_found.append(f"{qid}/{method}: {r['rationale']}")
                elif r["score"] < 0.35 and r["label"] == 2:
                    ambiguous.append(f"{qid}/{method}/chunk_{r['chunk_id']}: score={r['score']:.3f} (promoted, low confidence)")

        # Cek query dengan semua method NOT_FOUND
        qm_results = [r for r in rows_out if r["query_id"] == qid]
        all_nf = all(r["chunk_id"] == "NOT_FOUND" for r in qm_results)
        if all_nf:
            ambiguous.append(f"{qid}: NOT_FOUND di semua method - perlu validasi manual")

    # ── Write CSV ────────────────────────────────────────────────────────────────
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["query_id", "doc_id", "method", "chunk_id", "label", "rationale", "confidence", "notes"]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\n✓ CSV disimpan ke: {OUT_CSV}")
    print(f"  Total rows : {len(rows_out)}")
    print(f"  Label 2    : {sum(1 for r in rows_out if r['label'] == 2)}")
    print(f"  Label 1    : {sum(1 for r in rows_out if r['label'] == 1)}")
    print(f"  NOT_FOUND  : {len(not_found)}")

    # ── Summary per query×method ─────────────────────────────────────────────────
    print("\n=== Coverage per method ===")
    for method in methods:
        method_rows = [r for r in rows_out if r["method"] == method]
        n_queries  = len(set(r["query_id"] for r in method_rows))
        n_nf       = sum(1 for r in method_rows if r["chunk_id"] == "NOT_FOUND")
        n_l2       = sum(1 for r in method_rows if r["label"] == 2)
        print(f"  {method:<20}: queries={n_queries} | label2={n_l2} | not_found={n_nf}")

    # ── Ambiguous list ────────────────────────────────────────────────────────────
    if ambiguous:
        print(f"\n=== Query yang perlu validasi manual ({len(ambiguous)}) ===")
        for a in ambiguous:
            print(f"  ! {a}")

    if not_found:
        print(f"\n=== NOT_FOUND ({len(not_found)}) ===")
        for nf in not_found:
            print(f"  - {nf}")

    # ── Save ambiguous list to txt ────────────────────────────────────────────────
    amb_path = OUT_CSV.parent / "validation_needed.txt"
    with open(amb_path, "w", encoding="utf-8") as f:
        f.write("=== QUERY PERLU VALIDASI MANUAL ===\n")
        for a in ambiguous:
            f.write(f"! {a}\n")
        f.write("\n=== NOT_FOUND ===\n")
        for nf in not_found:
            f.write(f"- {nf}\n")
    print(f"\n  Daftar validasi disimpan ke: {amb_path}")


if __name__ == "__main__":
    main()
