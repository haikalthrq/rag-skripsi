"""
Validasi gold_answer qa_gold terhadap teks PDF asli.

Alur:
  1. Baca qa_gold dari xlsx
  2. Ekstrak teks halaman PDF sesuai evidence_page_pdf
  3. Validasi gold_answer (angka, istilah, referensi tabel/anchor)
  4. Output qa_gold_validated.csv + generation_validation_report.md
"""

import csv
import re
import json
from pathlib import Path
import openpyxl

try:
    import fitz  # PyMuPDF
except ImportError:
    raise SystemExit("PyMuPDF (fitz) tidak tersedia. Install: pip install PyMuPDF")

ROOT      = Path(__file__).resolve().parent.parent
XLSX_PATH = ROOT / "data/ground_truth/gold_standard_rag_bps_draft.xlsx"
PDF_DIR   = ROOT / "data/raw"
OUT_CSV   = ROOT / "data/ground_truth/qa_gold_validated.csv"
OUT_MD    = ROOT / "data/ground_truth/generation_validation_report.md"

DOC_MAP = {
    "DOC01_BIK":    "benchmark-indeks-konstruksi--2016-100---2018---2023.pdf",
    "DOC02_BSK":    "benchmark-statistik-konstruksi--2018---2023.pdf",
    "DOC03_CERDAS": "cerita-data-statistik-untuk-indonesia---mismatch-pendidikan---pekerjaan-pemuda-indonesia--implikasi-bagi-bonus-demografi.pdf",
    "DOC04_IUV":    "indeks-unit-value-ekspor-impor---agustus-2025.pdf",
    "DOC05_LNPRT":  "neraca-lembaga-non-profit-yang-melayani-rumahtangga--2022-2024.pdf",
    "DOC06_NPU":    "neraca-pemerintahan-umum-indonesia-2019-2024.pdf",
    "DOC07_NRT":    "neraca-rumah-tangga-indonesia--2022-2024.pdf",
    "DOC08_PEND":   "statistik-pendidikan-2025.pdf",
    "DOC09_IMPOR":  "statistik-perdagangan-luar-negeri-bulanan-impor--agustus-2025.pdf",
    "DOC10_MODA":   "statistik-perdagangan-luar-negeri-menurut-moda-transportasi--2023-dan-2024.pdf",
}

# ─── PDF Extraction ────────────────────────────────────────────────────────────

_pdf_cache: dict = {}

def get_pdf(doc_id: str):
    fname = DOC_MAP.get(doc_id)
    if not fname:
        return None
    if doc_id not in _pdf_cache:
        path = PDF_DIR / fname
        if path.exists():
            _pdf_cache[doc_id] = fitz.open(str(path))
        else:
            return None
    return _pdf_cache[doc_id]


def extract_page_text(doc_id: str, pdf_page_num: int) -> str:
    """Ekstrak teks dari halaman PDF (1-indexed)."""
    pdf = get_pdf(doc_id)
    if pdf is None:
        return ""
    idx = pdf_page_num - 1
    if idx < 0 or idx >= len(pdf):
        return ""
    page = pdf[idx]
    return page.get_text("text")


def extract_pages_text(doc_id: str, pages: list[int]) -> str:
    """Ekstrak dan gabungkan teks dari beberapa halaman."""
    texts = []
    for pg in pages:
        t = extract_page_text(doc_id, pg)
        if t:
            texts.append(t)
    return "\n".join(texts)


def parse_page_num(raw: str) -> list[int]:
    """Ekstrak semua angka halaman dari 'PDF page 7' atau 'PDF page 21'."""
    if not raw:
        return []
    nums = re.findall(r"\d+", str(raw))
    return [int(n) for n in nums]

# ─── Validation Helpers ────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def extract_numbers(text: str) -> list[str]:
    """Ekstrak semua angka (termasuk desimal dengan koma/titik, ribuan)."""
    return re.findall(r"\d[\d.,]*\d|\d", text)


def check_numbers_in_text(answer: str, source_text: str) -> tuple[list[str], list[str]]:
    """
    Cek apakah semua angka spesifik dalam answer ada di source_text.
    Returns: (found_list, missing_list)
    """
    ans_nums = extract_numbers(answer)
    src_norm = normalize(source_text)
    found, missing = [], []
    for num in ans_nums:
        # Cek exact dan variasi (titik↔koma)
        num_variants = {num, num.replace(",", "."), num.replace(".", ",")}
        if any(v in src_norm for v in num_variants):
            found.append(num)
        else:
            missing.append(num)
    return found, missing


def check_key_terms(answer: str, source_text: str, min_len: int = 6) -> tuple[list[str], list[str]]:
    """
    Cek kata-kata penting (panjang >= min_len) dari answer ada di source.
    Mengabaikan kata fungsi dan stopwords umum.
    Returns: (found_terms, missing_terms)
    """
    STOPWORDS = {
        "adalah","yang","dari","dalam","pada","untuk","dengan","ini","itu",
        "dan","atau","ke","di","tidak","akan","telah","sudah","dapat","juga",
        "serta","bahwa","oleh","karena","sebagai","antara","terhadap","melalui",
        "sesuai","secara","tersebut","publikasi","indonesia","menurut","berupa",
        "merupakan","meliputi","mencakup","mencapai","terdiri","maupun",
    }
    src_norm = normalize(source_text)
    ans_words = set(w.lower() for w in re.findall(r"[a-zA-Z0-9À-ÿ\u00C0-\u024F]{%d,}" % min_len, answer))
    ans_words -= STOPWORDS
    found, missing = [], []
    for w in ans_words:
        if w in src_norm:
            found.append(w)
        else:
            missing.append(w)
    return found, missing


def check_anchor_in_text(anchor: str, source_text: str) -> bool:
    """
    Soft anchor check:
    - Split compound anchors (e.g. 'Bab 1 Ringkasan / Tabel 1.5') → cek bagian terpendek
    - Cek setiap kata signifikan dari anchor ada di source
    """
    if not anchor or anchor in ("", "None"):
        return True
    src_norm = normalize(source_text)
    # Coba exact match dulu
    if normalize(anchor) in src_norm:
        return True
    # Split compound anchors (Bab 1 Ringkasan / Tabel 1.5 → ['Bab 1 Ringkasan', 'Tabel 1.5'])
    parts = [p.strip() for p in anchor.split("/")]
    for part in parts:
        part_norm = normalize(part)
        # Check if all significant words of this part are in text
        words = [w for w in part_norm.split() if len(w) >= 4]
        if words and sum(1 for w in words if w in src_norm) >= max(1, len(words) // 2):
            return True
    return False


def anchor_correction_note(anchor: str, doc_id: str, page_num: int, pdf) -> str:
    """Cari halaman yang benar-benar mengandung anchor (±5 halaman)."""
    if not anchor or not pdf:
        return ""
    total = len(pdf)
    anchor_primary = normalize(anchor.split("/")[0].strip())
    for nearby in range(max(1, page_num - 4), min(total + 1, page_num + 6)):
        t = re.sub(r"\s+", " ", pdf[nearby - 1].get_text("text")).lower()
        if anchor_primary in t:
            if nearby != page_num:
                return f"Heading '{anchor}' ditemukan di halaman {nearby}, bukan halaman {page_num} — kemungkinan halaman konten dimulai setelah halaman judul bab."
            return ""
    return f"Heading '{anchor}' tidak ditemukan dalam ±5 halaman dari halaman {page_num}."


# ─── Core Validation Logic ────────────────────────────────────────────────────

MANUAL_CORRECTIONS = {}  # Diisi setelah deteksi masalah

def validate_qa(qa: dict) -> dict:
    """
    Validasi satu QA item terhadap teks PDF.
    Returns dict dengan semua field output CSV.
    """
    qid        = qa["query_id"]
    doc_id     = qa["doc_id"]
    question   = str(qa.get("question", ""))
    gold_ans   = str(qa.get("gold_answer", ""))
    ev_page    = str(qa.get("evidence_page_pdf", ""))
    ev_anchor  = str(qa.get("evidence_anchor", "") or "")
    ev_summary = str(qa.get("evidence_summary", "") or "")
    ev_type    = str(qa.get("evidence_type", "") or "")
    ev_terms   = str(qa.get("evidence_search_terms", "") or "")

    pages = parse_page_num(ev_page)

    # Ambil halaman evidence + 1 halaman sebelum/sesudah untuk konteks
    context_pages = sorted(set(pages + [p+1 for p in pages] + [p-1 for p in pages if p > 1]))
    page_text  = extract_pages_text(doc_id, pages)
    ctx_text   = extract_pages_text(doc_id, context_pages)

    # ── Cek PDF tersedia ──────────────────────────────────────────────────────
    if not page_text.strip():
        return {
            "query_id":           qid,
            "doc_id":             doc_id,
            "question":           question,
            "gold_answer_original": gold_ans,
            "revised_gold_answer": gold_ans,
            "evidence_page":      ev_page,
            "evidence_anchor":    ev_anchor,
            "evidence_text":      "(halaman tidak dapat diekstrak)",
            "status":             "NEEDS_EVIDENCE_REVIEW",
            "evidence_correction": f"Teks halaman {ev_page} tidak dapat diekstrak dari PDF.",
            "rationale":          "Halaman PDF kosong atau tidak dapat dibaca dengan fitz.",
        }

    page_norm = normalize(page_text)
    ctx_norm  = normalize(ctx_text)

    # ── Cek angka numerik ────────────────────────────────────────────────────
    found_nums, missing_nums = check_numbers_in_text(gold_ans, ctx_text)
    num_total = len(found_nums) + len(missing_nums)

    # ── Cek kata kunci penting ───────────────────────────────────────────────
    found_terms, missing_terms = check_key_terms(gold_ans, ctx_text)
    term_total = len(found_terms) + len(missing_terms)

    # ── Cek anchor ──────────────────────────────────────────────────────────
    anchor_ok = check_anchor_in_text(ev_anchor, ctx_text)
    pdf_obj   = get_pdf(doc_id)

    # ── Hitung skor ─────────────────────────────────────────────────────────
    num_score  = (len(found_nums) / num_total) if num_total else 1.0
    term_score = (len(found_terms) / term_total) if term_total else 1.0
    combined   = 0.6 * num_score + 0.4 * term_score

    # ── Snippet evidence teks (40 kata sekitar anchor/keyword) ───────────────
    ev_snippet = _extract_snippet(page_text, gold_ans, ev_anchor, max_chars=400)

    # ── Tentukan status ─────────────────────────────────────────────────────
    issues       = []
    corrections  = []
    status       = "VALID"
    revised_ans  = gold_ans

    # Cek angka yang hilang (ini kritis)
    critical_missing_nums = [n for n in missing_nums if len(n) >= 2]
    if critical_missing_nums:
        issues.append(f"Angka tidak ditemukan di halaman PDF: {critical_missing_nums}")

        # Coba cari di halaman lain (±2)
        wider_pages = list(range(max(1, pages[0]-2), pages[-1]+3)) if pages else []
        wider_text  = extract_pages_text(doc_id, wider_pages)
        _, still_missing = check_numbers_in_text(gold_ans, wider_text)
        if still_missing:
            status = "NEEDS_EVIDENCE_REVIEW"
            corrections.append(f"Angka {still_missing} tidak ditemukan di halaman {ev_page} maupun ±2 halaman sekitarnya.")
        else:
            # Ditemukan di halaman yang berdekatan → kemungkinan salah halaman
            status = "REVISED"
            corrections.append(
                f"Angka {critical_missing_nums} ada di halaman sekitar {ev_page}, bukan tepat di halaman tersebut. "
                f"Periksa rentang halaman evidence."
            )

    # Cek istilah penting yang hilang (lebih toleran)
    critical_missing_terms = [t for t in missing_terms if len(t) >= 8]
    if critical_missing_terms and status == "VALID":
        if len(critical_missing_terms) > 3:
            issues.append(f"Istilah tidak ditemukan: {critical_missing_terms[:5]}")
            status = "NEEDS_EVIDENCE_REVIEW"
            corrections.append(f"Istilah penting tidak ditemukan di halaman PDF: {critical_missing_terms[:5]}")
        elif len(critical_missing_terms) > 1:
            issues.append(f"Istilah mungkin berbeda: {critical_missing_terms}")

    # Cek anchor — hanya catat sebagai koreksi, tidak mengubah status jika konten benar
    if not anchor_ok and ev_anchor not in ("", "None"):
        anc_note = anchor_correction_note(ev_anchor, doc_id, pages[0] if pages else 0, pdf_obj)
        if anc_note:
            corrections.append(anc_note)
            # Hanya eskalasi ke REVISED jika konten juga bermasalah
            if status == "VALID" and critical_missing_nums:
                status = "REVISED"

    # Compose rationale
    if status == "VALID":
        rationale = (
            f"Semua angka ({len(found_nums)}/{num_total}) dan istilah kunci "
            f"({len(found_terms)}/{term_total}) ditemukan di halaman {ev_page}. "
            f"Gold answer konsisten dengan teks PDF."
        )
    elif status == "REVISED":
        rationale = " | ".join(issues + corrections) if (issues or corrections) else "Koreksi minor diperlukan."
    else:
        rationale = " | ".join(issues + corrections) if (issues or corrections) else "Evidence tidak cukup membuktikan jawaban."

    return {
        "query_id":             qid,
        "doc_id":               doc_id,
        "question":             question,
        "gold_answer_original": gold_ans,
        "revised_gold_answer":  revised_ans,
        "evidence_page":        ev_page,
        "evidence_anchor":      ev_anchor,
        "evidence_text":        ev_snippet,
        "status":               status,
        "evidence_correction":  " | ".join(corrections) if corrections else "",
        "rationale":            rationale,
    }


def _extract_snippet(page_text: str, gold_ans: str, anchor: str, max_chars: int = 400) -> str:
    """Ambil snippet teks PDF yang paling relevan dengan gold_answer."""
    # Cari baris yang mengandung angka atau anchor
    lines = page_text.split("\n")
    gold_nums = set(extract_numbers(gold_ans))
    anchor_norm = normalize(anchor)

    scored_lines = []
    for i, line in enumerate(lines):
        ln = normalize(line)
        score = 0
        for num in gold_nums:
            if num in ln or num.replace(",", ".") in ln:
                score += 2
        if anchor_norm and anchor_norm in ln:
            score += 1
        if score > 0:
            # Include context: previous and next lines
            start = max(0, i - 1)
            end   = min(len(lines), i + 3)
            snippet = " ".join(lines[start:end]).strip()
            scored_lines.append((score, snippet))

    if not scored_lines:
        # Fallback: first 400 chars of page
        return page_text[:max_chars].replace("\n", " ").strip()

    scored_lines.sort(reverse=True)
    return scored_lines[0][1][:max_chars]


# ─── Load qa_gold ──────────────────────────────────────────────────────────────

def load_qa_gold() -> list[dict]:
    wb = openpyxl.load_workbook(str(XLSX_PATH), read_only=True, data_only=True)
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


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading qa_gold...")
    qa_items = load_qa_gold()
    print(f"  {len(qa_items)} items loaded.")

    results = []
    for qa in qa_items:
        qid = qa["query_id"]
        print(f"  Validating {qid}...", end="")
        r = validate_qa(qa)
        results.append(r)
        print(f" [{r['status']}]")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    fieldnames = [
        "query_id", "doc_id", "question",
        "gold_answer_original", "revised_gold_answer",
        "evidence_page", "evidence_anchor", "evidence_text",
        "status", "evidence_correction", "rationale",
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\n✓ CSV: {OUT_CSV}")

    # ── Stats ─────────────────────────────────────────────────────────────────
    status_counts = {}
    for r in results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    n_valid   = status_counts.get("VALID", 0)
    n_revised = status_counts.get("REVISED", 0)
    n_needs   = status_counts.get("NEEDS_EVIDENCE_REVIEW", 0)
    n_ambig   = status_counts.get("AMBIGUOUS", 0)
    total     = len(results)

    needs_manual = [r for r in results if r["status"] in ("REVISED", "NEEDS_EVIDENCE_REVIEW", "AMBIGUOUS")]

    # ── Write Markdown Report ─────────────────────────────────────────────────
    lines = [
        "# Laporan Validasi Ground Truth Generation",
        "",
        "## Ringkasan",
        "",
        f"| Metric | Jumlah |",
        f"|--------|--------|",
        f"| Total query | {total} |",
        f"| VALID | {n_valid} |",
        f"| REVISED | {n_revised} |",
        f"| NEEDS_EVIDENCE_REVIEW | {n_needs} |",
        f"| AMBIGUOUS | {n_ambig} |",
        "",
        "---",
        "",
        "## Detail per Query",
        "",
    ]

    for r in results:
        s = r["status"]
        icon = "✅" if s == "VALID" else ("⚠️" if s == "REVISED" else "❌")
        lines.append(f"### {icon} {r['query_id']} — {s}")
        lines.append(f"**Dokumen:** {r['doc_id']}  ")
        lines.append(f"**Pertanyaan:** {r['question']}  ")
        lines.append(f"**Gold Answer:** {r['gold_answer_original']}  ")
        if s != "VALID":
            lines.append(f"**Revised Answer:** {r['revised_gold_answer']}  ")
        lines.append(f"**Evidence Page:** {r['evidence_page']} | **Anchor:** {r['evidence_anchor']}  ")
        lines.append(f"**Rationale:** {r['rationale']}  ")
        if r["evidence_correction"]:
            lines.append(f"**Koreksi Evidence:** {r['evidence_correction']}  ")
        if r["evidence_text"]:
            lines.append(f"**Snippet PDF:**")
            lines.append(f"> {r['evidence_text'][:300]}  ")
        lines.append("")

    lines += [
        "---",
        "",
        "## Query yang Perlu Dicek Manual",
        "",
    ]
    if needs_manual:
        for r in needs_manual:
            lines.append(f"- **{r['query_id']}** ({r['status']}): {r['rationale'][:120]}")
    else:
        lines.append("- Tidak ada.")

    lines += [
        "",
        "---",
        "",
        "## Catatan Umum",
        "",
        "- Validasi dilakukan dengan membandingkan gold_answer terhadap teks yang diekstrak dari halaman PDF yang direferensikan.",
        "- `evidence_text` pada qa_gold kosong sehingga teks evidence diambil langsung dari PDF menggunakan PyMuPDF.",
        "- Angka dengan format berbeda (titik vs koma) dicocokan secara fleksibel.",
        "- Teks tabel yang tidak ter-ekstrak dengan baik oleh fitz dapat menyebabkan false NEEDS_EVIDENCE_REVIEW.",
        "- Status REVISED tidak mengubah makna gold_answer, hanya koreksi minor (halaman anchor, penulisan angka).",
        "",
        f"_Dihasilkan secara otomatis. Semua item berstatus non-VALID wajib dicek manual._",
    ]

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✓ Report: {OUT_MD}")

    print(f"\n=== Hasil Validasi ===")
    print(f"  VALID                : {n_valid}/{total}")
    print(f"  REVISED              : {n_revised}/{total}")
    print(f"  NEEDS_EVIDENCE_REVIEW: {n_needs}/{total}")
    print(f"  AMBIGUOUS            : {n_ambig}/{total}")
    if needs_manual:
        print(f"\n  Query perlu cek manual ({len(needs_manual)}):")
        for r in needs_manual:
            print(f"    ! {r['query_id']}: {r['rationale'][:80]}")


if __name__ == "__main__":
    main()
