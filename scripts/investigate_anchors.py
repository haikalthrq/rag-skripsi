"""Investigasi anchor mismatch pada REVISED queries."""
import fitz, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data/raw"

DOC_MAP = {
    "DOC06_NPU":   "neraca-pemerintahan-umum-indonesia-2019-2024.pdf",
    "DOC08_PEND":  "statistik-pendidikan-2025.pdf",
    "DOC09_IMPOR": "statistik-perdagangan-luar-negeri-bulanan-impor--agustus-2025.pdf",
    "DOC10_MODA":  "statistik-perdagangan-luar-negeri-menurut-moda-transportasi--2023-dan-2024.pdf",
}

CASES = [
    ("DOC06_NPU",   7,  "Kata Pengantar"),      # Q026-Q029
    ("DOC08_PEND",  11, "Ringkasan"),            # Q040
    ("DOC09_IMPOR", 21, "Bab 1 Ringkasan"),      # Q041-Q042
    ("DOC09_IMPOR", 24, "Bab 1 Ringkasan / Tabel 1.5"),  # Q044
    ("DOC09_IMPOR", 25, "Bab 1 Ringkasan / Gambar 4"),   # Q045
    ("DOC10_MODA",  17, "Bab I Pendahuluan"),    # Q049
    ("DOC10_MODA",  21, "Bab II Metodologi"),    # Q050
]

for doc_id, page_num, anchor in CASES:
    pdf_path = PDF_DIR / DOC_MAP[doc_id]
    pdf = fitz.open(str(pdf_path))
    total_pages = len(pdf)
    
    # Get page text (1-indexed)
    idx = page_num - 1
    if idx < 0 or idx >= total_pages:
        print(f"\n[{doc_id} p{page_num}] Page index OOB (total={total_pages})")
        continue
    
    text = pdf[idx].get_text("text")
    text_norm = re.sub(r"\s+", " ", text).lower().strip()
    
    anchor_norm = anchor.lower().strip()
    found = anchor_norm in text_norm
    
    print(f"\n[{doc_id} p{page_num}] anchor='{anchor}' → found={found}")
    print(f"  Total PDF pages: {total_pages}")
    print(f"  First 300 chars of page {page_num}:")
    print(f"  {text[:300].replace(chr(10), ' ').strip()}")
    
    # Search for anchor in nearby pages
    if not found:
        for nearby_p in range(max(1, page_num-3), min(total_pages+1, page_num+5)):
            t = pdf[nearby_p-1].get_text("text")
            t_norm = re.sub(r"\s+", " ", t).lower()
            if anchor_norm in t_norm or anchor.split("/")[0].strip().lower() in t_norm:
                print(f"  → Anchor found at page {nearby_p}!")
                break
        else:
            # Search entire PDF
            print(f"  → Searching entire PDF for '{anchor}'...")
            for i in range(total_pages):
                t = pdf[i].get_text("text")
                t_norm = re.sub(r"\s+", " ", t).lower()
                if anchor.split()[0].lower() in t_norm and anchor.split()[-1].lower() in t_norm:
                    print(f"  → Partial match at page {i+1}")
                    break
