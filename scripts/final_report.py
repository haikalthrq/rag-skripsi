"""Generate laporan ringkasan ground truth retrieval."""
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data/ground_truth/retrieval_ground_truth.csv"

rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))

METHODS = ["element", "maxmin_semantic", "recursive"]
ALL_QIDS = sorted(set(r["query_id"] for r in rows))

print("=" * 65)
print("  GROUND TRUTH RETRIEVAL — LAPORAN RINGKASAN")
print("=" * 65)
print(f"  Total rows  : {len(rows)}")
print(f"  Total query : {len(ALL_QIDS)}")
print(f"  Label 2     : {sum(1 for r in rows if r['label']=='2')}")
print(f"  Label 1     : {sum(1 for r in rows if r['label']=='1')}")
print(f"  NOT_FOUND   : {sum(1 for r in rows if r['chunk_id']=='NOT_FOUND')}")

# Per-method coverage
print(f"\n{'Method':<20} {'Queries':>8} {'Label2':>8} {'Label1':>8} "
      f"{'NOT_FOUND':>10} {'Avg_L2/q':>10}")
print("-" * 65)

for method in METHODS:
    m_rows = [r for r in rows if r["method"] == method]
    n_q  = len(set(r["query_id"] for r in m_rows))
    n_l2 = sum(1 for r in m_rows if r["label"] == "2")
    n_l1 = sum(1 for r in m_rows if r["label"] == "1")
    n_nf = sum(1 for r in m_rows if r["chunk_id"] == "NOT_FOUND")
    avg  = n_l2 / n_q if n_q else 0
    print(f"{method:<20} {n_q:>8} {n_l2:>8} {n_l1:>8} {n_nf:>10} {avg:>10.2f}")

# Per-query label-2 count across methods
print(f"\n{'QID':<8} {'element':>10} {'maxmin':>10} {'recursive':>10}  {'status':}")
print("-" * 65)

l2_map = defaultdict(lambda: defaultdict(list))
for r in rows:
    if r["label"] == "2":
        l2_map[r["query_id"]][r["method"]].append(r["chunk_id"])

needs_review = []
for qid in ALL_QIDS:
    el  = len(l2_map[qid].get("element", []))
    mx  = len(l2_map[qid].get("maxmin_semantic", []))
    rc  = len(l2_map[qid].get("recursive", []))
    note = ""
    if el == 0 or mx == 0 or rc == 0:
        note = "⚠ MISSING_LABEL2"
        needs_review.append((qid, f"el={el} mx={mx} rc={rc}"))
    elif el == 1 or mx == 1 or rc == 1:
        # Check if single label2 has low confidence
        for method, key in [("element","el"), ("maxmin_semantic","mx"), ("recursive","rc")]:
            chunks = l2_map[qid].get(method, [])
            if len(chunks) == 1:
                match = [r for r in rows if r["query_id"]==qid and r["method"]==method and r["label"]=="2"]
                if match and match[0]["confidence"] == "low":
                    note = f"⚡ low_conf ({method})"
                    needs_review.append((qid, f"{method} single low-conf chunk={chunks[0]}"))
    print(f"{qid:<8} {el:>10} {mx:>10} {rc:>10}  {note}")

# Confidence distribution
print(f"\n=== Confidence Distribution (Label-2 only) ===")
conf_dist = defaultdict(int)
for r in rows:
    if r["label"] == "2":
        conf_dist[r["confidence"]] += 1
for k, v in sorted(conf_dist.items()):
    pct = v / sum(conf_dist.values()) * 100
    print(f"  {k:<8}: {v:>4} ({pct:.1f}%)")

# Needs review list
if needs_review:
    print(f"\n=== {len(needs_review)} Query/Method Perlu Validasi Manual ===")
    for qid, note in needs_review:
        print(f"  ! {qid}: {note}")
else:
    print(f"\n✓ Semua query memiliki label-2 di ketiga method.")

print(f"\nFile: {CSV_PATH}")
