import csv

rows = list(csv.DictReader(open("data/ground_truth/retrieval_ground_truth.csv", encoding="utf-8")))

print("Total rows:", len(rows))
print("Label 2:", sum(1 for r in rows if r["label"] == "2"))
print("Label 1:", sum(1 for r in rows if r["label"] == "1"))

# Sample label=2 per method
for method in ["element", "maxmin_semantic", "recursive"]:
    subset = [r for r in rows if r["method"] == method and r["label"] == "2"][:3]
    print(f"\n--- {method} (label=2 samples) ---")
    for r in subset:
        qid = r["query_id"]
        cid = r["chunk_id"]
        conf = r["confidence"]
        rat = r["rationale"][:75]
        print(f"  {qid} chunk={cid} conf={conf} | {rat}")

# Check low confidence label-2
low_conf_l2 = [r for r in rows if r["label"] == "2" and r["confidence"] == "low"]
print(f"\nLabel-2 confidence=low: {len(low_conf_l2)}")
for r in low_conf_l2[:15]:
    qid = r["query_id"]
    method = r["method"]
    cid = r["chunk_id"]
    rat = r["rationale"][:65]
    print(f"  {qid}/{method}/chunk_{cid} | {rat}")

# Check per-query coverage: how many label-2 per query×method
print("\n=== Queries with only 1 label-2 chunk per method ===")
from collections import defaultdict
coverage = defaultdict(lambda: defaultdict(list))
for r in rows:
    if r["label"] == "2":
        coverage[r["query_id"]][r["method"]].append(r["chunk_id"])

sparse = []
for qid in sorted(coverage):
    for method in ["element", "maxmin_semantic", "recursive"]:
        chunks = coverage[qid].get(method, [])
        if len(chunks) == 0:
            sparse.append((qid, method, "NO_LABEL2"))
        elif len(chunks) == 1 and any(
            r["confidence"] == "low"
            for r in rows
            if r["query_id"] == qid and r["method"] == method and r["label"] == "2"
        ):
            sparse.append((qid, method, f"single_low_conf_chunk={chunks[0]}"))

for qid, method, note in sparse[:20]:
    print(f"  {qid}/{method}: {note}")

print(f"\nTotal sparse/missing label-2: {len(sparse)}")
