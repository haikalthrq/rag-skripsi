import csv
rows = list(csv.DictReader(open("data/ground_truth/qa_gold_validated.csv", encoding="utf-8")))
print("=== Rows with evidence_correction (anchor notes) ===")
for r in rows:
    ec = r.get("evidence_correction", "")
    if ec:
        qid = r["query_id"]
        print(f"  {qid}: {ec[:120]}")

print()
print("=== Sample Q041 ===")
r41 = [r for r in rows if r["query_id"] == "Q041"][0]
for k, v in r41.items():
    if v:
        print(f"  {k}: {str(v)[:130]}")
