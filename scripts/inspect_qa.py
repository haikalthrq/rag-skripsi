import pandas as pd

qa = pd.read_excel(
    'data/ground_truth/qa_gold_standard_rag_bps_30qa_question_newest.xlsx',
    sheet_name='qa_gold', dtype=str
).fillna('')

print('=== QA GOLD SAMPLE ===')
for _, r in qa.iterrows():
    print(r['query_id'], '|', r['evidence_type'], '| doc=', r['doc_id'],
          '| src=', r['source_file'][:50])

print()
print('evidence_types:', qa['evidence_type'].value_counts().to_dict())
print('doc_ids:', qa['doc_id'].unique().tolist())
print('source_files:', qa['source_file'].unique().tolist())
print()
print('Columns:', list(qa.columns))
