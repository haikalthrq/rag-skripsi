# Streamlit Applications

## RAG Evaluation UI

```bash
streamlit run src/streamlit/rag_chat.py
```

Despite the historical filename, the question input is selected from
`data/ground_truth/qa_gold_standard_rag_bps_30qa_question_newest.xlsx`; arbitrary
free-text input is not currently exposed. The UI supports one chunking method or
three-method comparison, Top-k 1-10 for interactive runs, and batch evaluation.

Batch evaluation supports full 30-QA and quick 5-QA modes with an interactive
Top-k range up to 20. CSV files are written to `results/final/generation/`.
Existing files are considered reusable when they are nonempty and contain the
required timing columns. This check does not validate model configuration,
ground-truth revision, full QA/method coverage, or absence of errors; inspect
the CSV before treating a skipped run as equivalent.

Model/backend selection depends on available local assets. Review
`_detect_environment()` and `load_pipeline()` in `rag_chat.py`, and verify the
selected paths shown by the UI before running measurements.

Chat history is runtime data under `results/chat_history/` and is excluded from
source distribution.

## Ground-Truth Annotation UI

```bash
streamlit run src/streamlit/app.py
```

The app expects:

```text
data/ground_truth/retrieval_relevant_chunks_candidate_v3_evidence_aware.xlsx
```

That candidate workbook is not included in a clean source handoff. Build it
with `scripts/build_candidates_v3.py` or provide an approved copy before running
the annotation UI. Output is written to `retrieval_labels_final.xlsx` and
`retrieval_labels_final.csv` in `data/ground_truth/`.
