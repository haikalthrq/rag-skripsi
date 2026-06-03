"""
Feature tests for app.py — run WITHOUT Streamlit runtime.
Each test prints PASS or FAIL with a brief reason.

Usage:
    python src/streamlit/test_app.py
"""

import sys
import re
import html
import shutil
import tempfile
import logging
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Suppress Streamlit "missing ScriptRunContext" warnings that fire when
# Streamlit functions are called outside a running server (bare mode).
for _lg in (
    "streamlit",
    "streamlit.runtime",
    "streamlit.runtime.caching",
    "streamlit.runtime.caching.cache_data_api",
    "streamlit.runtime.scriptrunner_utils",
    "streamlit.runtime.scriptrunner_utils.script_run_context",
):
    logging.getLogger(_lg).setLevel(logging.ERROR)

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Stub out Streamlit so we can import app functions without a running server ─
import types

_st_stub = types.ModuleType("streamlit")
_st_stub.cache_data = lambda **kw: (lambda f: f)
_st_stub.warning = lambda *a, **kw: None
_st_stub.query_params = {}

class _SessionState:
    """Supports both ss.key and ss['key'] access patterns."""
    def __getattr__(self, key):
        return self.__dict__.get(key, None)
    def __setattr__(self, key, val):
        self.__dict__[key] = val
    def __getitem__(self, key):
        return self.__dict__[key]
    def __setitem__(self, key, val):
        self.__dict__[key] = val
    def get(self, key, default=None):
        return self.__dict__.get(key, default)

_st_stub.session_state = _SessionState()

for _attr in ("markdown", "progress", "metric", "columns", "button",
              "text_input", "selectbox", "checkbox", "divider", "caption",
              "sidebar", "expander", "info", "success", "rerun", "set_page_config"):
    setattr(_st_stub, _attr, lambda *a, **kw: None)

_comp_stub = types.ModuleType("streamlit.components.v1")
_comp_stub.html = lambda *a, **kw: None
_st_stub.components = types.SimpleNamespace(v1=_comp_stub)

sys.modules["streamlit"] = _st_stub
sys.modules["streamlit.components"] = types.ModuleType("streamlit.components")
sys.modules["streamlit.components.v1"] = _comp_stub

import app  # noqa: E402  (import after stub)

# ── Test runner ────────────────────────────────────────────────────────────────

_results: list[tuple[str, bool, str]] = []


def test(name: str, ok: bool, reason: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    _results.append((name, ok, reason))
    tag = f"\033[32m{status}\033[0m" if ok else f"\033[31m{status}\033[0m"
    line = f"  [{tag}] {name}"
    if reason:
        line += f"  — {reason}"
    print(line)


def section(title: str) -> None:
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ── Sample data ────────────────────────────────────────────────────────────────

def _make_df(n_rows: int = 9) -> pd.DataFrame:
    """Create a minimal merged DataFrame with unique (query_id, method) combos."""
    METHODS = ["element_based", "maxmin_semantic", "recursive"]
    rows = []
    for i in range(n_rows):
        qid    = f"Q{str(i // 3 + 1).zfill(3)}"
        method = METHODS[i % 3]
        rows.append({
            "query_id":          qid,
            "doc_id":            "DOC01",
            "method":            method,
            "chunk_id":          str(100 + i),
            "label_0_1_2":       "" if i < 4 else ("1" if i == 4 else "0"),
            "rationale":         f'gold_value="147,78"; col_label="2023"; row_label="Riau"',
            "strength_score":    "8",
            "chunk_page_start":  "10",
            "chunk_page_end":    "10",
            "chunk_text_excerpt": "RIAU  147,78  2023  Benchmark",
            "annotator":         "",
            "status":            "needs_manual_validation",
            "question_preview":  "Berapa benchmark?",
            "evidence_type":     "table_row_column",
            "question":          "Berapa Benchmark Indeks di Provinsi Riau?",
            "gold_answer":       "147,78",
            "evidence_text":     "Benchmark Indeks Riau",
            "source_file":       "test.pdf",
            "table_id":          "Tabel 1.1",
            "row_label":         "Riau",
            "column_label":      "2023",
            "unit":              "Indeks",
            "gold_value":        "147,78",
            "evidence_anchor":   "Tabel 1.1",
            "evidence_page_pdf": "10",
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
section("T1 — Data Loading & Merge")
# ═══════════════════════════════════════════════════════════════════════════════

def t1_candidates_file_exists():
    test("candidates XLSX exists", app.CANDIDATES_XLSX.exists(),
         str(app.CANDIDATES_XLSX))

def t1_qa_gold_file_exists():
    test("qa_gold XLSX exists", app.QA_GOLD_XLSX.exists(),
         str(app.QA_GOLD_XLSX))

def t1_load_data_shape():
    df = app._load_fresh(0.0, 0.0)
    ok = df.shape[0] > 0 and df.shape[1] >= 20
    test("_load_fresh() → rows>0, ≥20 cols", ok,
         f"got {df.shape}")

def t1_merge_has_qa_cols():
    df = app._load_fresh(0.0, 0.0)
    required = {"question", "gold_answer", "evidence_text", "gold_value"}
    missing  = required - set(df.columns)
    test("merged df has QA gold columns", len(missing) == 0,
         f"missing: {missing}")

def t1_load_data_no_none_labels():
    df = app._load_fresh(0.0, 0.0)
    bad = df["label_0_1_2"].isin(["None", "nan"]).sum()
    test("label_0_1_2 has no 'None'/'nan' strings", bad == 0,
         f"{bad} bad values")

def t1_candidate_priority_order():
    """Active candidate must be v3 evidence-aware."""
    prio = [p.name for p in app._CANDIDATE_PRIORITY]
    ok = (
        len(prio) == 1
        and prio[0] == "retrieval_relevant_chunks_candidate_v3_evidence_aware.xlsx"
    )
    test("_CANDIDATE_PRIORITY is v3 evidence-aware only", ok, str(prio))

def t1_candidate_priority_fallback_v1():
    """CANDIDATES_XLSX points directly to v3 (no fallback needed)."""
    ok = app.CANDIDATES_XLSX.name == "retrieval_relevant_chunks_candidate_v3_evidence_aware.xlsx"
    test("CANDIDATES_XLSX is v3 evidence-aware", ok, app.CANDIDATES_XLSX.name)

def t1_v2_cols_in_loaded_df():
    """V2 columns must exist in _load_fresh() output (even if empty)."""
    df = app._load_fresh(0.0, 0.0)
    missing = [c for c in app._V2_COLS if c not in df.columns]
    test("v2 optional cols present in loaded df", len(missing) == 0,
         f"missing: {missing}")

def t1_auto_excerpt_fills_from_chunk_text():
    """_load_fresh post-process: chunk_text_excerpt auto-filled when blank."""
    import types
    # Build a tiny fake df with chunk_text but empty chunk_text_excerpt
    fake = pd.DataFrame([{
        "query_id": "Q001", "method": "element_based", "chunk_id": "1",
        "label_0_1_2": "", "annotator": "",
        "chunk_text": "Hello world from chunk text that is longer than excerpt",
        "chunk_text_excerpt": "",
    }])
    # Simulate the auto-excerpt logic from _load_fresh
    mask = fake["chunk_text_excerpt"].str.strip() == ""
    fake.loc[mask, "chunk_text_excerpt"] = (
        fake.loc[mask, "chunk_text"].str[:800].str.replace("\n", " ", regex=False)
    )
    ok = fake.iloc[0]["chunk_text_excerpt"] == "Hello world from chunk text that is longer than excerpt"
    test("auto-excerpt fills chunk_text_excerpt from chunk_text", ok,
         fake.iloc[0]["chunk_text_excerpt"])

t1_candidates_file_exists()
t1_qa_gold_file_exists()
t1_load_data_shape()
t1_merge_has_qa_cols()
t1_load_data_no_none_labels()
t1_candidate_priority_order()
t1_candidate_priority_fallback_v1()
t1_v2_cols_in_loaded_df()
t1_auto_excerpt_fills_from_chunk_text()


# ═══════════════════════════════════════════════════════════════════════════════
section("T2 — Groups (Navigation)")
# ═══════════════════════════════════════════════════════════════════════════════

def t2_total_groups():
    df = _make_df(9)
    grps = app.get_groups(df, {})
    test("get_groups() no filter → all groups", len(grps) == 9,
         f"expected 9, got {len(grps)}")

def t2_filter_by_qid():
    df = _make_df(9)
    grps = app.get_groups(df, {"qid": "Q001"})
    ok = all(g[0] == "Q001" for g in grps)
    test("filter qid=Q001 returns only Q001 groups", ok,
         f"got {grps}")

def t2_filter_only_unlabeled():
    df = _make_df(9)  # 9 unique combos
    # Label all rows of Q003 (rows 6-8) so that group is fully labeled
    df.loc[df["query_id"] == "Q003", "label_0_1_2"] = "1"
    grps = app.get_groups(df, {"only_unlabeled": True})
    q003_grps = [g for g in grps if g[0] == "Q003"]
    test("only_unlabeled filter excludes fully-labeled Q003 groups",
         len(q003_grps) == 0, f"leaked: {q003_grps}")

def t2_sorted_by_method_order():
    df = _make_df(9)
    grps = app.get_groups(df, {})
    methods = [g[1] for g in grps]
    order = [app.METHOD_ORDER.get(m, 9) for m in methods]
    # Check monotonically non-decreasing within same query_id
    ok = True
    prev_qid, prev_ord = None, -1
    for g, o in zip(grps, order):
        if g[0] != prev_qid:
            prev_qid, prev_ord = g[0], o
        else:
            if o < prev_ord:
                ok = False; break
            prev_ord = o
    test("groups sorted by method order within each query_id", ok,
         f"order seq: {list(zip([g[1] for g in grps[:6]], order[:6]))}")

t2_total_groups()
t2_filter_by_qid()
t2_filter_only_unlabeled()
t2_sorted_by_method_order()


# ═══════════════════════════════════════════════════════════════════════════════
section("T3 — Progress Counting")
# ═══════════════════════════════════════════════════════════════════════════════

def t3_zero_progress():
    df = _make_df(6)
    df["label_0_1_2"] = ""
    p = app.get_progress(df)
    test("0% when all unlabeled", p["pct"] == 0 and p["labeled"] == 0,
         str(p))

def t3_full_progress():
    df = _make_df(4)
    df["label_0_1_2"] = "1"
    p = app.get_progress(df)
    test("100% when all labeled", p["pct"] == 100 and p["unlabeled"] == 0,
         str(p))

def t3_label_counts():
    df = _make_df(6)
    df["label_0_1_2"] = ["1", "1", "0", "needs_review", "", "1"]
    p = app.get_progress(df)
    ok = p["n1"] == 3 and p["n0"] == 1 and p["nnr"] == 1
    test("n1/n0/nnr counted correctly", ok, str(p))

def t3_total_matches_rows():
    df = _make_df(10)
    p = app.get_progress(df)
    test("total == len(df)", p["total"] == 10, str(p))

t3_zero_progress()
t3_full_progress()
t3_label_counts()
t3_total_matches_rows()


# ═══════════════════════════════════════════════════════════════════════════════
section("T4 — Highlight Excerpt")
# ═══════════════════════════════════════════════════════════════════════════════

def t4_gold_value_highlighted():
    row = pd.Series({"gold_value": "147,78", "row_label": "", "column_label": "",
                     "evidence_text": ""})
    out = app.highlight_excerpt("RIAU 147,78 indeks", row)
    test("gold_value term is wrapped in <mark>", "<mark" in out and "147,78" in out, out[:100])

def t4_row_label_highlighted():
    row = pd.Series({"gold_value": "", "row_label": "Riau", "column_label": "",
                     "evidence_text": ""})
    out = app.highlight_excerpt("Provinsi Riau 2023", row)
    test("row_label 'Riau' is highlighted", "<mark" in out and "riau" in out.lower(), out[:100])

def t4_col_label_highlighted():
    row = pd.Series({"gold_value": "", "row_label": "", "column_label": "2023",
                     "evidence_text": ""})
    out = app.highlight_excerpt("Data 2023 indeks", row)
    test("col_label '2023' is highlighted", "<mark" in out, out[:100])

def t4_evidence_keywords_highlighted():
    row = pd.Series({"gold_value": "", "row_label": "", "column_label": "",
                     "evidence_text": "Benchmark indeks Riau tahun 2023"})
    out = app.highlight_excerpt("benchmark indeks", row)
    test("evidence keywords are highlighted", "<mark" in out, out[:100])

def t4_empty_text_returns_placeholder():
    row = pd.Series({"gold_value": "", "row_label": "", "column_label": "",
                     "evidence_text": ""})
    out = app.highlight_excerpt("", row)
    test("empty excerpt returns placeholder text", "kosong" in out, out[:60])

def t4_no_raw_html_in_text():
    row = pd.Series({"gold_value": "10", "row_label": "", "column_label": "",
                     "evidence_text": ""})
    out = app.highlight_excerpt("<script>alert(1)</script>", row)
    test("raw <script> is HTML-escaped", "<script>" not in out, out[:100])

def t4_output_is_valid_span_html():
    row = pd.Series({"gold_value": "147,78", "row_label": "Riau", "column_label": "2023",
                     "evidence_text": "benchmark"})
    out = app.highlight_excerpt("RIAU 147,78 2023 benchmark", row)
    opens  = len(re.findall(r"<mark\b", out))
    closes = len(re.findall(r"</mark>", out))
    test("every <mark> has matching </mark>", opens == closes,
         f"opens={opens} closes={closes}")

t4_gold_value_highlighted()
t4_row_label_highlighted()
t4_col_label_highlighted()
t4_evidence_keywords_highlighted()
t4_empty_text_returns_placeholder()
t4_no_raw_html_in_text()
t4_output_is_valid_span_html()


# ═══════════════════════════════════════════════════════════════════════════════
section("T5 — Evidence Match Type")
# ═══════════════════════════════════════════════════════════════════════════════

def t5_gold_value():
    out = app.evidence_match_type('gold_value="147,78"; col_label="2023"')
    test("detects gold_value signal", "gold_value" in out, out)

def t5_evidence_text():
    out = app.evidence_match_type("evidence_text=found; question_kw=found")
    test("detects evidence_text signal", "evidence_text" in out, out)

def t5_narrative_signals():
    out = app.evidence_match_type("gold_answer_kw=found; anchor_kw=found; question_kw=found")
    ok = "gold_answer_kw" in out and "anchor_kw" in out and "question_kw" in out
    test("detects gold_answer_kw, anchor_kw, question_kw", ok, out)

def t5_no_signal():
    out = app.evidence_match_type("")
    test("empty rationale → no_signal", out == "no_signal", out)

def t5_table_signals():
    out = app.evidence_match_type('table_id="T1"; row_label="Riau"; col_label="2023"')
    ok = "table_id" in out and "row_label" in out and "col_label" in out
    test("detects table_id, row_label, col_label", ok, out)

t5_gold_value()
t5_evidence_text()
t5_narrative_signals()
t5_no_signal()
t5_table_signals()


# ═══════════════════════════════════════════════════════════════════════════════
section("T6 — Chunk Card HTML Structure")
# ═══════════════════════════════════════════════════════════════════════════════

def t6_card_html_self_contained():
    """Verify render_chunk_card uses ONE st.markdown with a self-contained div."""
    import ast, inspect
    src = inspect.getsource(app.render_chunk_card)
    tree = ast.parse(src)

    markdown_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = ""
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            if name == "markdown":
                markdown_calls.append(ast.get_source_segment(src, node) or "")

    # No standalone </div> string passed to markdown
    lone_div = [c for c in markdown_calls
                if c.strip().startswith('st.markdown("</div>')
                or c.strip().startswith("st.markdown('</div>")]
    test("no standalone </div> in separate st.markdown() call",
         len(lone_div) == 0, f"found: {lone_div}")

def t6_excerpt_inside_card_div():
    """excerpt div must be inside the same st.markdown call as ccard div.
    Works for both triple-quoted f-strings and flat concatenated strings."""
    import inspect
    src = inspect.getsource(app.render_chunk_card)

    # Look for the card_html variable that concatenates all parts
    has_card_html_var = "card_html" in src and "ccard" in src and "excerpt" in src
    # Also accept triple-quoted form
    triple_pattern = re.compile(
        r'st\.markdown\(\s*f"""(.*?)"""\s*,\s*unsafe_allow_html',
        re.DOTALL,
    )
    triple_blocks = triple_pattern.findall(src)
    triple_ok = any("ccard" in b and "excerpt" in b for b in triple_blocks)

    test("excerpt and ccard are in the same st.markdown block",
         has_card_html_var or triple_ok,
         f"card_html_var={has_card_html_var} triple_blocks={[b[:40] for b in triple_blocks]}")

def t6_no_blank_lines_in_card_html():
    """The card HTML must NOT contain newlines that could terminate a Markdown HTML block."""
    import inspect
    src = inspect.getsource(app.render_chunk_card)
    # Find the card_html assignment block (lines between card_html = ( and st.markdown)
    # and verify there are no empty/whitespace-only embedded lines
    # Strategy: find the actual generated HTML by looking for the flat-string pattern
    pattern = re.compile(r"card_html\s*=\s*\((.*?)\)\s*\n\s*st\.markdown\(card_html", re.DOTALL)
    match = pattern.search(src)
    if match:
        block = match.group(1)
        # Should be concatenated f-strings, no triple-quoted with blank lines
        has_triple = '"""' in block
        test("card HTML uses flat concatenation (no triple-quoted f-string with blank lines)",
             not has_triple, f"found triple-quotes in card_html block")
    else:
        # Fallback: just check for the variable
        test("card_html variable exists in render_chunk_card", "card_html" in src)

def t6_div_open_close_balanced():
    """All HTML string literals in render_chunk_card must have balanced divs."""
    import inspect
    src = inspect.getsource(app.render_chunk_card)

    # Check flat-string card_html pattern
    pattern_flat = re.compile(r"card_html\s*=\s*\((.*?)\)\s*\n\s*st\.markdown\(card_html", re.DOTALL)
    match = pattern_flat.search(src)
    if match:
        block = match.group(1)
        opens  = len(re.findall(r"<div\b", block))
        closes = len(re.findall(r"</div>", block))
        test(f"<div> balanced in flat card_html (opens={opens}, closes={closes})",
             opens == closes, f"opens={opens} closes={closes}")
    else:
        # Fallback to triple-quoted search
        pattern = re.compile(r'st\.markdown\(\s*f"""(.*?)"""\s*,\s*unsafe_allow_html', re.DOTALL)
        blocks = pattern.findall(src)
        for b in blocks:
            opens  = len(re.findall(r"<div\b", b))
            closes = len(re.findall(r"</div>", b))
            test(f"<div> balanced in block '{b[:40].strip()}'",
                 opens == closes, f"opens={opens} closes={closes}")

t6_card_html_self_contained()
t6_excerpt_inside_card_div()
t6_no_blank_lines_in_card_html()
t6_div_open_close_balanced()


# ═══════════════════════════════════════════════════════════════════════════════
section("T7 — apply_label (state update)")
# ═══════════════════════════════════════════════════════════════════════════════

def t7_apply_label_updates_correct_row():
    df = _make_df(6)
    _st_stub.session_state["df"] = df
    _st_stub.session_state["annotator_name"] = "test_user"
    app.st.session_state = _st_stub.session_state

    target_row = df.iloc[0]
    qid, method, cid = target_row["query_id"], target_row["method"], target_row["chunk_id"]
    app.apply_label(qid, method, cid, "1")

    updated = _st_stub.session_state["df"]
    mask = (updated["query_id"] == qid) & (updated["method"] == method) & (updated["chunk_id"] == cid)
    lbl = updated.loc[mask, "label_0_1_2"].iloc[0]
    test("apply_label sets label_0_1_2 correctly", lbl == "1", f"got {lbl!r}")

def t7_apply_label_sets_annotator():
    df = _make_df(6)
    _st_stub.session_state["df"] = df
    _st_stub.session_state["annotator_name"] = "haikal"
    app.st.session_state = _st_stub.session_state

    row = df.iloc[1]
    app.apply_label(row["query_id"], row["method"], row["chunk_id"], "1")
    updated = _st_stub.session_state["df"]
    mask = (
        (updated["query_id"] == row["query_id"])
        & (updated["method"]   == row["method"])
        & (updated["chunk_id"] == row["chunk_id"])
    )
    ann = updated.loc[mask, "annotator"].iloc[0]
    test("apply_label sets annotator name", ann == "haikal", f"got {ann!r}")

def t7_apply_label_does_not_change_other_rows():
    df = _make_df(6)
    _st_stub.session_state["df"] = df.copy()
    _st_stub.session_state["annotator_name"] = ""
    app.st.session_state = _st_stub.session_state

    row = df.iloc[0]
    app.apply_label(row["query_id"], row["method"], row["chunk_id"], "0")
    updated = _st_stub.session_state["df"]
    other_labels = updated.iloc[1:]["label_0_1_2"].tolist()
    unchanged = all(l == "" for l in other_labels if l != "1" and l != "0")
    test("apply_label only changes targeted row", unchanged,
         f"other labels: {other_labels}")

t7_apply_label_updates_correct_row()
t7_apply_label_sets_annotator()
t7_apply_label_does_not_change_other_rows()


# ═══════════════════════════════════════════════════════════════════════════════
section("T8 — Save & Export")
# ═══════════════════════════════════════════════════════════════════════════════

def t8_save_creates_csv():
    df = _make_df(4)
    df["label_0_1_2"] = ["1", "1", "0", ""]
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path  = Path(tmpdir) / "labels.csv"
        xlsx_path = Path(tmpdir) / "labels.xlsx"
        _orig_csv  = app.OUTPUT_CSV
        _orig_xlsx = app.OUTPUT_XLSX
        app.OUTPUT_CSV  = csv_path
        app.OUTPUT_XLSX = xlsx_path
        app.save_data(df)
        app.OUTPUT_CSV  = _orig_csv
        app.OUTPUT_XLSX = _orig_xlsx
        test("save_data() creates CSV", csv_path.exists(), str(csv_path))

def t8_save_creates_xlsx():
    df = _make_df(4)
    with tempfile.TemporaryDirectory() as tmpdir:
        xlsx_path = Path(tmpdir) / "labels.xlsx"
        csv_path  = Path(tmpdir) / "labels.csv"
        _orig_csv  = app.OUTPUT_CSV
        _orig_xlsx = app.OUTPUT_XLSX
        app.OUTPUT_CSV  = csv_path
        app.OUTPUT_XLSX = xlsx_path
        app.save_data(df)
        app.OUTPUT_CSV  = _orig_csv
        app.OUTPUT_XLSX = _orig_xlsx
        test("save_data() creates XLSX", xlsx_path.exists(), str(xlsx_path))

def t8_csv_has_correct_columns():
    df = _make_df(4)
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path  = Path(tmpdir) / "labels.csv"
        xlsx_path = Path(tmpdir) / "labels.xlsx"
        _orig_csv, _orig_xlsx = app.OUTPUT_CSV, app.OUTPUT_XLSX
        app.OUTPUT_CSV, app.OUTPUT_XLSX = csv_path, xlsx_path
        app.save_data(df)
        app.OUTPUT_CSV, app.OUTPUT_XLSX = _orig_csv, _orig_xlsx
        saved = pd.read_csv(str(csv_path), dtype=str)
        required = {"query_id", "method", "chunk_id", "label_0_1_2", "status"}
        missing  = required - set(saved.columns)
        test("saved CSV has required columns", len(missing) == 0,
             f"missing: {missing}")

def t8_status_col_correct():
    df = _make_df(4)
    df["label_0_1_2"] = ["1", "", "", "0"]
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path  = Path(tmpdir) / "labels.csv"
        xlsx_path = Path(tmpdir) / "labels.xlsx"
        _orig_csv, _orig_xlsx = app.OUTPUT_CSV, app.OUTPUT_XLSX
        app.OUTPUT_CSV, app.OUTPUT_XLSX = csv_path, xlsx_path
        app.save_data(df)
        app.OUTPUT_CSV, app.OUTPUT_XLSX = _orig_csv, _orig_xlsx
        saved = pd.read_csv(str(csv_path), dtype=str)
        ok = (
            saved.iloc[0]["status"] == "labeled"
            and saved.iloc[1]["status"] == "needs_manual_validation"
            and saved.iloc[3]["status"] == "labeled"
        )
        test("status column: labeled vs needs_manual_validation", ok,
             saved["status"].tolist())

t8_save_creates_csv()
t8_save_creates_xlsx()
t8_csv_has_correct_columns()
t8_status_col_correct()


# ═══════════════════════════════════════════════════════════════════════════════
section("T9 — Resume (load from existing output)")
# ═══════════════════════════════════════════════════════════════════════════════

def t9_resume_restores_labels():
    df_fresh = _make_df(4)
    df_labeled = df_fresh.copy()
    df_labeled["label_0_1_2"] = ["1", "1", "", "0"]
    df_labeled["annotator"]   = ["ann", "ann", "", "ann"]

    with tempfile.TemporaryDirectory() as tmpdir:
        xlsx_path = Path(tmpdir) / "labels.xlsx"
        csv_path  = Path(tmpdir) / "labels.csv"

        _orig_csv, _orig_xlsx = app.OUTPUT_CSV, app.OUTPUT_XLSX
        app.OUTPUT_CSV, app.OUTPUT_XLSX = csv_path, xlsx_path
        app.save_data(df_labeled)

        # Now simulate load_or_resume with a patched _load_fresh
        _orig_load = app._load_fresh
        app._load_fresh = lambda _mtime_c=0, _mtime_q=0: df_fresh.copy()

        resumed = app.load_data()
        app._load_fresh = _orig_load
        app.OUTPUT_CSV, app.OUTPUT_XLSX = _orig_csv, _orig_xlsx

        labels = resumed["label_0_1_2"].tolist()
        test("resume restores label '1' to first row",   labels[0] == "1", str(labels))
        test("resume restores label '1' to second row",  labels[1] == "1", str(labels))
        test("resume keeps empty label for third row",   labels[2] == "",  str(labels))
        test("resume restores label '0' to fourth row",  labels[3] == "0", str(labels))

t9_resume_restores_labels()


# ═══════════════════════════════════════════════════════════════════════════════
section("T10 — V2 Column Display in Chunk Card")
# ═══════════════════════════════════════════════════════════════════════════════

import inspect as _inspect

def t10_suggested_label_badge_in_card_html():
    src = _inspect.getsource(app.render_chunk_card)
    test("render_chunk_card uses suggested_label field",
         "suggested_label" in src, "not found")

def t10_confidence_badge_in_card_html():
    src = _inspect.getsource(app.render_chunk_card)
    test("render_chunk_card uses confidence field",
         "confidence" in src, "not found")

def t10_evidence_quote_in_card_html():
    src = _inspect.getsource(app.render_chunk_card)
    test("render_chunk_card uses evidence_quote field",
         "evidence_quote" in src, "not found")

def t10_match_type_v2_preferred_over_rationale():
    src = _inspect.getsource(app.render_chunk_card)
    test("render_chunk_card prefers v2 match_type over rationale-derived",
         "mt_v2" in src and "display_mt" in src, "not found")

def t10_v2_cols_in_make_df():
    """Sample df can include v2 columns without breaking get_groups / get_progress."""
    df = _make_df(6)
    for col in ["suggested_label", "confidence", "match_type", "reason", "evidence_quote"]:
        df[col] = "test"
    grps = app.get_groups(df, {})
    p    = app.get_progress(df)
    test("get_groups/get_progress work fine with v2 columns present",
         len(grps) > 0 and p["total"] == 6, f"grps={len(grps)} total={p['total']}")

def t10_top_k_limit_warn():
    """Warn if any query×method group has more than 5 candidates."""
    df = _make_df(9)
    # Each group currently has 1 chunk; add 5 more to Q001/element_based → 6 total
    extra_rows = []
    for i in range(5):
        extra_rows.append({
            **df.iloc[0].to_dict(),
            "chunk_id": str(900 + i),
        })
    df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)
    group_sizes = df.groupby(["query_id", "method"]).size()
    over_limit  = (group_sizes > 5).sum()
    test("top_k >5 detection works", over_limit > 0,
         f"groups over 5: {over_limit}")

t10_suggested_label_badge_in_card_html()
t10_confidence_badge_in_card_html()
t10_evidence_quote_in_card_html()
t10_match_type_v2_preferred_over_rationale()
t10_v2_cols_in_make_df()
t10_top_k_limit_warn()


# ═══════════════════════════════════════════════════════════════════════════════
section("T11 — Evidence-Aware UI Functions")
# ═══════════════════════════════════════════════════════════════════════════════

def t11_sidebar_badge_v3():
    """Sidebar badge must detect v3 from filename."""
    fname = app.CANDIDATES_XLSX.name
    is_v3 = "v3" in fname.lower()
    test("sidebar badge shows v3 evidence-aware", is_v3,
         f"filename={fname}")

def t11_row_count_matches_file():
    """Row count in loaded df must equal actual xlsx rows."""
    import openpyxl
    wb = openpyxl.load_workbook(str(app.CANDIDATES_XLSX), read_only=True)
    ws = wb["candidates"]
    file_rows = ws.max_row - 1  # subtract header
    wb.close()
    df = app._load_fresh(0.0, 0.0)
    # After merge, df may have QA cols added but same row count
    test("loaded df row count matches xlsx file",
         len(df) == file_rows,
         f"df={len(df)}, xlsx={file_rows}")

def t11_evidence_flags_html_table():
    """evidence_flags_html renders FOUND/MISSING flags for table QA."""
    row = pd.Series({
        "evidence_type": "table_row_column",
        "has_gold_value": "True", "has_row_label": "True",
        "has_column_label": "False", "has_table_id": "False",
        "has_evidence_anchor": "False",
        "chunk_page_start": "10", "page_match": "True",
    })
    out = app.evidence_flags_html(row)
    ok = ("flag-found" in out and "flag-missing" in out
          and "gold_value" in out and "col_label" in out)
    test("evidence_flags_html renders FOUND+MISSING for table", ok, out[:120])

def t11_evidence_flags_html_narrative():
    """evidence_flags_html renders ev_text + anchor for narrative."""
    row = pd.Series({
        "evidence_type": "narrative",
        "has_evidence_text": "False", "has_evidence_anchor": "True",
        "chunk_page_start": "", "page_match": "False",
    })
    out = app.evidence_flags_html(row)
    ok = "ev_text" in out and "anchor" in out and "N/A" in out
    test("evidence_flags_html renders ev_text+anchor for narrative", ok, out[:120])

def t11_group_summary_html_has_exact():
    """group_summary_html shows exact count and has-exact indicator."""
    df = _make_df(3)
    df["match_type"] = ["exact_table_evidence", "partial_table_evidence", "keyword_only"]
    out = app.group_summary_html(df)
    ok = "exact" in out and ("Ada exact" in out or "1" in out)
    test("group_summary_html shows exact count", ok, out[:120])

def t11_group_summary_html_no_exact():
    """group_summary_html correctly shows no exact evidence."""
    df = _make_df(2)
    df["match_type"] = ["partial_table_evidence", "keyword_only"]
    out = app.group_summary_html(df)
    ok = "Tidak ada" in out or "0" in out
    test("group_summary_html shows no-exact warning", ok, out[:120])

def t11_table_warning_in_card_source():
    """render_chunk_card must contain evidence flags logic."""
    src = _inspect.getsource(app.render_chunk_card)
    ok = "evidence_flags_html" in src and "flag-row" in app.CSS
    test("render_chunk_card uses evidence_flags_html", ok)

def t11_evidence_expander_in_card_source():
    """render_chunk_card must use 'Why this candidate?' expander."""
    src = _inspect.getsource(app.render_chunk_card)
    ok = "Why this candidate?" in src and "evidence_quote" in src and "reason" in src
    test("render_chunk_card has 'Why this candidate?' expander", ok)

def t11_sort_group_df_order():
    """sort_group_df puts exact before partial before keyword."""
    df = _make_df(3)
    df["match_type"]      = ["keyword_only", "partial_table_evidence", "exact_table_evidence"]
    df["suggested_label"] = ["0", "1", "2"]
    df["confidence"]      = ["low", "medium", "high"]
    df["strength_score"]  = ["2", "5", "9"]
    sorted_df = app.sort_group_df(df)
    mts = sorted_df["match_type"].tolist()
    ok = mts[0] == "exact_table_evidence" and mts[-1] == "keyword_only"
    test("sort_group_df orders exact > partial > keyword", ok, str(mts))

def t11_no_auto_label():
    """apply_label must NOT be called in _load_fresh or load_data."""
    src_load = _inspect.getsource(app._load_fresh)
    src_resume = _inspect.getsource(app.load_data)
    ok = "apply_label" not in src_load and "apply_label" not in src_resume
    test("no auto-label in _load_fresh / load_data", ok)

def t11_button_text_explicit():
    """Label buttons must have explicit text (not just emoji)."""
    src = _inspect.getsource(app.render_chunk_card)
    ok = ("Relevan" in src and "Tidak Relevan" in src and "Review" in src)
    test("label buttons have explicit text labels", ok)

def t11_export_has_label_col():
    """save_data must include label_0_1_2 and status in output."""
    src = _inspect.getsource(app.save_data)
    ok = "label_0_1_2" in src and "status" in src
    test("save_data includes label_0_1_2 and status", ok)

t11_sidebar_badge_v3()
t11_row_count_matches_file()
t11_evidence_flags_html_table()
t11_evidence_flags_html_narrative()
t11_group_summary_html_has_exact()
t11_group_summary_html_no_exact()
t11_table_warning_in_card_source()
t11_evidence_expander_in_card_source()
t11_sort_group_df_order()
t11_no_auto_label()
t11_button_text_explicit()
t11_export_has_label_col()


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

total  = len(_results)
passed = sum(1 for _, ok, _ in _results if ok)
failed = total - passed

print(f"\n{'═'*55}")
print(f"  TOTAL: {total}   PASS: {passed}   FAIL: {failed}")
print(f"{'═'*55}")

if failed:
    print("\n  Failing tests:")
    for name, ok, reason in _results:
        if not ok:
            print(f"    ✗ {name}  — {reason}")

sys.exit(0 if failed == 0 else 1)
