"""
Unit tests untuk evaluation pipeline RAG.

Coverage:
  T1  — compute_bleu: perfect match → 1.0
  T2  — compute_bleu: zero match → 0.0
  T3  — compute_bleu: partial match (rentang 0–1)
  T4  — compute_bleu: empty response → 0.0
  T5  — compute_rouge: perfect match → 1.0
  T6  — compute_rouge: zero match → 0.0
  T7  — compute_rouge: partial match (rentang 0–1)
  T8  — compute_rouge: menggunakan mode='recall' (bukan precision/fmeasure)
  T9  — compute_rouge: empty response → 0.0
  T10 — compute_precision_at_k: semua relevan
  T11 — compute_precision_at_k: tidak ada yang relevan
  T12 — compute_precision_at_k: sebagian relevan
  T13 — compute_recall_at_k: semua relevan ditemukan
  T14 — compute_recall_at_k: tidak ada yang ditemukan
  T15 — compute_recall_at_k: empty relevant_ids → 0.0
  T16 — compute_mrr: relevan di posisi 1 → 1.0
  T17 — compute_mrr: relevan di posisi 2 → 0.5
  T18 — compute_mrr: tidak ada yang relevan → 0.0
  T19 — retrieval metrics: duplikat retrieved ID tidak menaikkan skor
  T20 — retrieval metrics: k <= 0 aman dan menghasilkan 0.0
  T21 — compute_f1_at_k: harmonic mean Precision@k dan Recall@k
  T22 — build_summary: agregasi mean BLEU + ROUGE-L per method
  T23 — load_qa_gold: baca xlsx dan validasi field wajib

Jalankan:
  python -m pytest tests/test_evaluation.py -v
  python tests/test_evaluation.py
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (
    compute_bleu,
    compute_rouge,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_mrr,
    compute_f1_at_k,
)


# ── Helper ─────────────────────────────────────────────────────────────────────

def _rouge_l_recall_manual(response: str, reference: str) -> float:
    """
    Implementasi manual ROUGE-L Recall untuk verifikasi silang.
    ROUGE-L-R = LCS(response, reference) / |reference_tokens|
    """
    r_tokens = response.split()
    ref_tokens = reference.split()
    if not ref_tokens:
        return 0.0
    m, n = len(r_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if r_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n] / n


# ── T1–T4: compute_bleu ────────────────────────────────────────────────────────

class TestComputeBleu(unittest.TestCase):

    def test_T1_perfect_match(self):
        """T1: BLEU = 1.0 untuk teks identik."""
        text = "nilai ekspor Indonesia meningkat pada tahun 2023"
        score = compute_bleu(text, text)
        self.assertAlmostEqual(score, 1.0, places=3,
                               msg=f"T1 FAILED: expected ~1.0, got {score}")

    def test_T2_zero_match(self):
        """T2: BLEU = 0.0 untuk teks yang benar-benar berbeda."""
        score = compute_bleu(
            "produk domestik bruto nasional",
            "nilai ekspor komoditas pertanian",
        )
        self.assertEqual(score, 0.0,
                         msg=f"T2 FAILED: expected 0.0, got {score}")

    def test_T3_partial_match(self):
        """T3: BLEU partial antara 0 dan 1."""
        score = compute_bleu(
            "nilai ekspor Indonesia meningkat",
            "nilai ekspor Indonesia meningkat pesat pada tahun 2023",
        )
        self.assertGreater(score, 0.0,  msg=f"T3 FAILED: score should be > 0, got {score}")
        self.assertLess(score,    1.0,  msg=f"T3 FAILED: score should be < 1, got {score}")

    def test_T4_empty_response(self):
        """T4: BLEU = 0.0 untuk response kosong."""
        score = compute_bleu("", "nilai ekspor Indonesia meningkat")
        self.assertEqual(score, 0.0,
                         msg=f"T4 FAILED: expected 0.0, got {score}")


# ── T5–T9: compute_rouge ──────────────────────────────────────────────────────

class TestComputeRouge(unittest.TestCase):

    def test_T5_perfect_match(self):
        """T5: ROUGE-L Recall = 1.0 untuk teks identik."""
        text = "nilai ekspor Indonesia meningkat pada tahun 2023"
        score = compute_rouge(text, text, "rougeL", "recall")
        self.assertAlmostEqual(score, 1.0, places=3,
                               msg=f"T5 FAILED: expected ~1.0, got {score}")

    def test_T6_zero_match(self):
        """T6: ROUGE-L Recall = 0.0 untuk teks benar-benar berbeda."""
        score = compute_rouge(
            "xyz abc def",
            "nilai ekspor komoditas pertanian",
            "rougeL", "recall",
        )
        self.assertEqual(score, 0.0,
                         msg=f"T6 FAILED: expected 0.0, got {score}")

    def test_T7_partial_match(self):
        """T7: ROUGE-L Recall partial antara 0 dan 1."""
        response  = "the cat is not there"
        reference = "the cat is on the mat"
        score = compute_rouge(response, reference, "rougeL", "recall")
        expected  = _rouge_l_recall_manual(response, reference)  # 0.5
        self.assertAlmostEqual(score, expected, places=3,
                               msg=f"T7 FAILED: expected {expected:.4f}, got {score:.4f}")

    def test_T8_mode_is_recall_not_precision(self):
        """
        T8: Mode 'recall' diverifikasi — ROUGE-L Recall != Precision
        ketika panjang response != panjang reference.
        Recall = LCS/|ref|, Precision = LCS/|response|.
        """
        response  = "the cat is on the mat and sat down quietly"
        reference = "the cat is on the mat"
        recall    = compute_rouge(response, reference, "rougeL", "recall")
        precision = compute_rouge(response, reference, "rougeL", "precision")
        self.assertAlmostEqual(recall, 1.0, places=3,
                               msg=f"T8 FAILED: recall should be 1.0, got {recall}")
        self.assertLess(precision, 1.0,
                        msg=f"T8 FAILED: precision should be < 1.0, got {precision}")

    def test_T9_empty_response(self):
        """T9: ROUGE-L Recall = 0.0 untuk response kosong."""
        score = compute_rouge("", "nilai ekspor Indonesia meningkat", "rougeL", "recall")
        self.assertEqual(score, 0.0,
                         msg=f"T9 FAILED: expected 0.0, got {score}")


# ── T10–T12: compute_precision_at_k ──────────────────────────────────────────

class TestComputePrecisionAtK(unittest.TestCase):

    def test_T10_all_relevant(self):
        """T10: Precision@3 = 1.0 semua retrieved relevan."""
        score = compute_precision_at_k(["c1", "c2", "c3"], ["c1", "c2", "c3"], k=3)
        self.assertAlmostEqual(score, 1.0, places=5, msg=f"T10 FAILED: got {score}")

    def test_T11_none_relevant(self):
        """T11: Precision@3 = 0.0 tidak ada yang relevan."""
        score = compute_precision_at_k(["c1", "c2", "c3"], ["c4", "c5"], k=3)
        self.assertAlmostEqual(score, 0.0, places=5, msg=f"T11 FAILED: got {score}")

    def test_T12_partial_relevant(self):
        """T12: Precision@4 = 0.5 (2 dari 4 relevan)."""
        score = compute_precision_at_k(["c1", "c2", "c3", "c4"], ["c1", "c3"], k=4)
        self.assertAlmostEqual(score, 0.5, places=5, msg=f"T12 FAILED: got {score}")

    def test_T19_duplicate_retrieved_ids_count_once(self):
        """T19: Duplikat retrieved ID tidak boleh menaikkan Precision@k."""
        score = compute_precision_at_k(["c1", "c1", "c2"], ["c1"], k=3)
        self.assertAlmostEqual(score, 1 / 3, places=5, msg=f"T19 FAILED: got {score}")

    def test_T20_non_positive_k(self):
        """T20: Precision@k = 0.0 untuk k <= 0."""
        self.assertEqual(compute_precision_at_k(["c1"], ["c1"], k=0), 0.0)
        self.assertEqual(compute_precision_at_k(["c1"], ["c1"], k=-1), 0.0)


# ── T13–T15: compute_recall_at_k ─────────────────────────────────────────────

class TestComputeRecallAtK(unittest.TestCase):

    def test_T13_all_found(self):
        """T13: Recall@5 = 1.0 semua relevan ditemukan."""
        score = compute_recall_at_k(["c1", "c2", "c3"], ["c1", "c2", "c3"], k=5)
        self.assertAlmostEqual(score, 1.0, places=5, msg=f"T13 FAILED: got {score}")

    def test_T14_none_found(self):
        """T14: Recall@5 = 0.0 tidak ada yang ditemukan."""
        score = compute_recall_at_k(["c1", "c2", "c3"], ["c4", "c5"], k=5)
        self.assertAlmostEqual(score, 0.0, places=5, msg=f"T14 FAILED: got {score}")

    def test_T15_empty_relevant(self):
        """T15: Recall = 0.0 jika relevant_ids kosong."""
        score = compute_recall_at_k(["c1", "c2"], [], k=5)
        self.assertAlmostEqual(score, 0.0, places=5, msg=f"T15 FAILED: got {score}")

    def test_T19_duplicate_retrieved_ids_count_once(self):
        """T19: Duplikat retrieved ID tidak boleh membuat Recall@k > 1.0."""
        score = compute_recall_at_k(["c1", "c1", "c1"], ["c1"], k=3)
        self.assertAlmostEqual(score, 1.0, places=5, msg=f"T19 FAILED: got {score}")

    def test_T20_non_positive_k(self):
        """T20: Recall@k = 0.0 untuk k <= 0."""
        self.assertEqual(compute_recall_at_k(["c1"], ["c1"], k=0), 0.0)
        self.assertEqual(compute_recall_at_k(["c1"], ["c1"], k=-1), 0.0)


# ── T16–T18: compute_mrr ──────────────────────────────────────────────────────

class TestComputeMRR(unittest.TestCase):

    def test_T16_relevant_at_rank1(self):
        """T16: MRR = 1.0 jika relevan ada di posisi 1."""
        score = compute_mrr(["c1", "c2", "c3"], ["c1"])
        self.assertAlmostEqual(score, 1.0, places=5, msg=f"T16 FAILED: got {score}")

    def test_T17_relevant_at_rank2(self):
        """T17: MRR = 0.5 jika relevan ada di posisi 2."""
        score = compute_mrr(["c0", "c1", "c2"], ["c1"])
        self.assertAlmostEqual(score, 0.5, places=5, msg=f"T17 FAILED: got {score}")

    def test_T18_none_relevant(self):
        """T18: MRR = 0.0 jika tidak ada yang relevan."""
        score = compute_mrr(["c1", "c2", "c3"], ["c99"])
        self.assertAlmostEqual(score, 0.0, places=5, msg=f"T18 FAILED: got {score}")


class TestComputeF1AtK(unittest.TestCase):

    def test_T21_harmonic_mean(self):
        """T21: F1@k = harmonic mean dari Precision@k dan Recall@k."""
        score = compute_f1_at_k(0.5, 1.0)
        self.assertAlmostEqual(score, 2 / 3, places=5, msg=f"T21 FAILED: got {score}")

    def test_T21_zero_denominator(self):
        """T21: F1@k = 0.0 ketika precision dan recall sama-sama 0."""
        score = compute_f1_at_k(0.0, 0.0)
        self.assertEqual(score, 0.0)


# ── T22: build_summary ────────────────────────────────────────────────────────

class TestBuildSummary(unittest.TestCase):

    def test_T22_aggregation(self):
        """T22: build_summary menghitung mean BLEU + ROUGE-L per method dengan benar."""
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_generation_eval import build_summary

        per_query = [
            {"method": "element_based",   "answer": "a", "bleu": 0.4, "rouge_l": 0.6},
            {"method": "element_based",   "answer": "b", "bleu": 0.6, "rouge_l": 0.8},
            {"method": "recursive",       "answer": "c", "bleu": 0.2, "rouge_l": 0.3},
            {"method": "recursive",       "answer": "d", "bleu": 0.8, "rouge_l": 0.7},
        ]
        summary = {s["method"]: s for s in build_summary(per_query)}

        self.assertAlmostEqual(summary["element_based"]["mean_bleu"],    0.5, places=5)
        self.assertAlmostEqual(summary["element_based"]["mean_rouge_l"], 0.7, places=5)
        self.assertAlmostEqual(summary["recursive"]["mean_bleu"],        0.5, places=5)
        self.assertAlmostEqual(summary["recursive"]["mean_rouge_l"],     0.5, places=5)
        self.assertEqual(summary["element_based"]["n_success"], 2)
        self.assertEqual(summary["recursive"]["n_success"],     2)


# ── T23: load_qa_gold ─────────────────────────────────────────────────────────

class TestLoadQaGold(unittest.TestCase):

    def test_T23_load_qa_gold(self):
        """T23: load_qa_gold membaca xlsx dan menghasilkan 30 item dengan field wajib."""
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_generation_eval import load_qa_gold, QA_GOLD_XLSX

        items = load_qa_gold(QA_GOLD_XLSX)

        self.assertEqual(len(items), 30,
                         msg=f"T23 FAILED: expected 30 QA items, got {len(items)}")

        required_fields = {"id", "question", "reference_answer", "relevant_chunk_ids"}
        for item in items:
            missing = required_fields - item.keys()
            self.assertFalse(missing,
                             msg=f"T23 FAILED: item {item.get('id')} missing fields: {missing}")
            self.assertTrue(item["id"],               msg="T23 FAILED: id kosong")
            self.assertTrue(item["question"],         msg="T23 FAILED: question kosong")
            self.assertTrue(item["reference_answer"], msg="T23 FAILED: reference_answer kosong")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()

    test_classes = [
        TestComputeBleu,
        TestComputeRouge,
        TestComputePrecisionAtK,
        TestComputeRecallAtK,
        TestComputeMRR,
        TestBuildSummary,
        TestLoadQaGold,
    ]
    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
