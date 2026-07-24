"""Lightweight tests for batch latency metrics; no RAG models are loaded."""

import csv
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_generation_eval as generation_eval


class FakeGenerator:
    def generate(self, question, contexts):
        assert question == "Pertanyaan uji"
        assert contexts == ["Konteks uji"]
        return "Jawaban uji"


class FakeRAGPipeline:
    def __init__(self, embedder, generator, chroma_client, chunking_method, top_k):
        self.chunking_method = chunking_method
        self.top_k = top_k

    def retrieve_by_vector(self, query_vector, k):
        assert query_vector == [0.1, 0.2]
        assert k == 1
        return [{"id": "chunk-1", "document": "Konteks uji"}]

    def retrieve(self, question, k):  # pragma: no cover - cached embedding is used.
        raise AssertionError("Query embedding should be precomputed")

    def _format_context(self, document):
        return document["document"]


def test_evaluate_top_k_records_split_latency_without_real_pipeline(tmp_path):
    fake_rag_module = ModuleType("src.rag.pipeline")
    fake_rag_module.RAGPipeline = FakeRAGPipeline
    pipeline = SimpleNamespace(
        embedder=object(),
        generator=FakeGenerator(),
        chroma_client=object(),
    )
    output_path = tmp_path / "eval.csv"

    with (
        patch.dict(sys.modules, {"src.rag.pipeline": fake_rag_module}),
        patch.object(generation_eval, "_synchronize_cuda"),
        patch.object(
            generation_eval.time,
            "perf_counter",
            side_effect=[10.0, 10.125, 10.2, 10.7, 10.75],
        ),
        patch.object(generation_eval, "compute_bleu", return_value=0.5),
        patch.object(generation_eval, "compute_rouge", return_value=0.6),
    ):
        rows = generation_eval.evaluate_top_k(
            pipeline=pipeline,
            qa_items=[{
                "query_id": "Q001",
                "question": "Pertanyaan uji",
                "gold_answer": "Jawaban acuan",
            }],
            gt_lookup={
                "Q001": {"relevant_chunk_ids": {"element_based": ["chunk-1"]}}
            },
            query_embeddings={"Q001": ([0.1, 0.2], True)},
            methods=["element_based"],
            current_k=1,
            existing_rows=[],
            done=set(),
            output_path=output_path,
            hardware_info="{}",
        )

    row = rows[0]
    assert row["top_k"] == 1
    assert row["retrieval_seconds"] == 0.125
    assert row["generation_seconds"] == 0.5
    assert row["total_response_seconds"] == 0.75
    assert row["f1_at_k"] == 1.0

    with output_path.open(newline="", encoding="utf-8") as file:
        saved = list(csv.DictReader(file))[0]
    assert saved["retrieval_seconds"] == "0.125"
    assert saved["generation_seconds"] == "0.5"
    assert saved["total_response_seconds"] == "0.75"


def test_build_summary_groups_latency_by_method_and_top_k():
    rows = [
        {
            "method": "Element-Based",
            "top_k": 5,
            "generated_answer": "a",
            "f1_at_k": 0.4,
            "retrieval_seconds": 0.1,
            "generation_seconds": 1.0,
            "total_response_seconds": 1.2,
        },
        {
            "method": "Element-Based",
            "top_k": "5",
            "generated_answer": "b",
            "f1_at_k": 0.6,
            "retrieval_seconds": "0.3",
            "generation_seconds": "2.0",
            "total_response_seconds": "2.4",
        },
    ]

    summary = generation_eval.build_summary(rows)

    assert len(summary) == 1
    assert summary[0]["top_k"] == 5
    assert summary[0]["n_timed"] == 2
    assert summary[0]["mean_f1_at_k"] == 0.5
    assert summary[0]["mean_retrieval_seconds"] == 0.2
    assert summary[0]["median_generation_seconds"] == 1.5
    assert summary[0]["mean_total_response_seconds"] == 1.8


def test_resume_reprocesses_legacy_rows_without_timing(tmp_path):
    path = tmp_path / "legacy.csv"
    path.write_text(
        "query_id,method,generated_answer,error,total_response_seconds\n"
        "Q001,Element-Based,Jawaban lama,,\n"
        "Q002,Recursive,Jawaban baru,,1.25\n",
        encoding="utf-8",
    )

    _, done = generation_eval.load_existing_done(path)

    assert ("Q001", "Element-Based") not in done
    assert ("Q002", "Recursive") in done
