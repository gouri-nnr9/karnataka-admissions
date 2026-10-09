"""Evaluate the existing RAG pipeline against evaluation/questions.json.

Run from the project root:
    .\.venv\Scripts\python.exe evaluation/evaluate.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from google import genai
from ask import api_key, process_question

EVALUATION_DIR = PROJECT_DIR / "evaluation"
QUESTIONS_PATH = EVALUATION_DIR / "questions.json"
RESULTS_PATH = EVALUATION_DIR / "results.json"


def save_results(records, summary):
    RESULTS_PATH.write_text(
        json.dumps({"summary": summary, "results": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError("questions.json must contain a JSON list")

    records = []
    with genai.Client(api_key=api_key) as client:
        for index, item in enumerate(questions, 1):
            print(f"\n{'=' * 55}\n[{index}/{len(questions)}] {item['id']}: {item['question']}")
            expected_ids = item.get("expected_chunk_ids", [])
            expected_answerable = bool(item["answerable"])
            try:
                output = process_question(item["question"], client)
                status = output.get("status", "unknown")
                retrieved = output.get("retrieved_chunk_ids", [])
                predicted_answerable = bool(output.get("answerable", False))
                # API failures are not refusals; exclude them from answerability scoring.
                valid_prediction = status not in ("api_error", "error")
                hit1 = bool(expected_ids) and bool(retrieved) and retrieved[0] in expected_ids
                hit3 = bool(expected_ids) and any(cid in expected_ids for cid in retrieved[:3])
                answerability_correct = (
                    predicted_answerable == expected_answerable if valid_prediction else None
                )
                record = {
                    "id": item["id"],
                    "question": item["question"],
                    "test_category": item.get("test_category"),
                    "expected_answerable": expected_answerable,
                    "predicted_answerable": predicted_answerable if valid_prediction else None,
                    "answerability_correct": answerability_correct,
                    "expected_chunk_ids": expected_ids,
                    "retrieved_chunk_ids": retrieved,
                    "hit_at_1": hit1 if expected_ids else None,
                    "hit_at_3": hit3 if expected_ids else None,
                    "expected_answer": item.get("expected_answer"),
                    "actual_answer": output.get("answer", ""),
                    "top_reranker_score": output.get("top_reranker_score"),
                    "status": status,
                }
                print(f"Evaluation: answerability={answerability_correct}, Hit@1={record['hit_at_1']}, Hit@3={record['hit_at_3']}")
            except Exception as exc:
                record = {"id": item.get("id"), "question": item.get("question"), "status": "error", "error": str(exc), "answerability_correct": None, "hit_at_1": None, "hit_at_3": None}
                print(f"Evaluation error: {exc}")
            records.append(record)
            # Save after each question so progress survives interruptions.
            save_results(records, {"completed": len(records), "total": len(questions), "updated_utc": datetime.now(timezone.utc).isoformat()})

    evaluated = [r for r in records if r.get("answerability_correct") is not None]
    retrieval = [r for r in records if r.get("hit_at_1") is not None]
    correct = sum(r["answerability_correct"] for r in evaluated)
    hit1 = sum(r["hit_at_1"] for r in retrieval)
    hit3 = sum(r["hit_at_3"] for r in retrieval)
    false_accepts = sum(r["expected_answerable"] is False and r["predicted_answerable"] is True for r in evaluated)
    false_rejects = sum(r["expected_answerable"] is True and r["predicted_answerable"] is False for r in evaluated)
    summary = {
        "completed": len(records), "total": len(questions),
        "answerability_accuracy": correct / len(evaluated) if evaluated else None,
        "answerability_evaluated": len(evaluated),
        "hit_at_1": hit1 / len(retrieval) if retrieval else None,
        "hit_at_3": hit3 / len(retrieval) if retrieval else None,
        "retrieval_evaluated": len(retrieval),
        "false_acceptances": false_accepts, "false_rejections": false_rejects,
        "errors_or_unscored": len(records) - len(evaluated),
        "note": "Expected answer text and citation correctness require separate review; this script does not automatically grade them.",
    }
    save_results(records, summary)
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Answerability: {correct}/{len(evaluated)} correct")
    print(f"Hit@1: {hit1}/{len(retrieval)}")
    print(f"Hit@3: {hit3}/{len(retrieval)}")
    print(f"False acceptances: {false_accepts}; false rejections: {false_rejects}")
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
