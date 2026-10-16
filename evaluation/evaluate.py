r"""Evaluate the existing RAG pipeline against evaluation/questions.json.

Run from the project root:
    .\.venv\Scripts\python.exe evaluation/evaluate.py --only q16 q17 q18 q19
"""

import json
import sys
import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from google import genai
from ask import api_key, process_question, configure_gemini

EVALUATION_DIR = PROJECT_DIR / "evaluation"
QUESTIONS_PATH = EVALUATION_DIR / "questions.json"
RESULTS_PATH = EVALUATION_DIR / "results.json"


def save_results(records, summary, run_path):
    payload = json.dumps(
        {"summary": summary, "results": records}, ensure_ascii=False, indent=2
    )
    # Latest run + dated copy: narrow test runs no longer destroy prior snapshots.
    RESULTS_PATH.write_text(payload, encoding="utf-8")
    run_path.write_text(payload, encoding="utf-8")


def save_review_sheet(records, review_path):
    """Human review of factual support; a matching [S1] label alone is not proof."""
    lines = [
        "# Step 47J — Manual answer and citation review",
        "",
        "Review only completed, answered questions. Mark factual correctness and",
        "whether each cited excerpt actually supports the claim.",
        "",
    ]
    for r in records:
        lines.extend(
            [
                f"## {r['id']}: {r['question']}",
                "",
                f"Status: `{r.get('status')}` | Query mode: `{r.get('query_mode', 'unknown')}`",
                "",
                f"Expected answer: {r.get('expected_answer') or '(refusal / not supplied)'}",
                "",
                f"Actual answer: {r.get('actual_answer') or '(no answer)' }",
                "",
                f"Citation labels valid: {r.get('citation_labels_valid')}",
                "",
            ]
        )
        for src in r.get("sources", []):
            excerpt = " ".join(src.get("text", "").split())[:450]
            lines.append(
                f"- [{src['label']}] {src['source_file']} p.{src['page']}: {excerpt}"
            )
        lines.extend(
            [
                "",
                "- [ ] Factual answer correct (or correct refusal)",
                "- [ ] Cited excerpts directly support the answer (if answered)",
                "- Reviewer notes: ",
                "",
                "---",
                "",
            ]
        )
    review_path.write_text("\n".join(lines), encoding="utf-8")


def check_answer_language(question, answer):
    """Simple script check, not a full language-quality evaluator."""
    if not answer or not answer.strip():
        return None
    question_is_kannada = any("\u0c80" <= c <= "\u0cff" for c in question)
    answer_kannada_count = sum("\u0c80" <= c <= "\u0cff" for c in answer)
    if question_is_kannada:
        return answer_kannada_count > 0
    return answer_kannada_count == 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate Karnataka admissions RAG")
    parser.add_argument(
        "--only", nargs="+", help="Evaluate specific IDs, e.g. --only q16 q17"
    )
    parser.add_argument(
        "--min-api-interval",
        type=float,
        default=13.0,
        help="Minimum seconds between Gemini requests (default: 13).",
    )
    args = parser.parse_args()
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError("questions.json must contain a JSON list")

    if args.only:
        questions = [q for q in questions if q["id"] in set(args.only)]
        if not questions:
            raise ValueError("No matching question IDs found")
    configure_gemini(args.min_api_interval)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_path = EVALUATION_DIR / f"results_{run_stamp}.json"
    review_path = EVALUATION_DIR / f"review_{run_stamp}.md"
    records = []
    with genai.Client(api_key=api_key) as client:
        for index, item in enumerate(questions, 1):
            print(
                f"\n{'=' * 55}\n[{index}/{len(questions)}] {item['id']}: {item['question']}"
            )
            expected_ids = item.get("expected_chunk_ids", [])
            expected_answerable = bool(item["answerable"])
            try:
                output = process_question(item["question"], client)
                status = output.get("status", "unknown")
                retrieved = output.get("retrieved_chunk_ids", [])
                predicted_answerable = bool(output.get("answerable", False))
                # API failures are not refusals; exclude them from answerability scoring.
                valid_prediction = status not in ("api_error", "error")
                # If a failure happened before retrieval, retrieval is unscored.
                # If retrieval completed before a later API failure, still score it.
                retrieval_completed = output.get("retrieval_completed", bool(retrieved))
                hit1 = (
                    (bool(retrieved) and retrieved[0] in expected_ids)
                    if expected_ids and retrieval_completed
                    else None
                )
                hit3 = (
                    (any(cid in expected_ids for cid in retrieved[:3]))
                    if expected_ids and retrieval_completed
                    else None
                )
                answerability_correct = (
                    predicted_answerable == expected_answerable
                    if valid_prediction
                    else None
                )
                language_correct = (
                    check_answer_language(item["question"], output.get("answer", ""))
                    if status == "answered"
                    else None
                )

                # Supports individual and combined citations:
                # [S1], [S2, S3], [S1, S2, S3]
                citation_groups = re.findall(
                    r"\[(S\d+(?:\s*,\s*S\d+)*)\]", output.get("answer", "")
                )
                cited_labels = {
                    label.strip()
                    for group in citation_groups
                    for label in group.split(",")
                }
                known_labels = {source["label"] for source in output.get("sources", [])}
                citation_labels_valid = (
                    bool(cited_labels) and cited_labels <= known_labels
                    if status == "answered"
                    else None
                )

                record = {
                    "query_mode": output.get("query_mode"),
                    "retrieval_completed": retrieval_completed,
                    "sources": output.get("sources", []),
                    "citation_labels_valid": citation_labels_valid,
                    "language_correct": language_correct,
                    "error": output.get("error"),
                    "id": item["id"],
                    "question": item["question"],
                    "test_category": item.get("test_category"),
                    "expected_answerable": expected_answerable,
                    "predicted_answerable": (
                        predicted_answerable if valid_prediction else None
                    ),
                    "answerability_correct": answerability_correct,
                    "expected_chunk_ids": expected_ids,
                    "retrieved_chunk_ids": retrieved,
                    "hit_at_1": hit1,
                    "hit_at_3": hit3,
                    "expected_answer": item.get("expected_answer"),
                    "actual_answer": output.get("answer", ""),
                    "top_reranker_score": output.get("top_reranker_score"),
                    "status": status,
                }
                print(
                    f"Evaluation: answerability={answerability_correct}, Hit@1={record['hit_at_1']}, Hit@3={record['hit_at_3']}"
                )
            except Exception as exc:
                record = {
                    "id": item.get("id"),
                    "question": item.get("question"),
                    "status": "error",
                    "error": str(exc),
                    "answerability_correct": None,
                    "hit_at_1": None,
                    "hit_at_3": None,
                }
                print(f"Evaluation error: {exc}")
            records.append(record)
            # Save after each question so progress survives interruptions.
            save_results(
                records,
                {
                    "completed": len(records),
                    "total": len(questions),
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                },
                run_path,
            )

    evaluated = [r for r in records if r.get("answerability_correct") is not None]
    retrieval = [r for r in records if r.get("hit_at_1") is not None]
    correct = sum(r["answerability_correct"] for r in evaluated)
    hit1 = sum(r["hit_at_1"] for r in retrieval)
    hit3 = sum(r["hit_at_3"] for r in retrieval)
    false_accepts = sum(
        r["expected_answerable"] is False and r["predicted_answerable"] is True
        for r in evaluated
    )
    false_rejects = sum(
        r["expected_answerable"] is True and r["predicted_answerable"] is False
        for r in evaluated
    )
    language_records = [r for r in records if r.get("language_correct") is not None]
    language_correct = sum(r["language_correct"] for r in language_records)
    api_errors = sum(r.get("status") == "api_error" for r in records)
    answered_records = [r for r in records if r.get("status") == "answered"]
    cited_pass = sum(r.get("citation_labels_valid") is True for r in answered_records)
    summary = {
        "api_errors": api_errors,
        "original_only_queries": sum(
            r.get("query_mode") == "original_only" for r in records
        ),
        "citation_labels_valid": cited_pass,
        "citation_labels_evaluated": len(answered_records),
        "language_correct": language_correct,
        "language_evaluated": len(language_records),
        "language_accuracy": (
            language_correct / len(language_records) if language_records else None
        ),
        "completed": len(records),
        "total": len(questions),
        "answerability_accuracy": correct / len(evaluated) if evaluated else None,
        "answerability_evaluated": len(evaluated),
        "hit_at_1": hit1 / len(retrieval) if retrieval else None,
        "hit_at_3": hit3 / len(retrieval) if retrieval else None,
        "retrieval_evaluated": len(retrieval),
        "false_acceptances": false_accepts,
        "false_rejections": false_rejects,
        "errors_or_unscored": len(records) - len(evaluated),
        "note": "Expected answer text and citation correctness require separate review; this script does not automatically grade them.",
    }
    save_results(records, summary, run_path)
    save_review_sheet(records, review_path)
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Answerability: {correct}/{len(evaluated)} correct")
    print(f"Hit@1: {hit1}/{len(retrieval)}")
    print(f"Hit@3: {hit3}/{len(retrieval)}")
    print(f"False acceptances: {false_accepts}; false rejections: {false_rejects}")
    print(f"API errors (unscored): {api_errors}")
    print(
        f"Answer language: {language_correct}/{len(language_records)} matched question language"
    )
    print(
        f"Citation labels valid: {cited_pass}/{len(answered_records)} (label check only)"
    )
    print(f"Original-only fallback: {summary['original_only_queries']} question(s)")
    print(f"Latest results: {RESULTS_PATH}")
    print(f"Saved run snapshot: {run_path}")
    print(f"Manual review checklist: {review_path}")


if __name__ == "__main__":
    main()
