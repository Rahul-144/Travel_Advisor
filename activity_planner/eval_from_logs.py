"""
eval_from_logs.py
=================
Full Ragas evaluation pipeline driven by qa_context_log.json.

Steps:
  1. Load qa_context_log.json
  2. Filter entries that have real retrieved context (non-empty strings)
  3. For each filtered entry, use a powerful LLM to generate synthetic ground truth
     grounded strictly in the retrieved context
  4. Build a Ragas EvaluationDataset and run metrics
  5. Save a JSON report + print a summary

Usage:
    python eval_from_logs.py
    python eval_from_logs.py --log /path/to/qa_context_log.json --output report.json
"""

import json
import os
import re
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from ragas_eval import RagasEvaluator, save_evaluation_report

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOG_FILE = Path(__file__).parent.parent / "qa_context_log.json"
PREPARED_DATASET_FILE = Path(__file__).parent.parent / "prepared_eval_dataset.json"
DEFAULT_REPORT_FILE = Path(__file__).parent.parent / "ragas_eval_report.json"

# Ground-truth generation prompt (strict: only from context)
GT_PROMPT = """\
You are a travel expert evaluator. Given a travel question and retrieved context, write a \
concise ideal ground-truth answer (2-4 sentences) that:
- Is grounded ONLY in the provided context — do NOT add outside knowledge
- Directly and accurately answers the travel question
- Is written in plain prose (no lists, no JSON, no markdown)

Question: {question}

Retrieved Context:
{context}

Ground-truth answer:"""

# LLM is created lazily on first use so importing this module never fails
_gt_llm = None

def _get_gt_llm() -> ChatOpenAI:
    """Return (or create) the ground-truth generation LLM."""
    global _gt_llm
    if _gt_llm is None:
        _gt_llm = ChatOpenAI(
            openai_api_base="https://api.groq.com/openai/v1",
            openai_api_key=os.environ.get("OPEN_API_KEY"),
            model_name="qwen/qwen3-32b",
        )
    return _gt_llm


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks produced by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_answer_text(answer: Any) -> str:
    """
    Convert the `answer` field (which may be a dict or string) to plain text
    suitable for Ragas.
    """
    if isinstance(answer, str):
        # Try to decode JSON-encoded answers stored as strings
        try:
            return extract_answer_text(json.loads(answer))
        except (json.JSONDecodeError, ValueError):
            # Strip markdown code fences if present
            cleaned = re.sub(r"```(?:json)?", "", answer).strip("`").strip()
            try:
                return extract_answer_text(json.loads(cleaned))
            except Exception:
                return answer

    if isinstance(answer, dict):
        # non_trip response: just use the message
        if "message" in answer:
            return answer["message"]
        # raw content field
        if "content" in answer:
            content = answer["content"]
            return extract_answer_text(content) if isinstance(content, str) else str(content)
        # trip_plan: build a readable prose summary
        parts: List[str] = []
        if dest := answer.get("destination"):
            parts.append(f"Destination: {dest}.")
        if acts := answer.get("activities"):
            parts.append("Key activities include " + ", ".join(acts[:4]) + ".")
        if trans := answer.get("transportation"):
            parts.append("Transportation options: " + "; ".join(trans[:3]) + ".")
        if itin := answer.get("itinerary"):
            parts.append("Suggested itinerary: " + " ".join(itin[:3]))
        return " ".join(parts) if parts else json.dumps(answer)

    return str(answer)


def has_real_context(entry: Dict) -> bool:
    """Return True iff at least one non-empty context string is present."""
    for item in entry.get("retrieved_context", []):
        if isinstance(item, dict) and item.get("context", "").strip():
            return True
    return False


def get_contexts(entry: Dict) -> List[str]:
    """Return a list of non-empty context strings."""
    return [
        item["context"]
        for item in entry.get("retrieved_context", [])
        if isinstance(item, dict) and item.get("context", "").strip()
    ]


def generate_ground_truth(question: str, contexts: List[str]) -> str:
    """Call the powerful LLM to produce a context-grounded ground-truth answer."""
    context_block = "\n---\n".join(contexts)
    prompt = GT_PROMPT.format(question=question, context=context_block)
    try:
        response = _get_gt_llm().invoke([HumanMessage(content=prompt)])
        return strip_thinking_tags(response.content)
    except Exception as exc:
        print(f"\n  ⚠️  GT generation failed: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def load_and_prepare(log_file: Optional[str] = None) -> List[Dict]:
    """
    Load qa_context_log.json, keep only entries with real retrieved context,
    and generate synthetic ground truth for each.

    Returns a list of dicts ready for RagasEvaluator:
        {question, answer, contexts, ground_truth}
    """
    path = log_file or str(LOG_FILE)
    with open(path, "r") as f:
        logs: List[Dict] = json.load(f)

    total = len(logs)
    print(f"📂  Loaded {total} entries from {Path(path).name}")

    # Filter: must have real context AND be a meaningful travel Q (has question text)
    with_context = [e for e in logs if e.get("question", "").strip() and has_real_context(e)]
    without_context = total - len(with_context)

    print(f"✅  {len(with_context)} entries have retrieved context  →  will be evaluated")
    print(f"⏭️   {without_context} entries skipped  (no context / greeting-only)\n")

    print("🤖  Generating synthetic ground truth with LLM...")
    test_cases: List[Dict] = []

    for i, entry in enumerate(with_context, 1):
        question = entry["question"].strip()
        contexts = get_contexts(entry)
        answer_text = extract_answer_text(entry.get("answer", ""))

        print(f"  [{i:>2}/{len(with_context)}] {question[:60]}…", end=" ", flush=True)
        ground_truth = generate_ground_truth(question, contexts)
        print("✓")

        test_cases.append(
            {
                "question": question,
                "answer": answer_text,
                "contexts": contexts,
                "ground_truth": ground_truth,
            }
        )

    return test_cases


def run_eval(
    log_file: Optional[str] = None,
    output: Optional[str] = None,
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Full pipeline:
      load logs → filter → generate GT → evaluate → save report → print summary
    """
    print("=" * 60)
    print("🚀  Ragas Evaluation  ←  qa_context_log.json")
    print("=" * 60 + "\n")

    # ── 1. Prepare dataset ──────────────────────────────────────────────────
    test_cases = load_and_prepare(log_file)

    if not test_cases:
        print("❌  No valid test cases found — all entries lack retrieved context.")
        return {}

    # Save the prepared dataset for manual inspection / re-runs
    with open(PREPARED_DATASET_FILE, "w") as f:
        json.dump(test_cases, f, indent=2, ensure_ascii=False)
    print(f"\n💾  Prepared dataset  →  {PREPARED_DATASET_FILE.name}  ({len(test_cases)} records)")

    # ── 2. Run Ragas ────────────────────────────────────────────────────────
    print("\n📊  Running Ragas evaluation metrics…")
    evaluator = RagasEvaluator()
    results = evaluator.evaluate(test_cases, metrics=metrics)

    # ── 3. Print summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📈  RAGAS EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Test cases evaluated : {len(test_cases)}")
    agg = results.get("aggregate_score", 0.0)
    print(f"  Aggregate score      : {agg:.4f}  {'🟢' if agg >= 0.7 else ('🟡' if agg >= 0.5 else '🔴')}\n")

    for metric, score in results.items():
        if metric in ("aggregate_score", "per_question"):
            continue
        icon = "✓" if score >= 0.7 else ("⚠" if score >= 0.5 else "✗")
        bar_len = int(score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {icon} {metric:<22} {score:.4f}  [{bar}]")

    # Per-question breakdown (top 5 worst)
    pq = results.get("per_question", [])
    if pq:
        print(f"\n  📋  Per-question scores (worst 5, sorted by faithfulness):")
        sorted_pq = sorted(pq, key=lambda x: x.get("faithfulness", 1.0))[:5]
        for row in sorted_pq:
            q = row["question"][:55]
            f_score = row.get("faithfulness", "—")
            print(f"    ▸ {q:<57} faithfulness={f_score if isinstance(f_score, str) else f'{f_score:.3f}'}")

    # ── 4. Save report ──────────────────────────────────────────────────────
    out_path = output or str(DEFAULT_REPORT_FILE)
    full_report = {
        "test_cases_count": len(test_cases),
        "metrics": {k: v for k, v in results.items() if k != "per_question"},
        "per_question": results.get("per_question", []),
    }
    save_evaluation_report(full_report, out_path)
    print(f"\n✅  Report saved →  {Path(out_path).name}")
    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ragas evaluation from qa_context_log.json"
    )
    parser.add_argument(
        "--log",
        default=None,
        help=f"Path to QA log file (default: {LOG_FILE})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Output report path (default: {DEFAULT_REPORT_FILE})",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help=(
            "Metrics to run: faithfulness answer_relevancy context_precision "
            "context_recall context_relevance  (default: all except context_recall)"
        ),
    )
    args = parser.parse_args()
    run_eval(log_file=args.log, output=args.output, metrics=args.metrics)
