"""
Ragas evaluation pipeline for Travel Advisor RAG system. (ragas 0.4.x)
Measures retrieval quality, generation quality, and overall system performance.
"""

import json
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# --- ragas 0.4.x imports ---
from ragas import evaluate
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextRelevance,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ---------------------------------------------------------------------------
# LLM + embeddings shared by the evaluator
# ---------------------------------------------------------------------------
# Module-level placeholders; actual objects are created lazily inside RagasEvaluator.__init__
_llm = None
_embeddings = None

# Available metric CLASSES — instantiated fresh in evaluate() so llm/embeddings inject cleanly
METRIC_REGISTRY: Dict[str, Any] = {
    "faithfulness":     Faithfulness,
    "answer_relevancy": AnswerRelevancy,
    "context_precision": ContextPrecision,
    "context_recall":   ContextRecall,
    "context_relevance": ContextRelevance,
}

DEFAULT_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_relevance",
]


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------
class RagasEvaluator:
    """Evaluate Travel Advisor RAG system using Ragas 0.4.x metrics."""

    def __init__(self):
        llm = ChatOpenAI(
            openai_api_base="https://api.groq.com/openai/v1",
            openai_api_key=os.environ.get("OPEN_API_KEY"),
            model_name="qwen/qwen3-32b",
        )
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.ragas_llm = LangchainLLMWrapper(llm)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    def prepare_dataset(self, test_cases: List[Dict[str, Any]]) -> EvaluationDataset:
        """
        Convert test cases to a ragas 0.4.x EvaluationDataset.

        Each test case must have:
            question     : str
            answer       : str
            contexts     : List[str]
            ground_truth : str  (may be empty; used only by reference metrics)
        """
        samples = []
        for case in test_cases:
            samples.append(
                SingleTurnSample(
                    user_input=case.get("question", ""),
                    response=case.get("answer", ""),
                    retrieved_contexts=case.get("contexts", []),
                    reference=case.get("ground_truth", ""),
                )
            )
        return EvaluationDataset(samples=samples)

    def evaluate(
        self,
        test_cases: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run Ragas evaluation on test cases.

        Args:
            test_cases: list of dicts with keys question/answer/contexts/ground_truth
            metrics:    names from METRIC_REGISTRY (None → DEFAULT_METRICS)

        Returns:
            dict with per-metric scores, aggregate_score, and per_question breakdown
        """
        selected_names = metrics if metrics else DEFAULT_METRICS
        selected_metric_classes = [
            METRIC_REGISTRY[m] for m in selected_names if m in METRIC_REGISTRY
        ]

        if not selected_metric_classes:
            raise ValueError(
                f"No valid metrics selected. Available: {list(METRIC_REGISTRY.keys())}"
            )

        # Instantiate each metric and inject the configured llm / embeddings
        selected_metrics = []
        for MetricClass in selected_metric_classes:
            m = MetricClass()
            if hasattr(m, "llm"):
                m.llm = self.ragas_llm
            if hasattr(m, "embeddings"):
                m.embeddings = self.ragas_embeddings
            selected_metrics.append(m)

        dataset = self.prepare_dataset(test_cases)

        results = evaluate(
            dataset=dataset,
            metrics=selected_metrics,
            llm=self.ragas_llm,
            embeddings=self.ragas_embeddings,
        )

        # ragas 0.4 returns a Results object; convert to plain dict
        scores = results.to_pandas()

        metric_cols = [
            c for c in scores.columns
            if c not in ("user_input", "response", "retrieved_contexts", "reference")
        ]

        results_dict: Dict[str, Any] = {}
        for col in metric_cols:
            try:
                results_dict[col] = float(scores[col].mean())
            except Exception:
                pass

        results_dict["aggregate_score"] = (
            sum(results_dict.values()) / len(results_dict)
            if results_dict else 0.0
        )

        # Per-question breakdown
        per_question = []
        for i, case in enumerate(test_cases):
            row: Dict[str, Any] = {"question": case.get("question", "")}
            for col in metric_cols:
                try:
                    row[col] = float(scores[col].iloc[i])
                except Exception:
                    pass
            per_question.append(row)

        results_dict["per_question"] = per_question
        return results_dict

    def _extract_per_question_scores(self, results, test_cases):
        """Kept for backward compatibility."""
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_test_dataset(filepath: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load test cases from a JSON file (falls back to built-in defaults)."""
    if filepath and os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return get_default_test_cases()


def get_default_test_cases() -> List[Dict[str, Any]]:
    """Built-in travel test cases for smoke-testing."""
    return [
        {
            "question": "What are the best activities in Paris?",
            "answer": (
                "Paris offers iconic attractions like the Eiffel Tower, Louvre Museum, "
                "Notre-Dame Cathedral, and Seine river cruises."
            ),
            "contexts": [
                "Paris attractions include the Eiffel Tower, Arc de Triomphe, and Louvre Museum.",
                "Popular activities: visiting museums, walking along the Seine, exploring Montmartre.",
            ],
            "ground_truth": "Paris features museums, historical monuments, river cruises, and cultural experiences.",
        },
        {
            "question": "How to get around Tokyo?",
            "answer": (
                "Tokyo has excellent public transportation. The subway system is extensive "
                "and efficient, with IC cards like Suica working across most lines."
            ),
            "contexts": [
                "Tokyo subway lines run throughout the city, trains connect neighboring areas.",
                "The Yamanote Line is a circular train that connects major districts.",
            ],
            "ground_truth": "Use subway, trains, buses, or taxis; IC cards are convenient for multiple trips.",
        },
    ]


def save_evaluation_report(
    results: Dict[str, Any],
    output_path: str = "evaluation_report.json",
) -> str:
    """Persist evaluation results as JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    return output_path


def run_evaluation(test_dataset: Optional[str] = None) -> Dict[str, Any]:
    """Convenience wrapper: load → evaluate → print → return."""
    print("🚀 Starting Ragas Evaluation Pipeline...")

    test_cases = load_test_dataset(test_dataset)
    print(f"✓ Loaded {len(test_cases)} test cases")

    evaluator = RagasEvaluator()
    print("✓ Initialized Ragas evaluator")

    print("📊 Running evaluation metrics...")
    results = evaluator.evaluate(test_cases)

    print("\n" + "=" * 50)
    print("📈 EVALUATION RESULTS")
    print("=" * 50)
    print(f"Aggregate Score: {results['aggregate_score']:.4f}")
    print("\nMetric Scores:")
    for metric, score in results.items():
        if metric not in ("aggregate_score", "per_question"):
            print(f"  {metric}: {score:.4f}")

    print("\n✓ Evaluation complete!")
    return results


if __name__ == "__main__":
    results = run_evaluation()
    output_file = save_evaluation_report(results)
    print(f"\n✓ Results saved to {output_file}")
