"""
Ragas evaluation pipeline for Travel Advisor RAG system.
Measures retrieval quality, generation quality, and overall system performance.
"""

import json
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_precision,
    context_recall,
)
from ragas.llm import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset

load_dotenv()

# Initialize LLM and embeddings for Ragas
llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.environ.get("OPEN_API_KEY"),
    model_name="qwen/qwen3-32b"
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


class RagasEvaluator:
    """Evaluate Travel Advisor RAG system using Ragas metrics."""
    
    def __init__(self):
        self.ragas_llm = LangchainLLMWrapper(llm)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
        
    def prepare_dataset(self, test_cases: List[Dict[str, Any]]) -> Dataset:
        """
        Convert test cases to Ragas dataset format.
        
        Expected format for each test case:
        {
            "question": "user query",
            "answer": "model response",
            "contexts": ["retrieval 1", "retrieval 2", ...],
            "ground_truth": "expected answer (optional)"
        }
        """
        # Prepare data for Ragas
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        
        for case in test_cases:
            data["question"].append(case.get("question", ""))
            data["answer"].append(case.get("answer", ""))
            data["contexts"].append(case.get("contexts", []))
            data["ground_truth"].append(case.get("ground_truth", ""))
            
        return Dataset.from_dict(data)
    
    def evaluate(
        self, 
        test_cases: List[Dict[str, Any]],
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run Ragas evaluation on test cases.
        
        Args:
            test_cases: List of test case dictionaries
            metrics: Which metrics to calculate (None = all available)
        
        Returns:
            Dictionary with evaluation results
        """
        if metrics is None:
            metrics = [
                "faithfulness",
                "answer_relevancy",
                "context_relevancy",
                "context_precision",
                "context_recall"
            ]
        
        dataset = self.prepare_dataset(test_cases)
        
        # Select metrics to evaluate
        metrics_dict = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_relevancy": context_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
        
        selected_metrics = [
            metrics_dict[m] for m in metrics 
            if m in metrics_dict
        ]
        
        if not selected_metrics:
            raise ValueError(f"Invalid metrics. Available: {list(metrics_dict.keys())}")
        
        # Run evaluation
        results = evaluate(
            dataset=dataset,
            metrics=selected_metrics,
            llm=self.ragas_llm,
            embeddings=self.ragas_embeddings
        )
        
        # Convert to dict for easier handling
        results_dict = {
            metric: float(results[metric]) 
            for metric in metrics 
            if metric in results
        }
        results_dict["aggregate_score"] = float(
            sum(results_dict.values()) / len(results_dict)
        ) if results_dict else 0.0
        
        # Add detailed results per question
        results_dict["per_question"] = self._extract_per_question_scores(
            results, test_cases
        )
        
        return results_dict
    
    def _extract_per_question_scores(
        self, 
        results: Dataset, 
        test_cases: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Extract per-question evaluation scores."""
        per_question = []
        
        for i, case in enumerate(test_cases):
            q_scores = {"question": case.get("question", "")}
            
            # Extract scores for this question from results
            for col in results.column_names:
                if col not in ["question", "answer", "contexts", "ground_truth"]:
                    try:
                        q_scores[col] = float(results[col][i])
                    except (IndexError, TypeError):
                        pass
            
            per_question.append(q_scores)
        
        return per_question


def load_test_dataset(filepath: str = None) -> List[Dict[str, Any]]:
    """Load test dataset from JSON file."""
    if filepath is None:
        filepath = "activity_planner/test_cases.json"
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        # Return default test cases
        return get_default_test_cases()


def get_default_test_cases() -> List[Dict[str, Any]]:
    """Return default test cases for Travel Advisor."""
    return [
        {
            "question": "What are the best activities in Paris?",
            "answer": "Paris offers iconic attractions like the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and Seine river cruises. Visitors can enjoy world-class museums, art galleries, cafes, and historic landmarks.",
            "contexts": [
                "Paris attractions include the Eiffel Tower, Arc de Triomphe, and Louvre Museum. The city is known for its art, culture, and cuisine.",
                "Popular activities: visiting museums, walking along the Seine, exploring neighborhoods like Marais and Montmartre, enjoying French cuisine."
            ],
            "ground_truth": "Paris features museums, historical monuments, river cruises, and cultural experiences."
        },
        {
            "question": "How to get around Tokyo?",
            "answer": "Tokyo has excellent public transportation. The subway system is extensive and efficient, with color-coded lines connecting all major areas. Buses, trains, and taxis are also available. Getting around is easy for visitors.",
            "contexts": [
                "Tokyo's transportation: subway lines run throughout the city, trains connect to neighboring areas, buses serve local routes, and taxis are available but expensive.",
                "The Yamanote Line is a circular train that connects major districts. IC cards like Suica can be used on most transport."
            ],
            "ground_truth": "Use subway, trains, buses, or taxis; IC cards are convenient for multiple trips."
        },
        {
            "question": "What's the best time to visit Rome?",
            "answer": "Spring (April-May) and fall (September-October) are ideal for visiting Rome. Weather is pleasant, crowds are manageable, and temperatures are comfortable for sightseeing.",
            "contexts": [
                "Rome weather: spring and fall offer mild temperatures (15-25°C), while summer is hot (30°C+) and crowded. Winter is cool (5-10°C) but less crowded.",
                "Best visiting periods: April-May and September-October avoid peak summer tourism and provide good weather."
            ],
            "ground_truth": "Spring and fall offer the best combination of pleasant weather and fewer tourists."
        }
    ]


def save_evaluation_report(
    results: Dict[str, Any], 
    output_path: str = "evaluation_report.json"
) -> str:
    """Save evaluation results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    return output_path


def run_evaluation(test_dataset: str = None) -> Dict[str, Any]:
    """
    Run full Ragas evaluation pipeline.
    
    Args:
        test_dataset: Path to test dataset JSON file (optional)
    
    Returns:
        Complete evaluation results
    """
    print("🚀 Starting Ragas Evaluation Pipeline...")
    
    # Load test cases
    test_cases = load_test_dataset(test_dataset)
    print(f"✓ Loaded {len(test_cases)} test cases")
    
    # Initialize evaluator
    evaluator = RagasEvaluator()
    print("✓ Initialized Ragas evaluator")
    
    # Run evaluation
    print("📊 Running evaluation metrics...")
    results = evaluator.evaluate(test_cases)
    
    # Report results
    print("\n" + "="*50)
    print("📈 EVALUATION RESULTS")
    print("="*50)
    print(f"Aggregate Score: {results['aggregate_score']:.4f}")
    print(f"\nMetric Scores:")
    for metric, score in results.items():
        if metric not in ["aggregate_score", "per_question"]:
            print(f"  {metric}: {score:.4f}")
    
    print("\n✓ Evaluation complete!")
    
    return results


if __name__ == "__main__":
    # Run evaluation with default test cases
    results = run_evaluation()
    
    # Save results
    output_file = save_evaluation_report(results)
    print(f"\n✓ Results saved to {output_file}")
