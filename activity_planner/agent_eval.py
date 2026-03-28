"""
Integration module to evaluate the Travel Advisor agent with Ragas.
Runs the agent on test cases and measures retrieval + generation quality.
"""

import json
import sys
from typing import List, Dict, Any
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from Agents import get_agent
from ragas_eval import RagasEvaluator, save_evaluation_report, load_test_dataset
from eval_from_logs import run_eval as eval_from_logs, LOG_FILE


class TravelAdvisorEvaluator:
    """Evaluate Travel Advisor agent using Ragas metrics."""
    
    def __init__(self):
        self.agent = get_agent()
        self.ragas_evaluator = RagasEvaluator()
        
    def run_agent_on_query(self, query: str) -> Dict[str, Any]:
        """
        Run agent on a query and extract answer + contexts.
        
        Returns:
            {
                "question": original query,
                "answer": agent's response,
                "contexts": list of retrieved documents,
                "raw_result": full agent output
            }
        """
        try:
            result = self.agent.run(query, thread={"configurable": {"thread_id": "eval"}})
            
            # Extract answer from last message
            answer = result["messages"][-1].content
            
            # Extract citation/contexts (if available)
            citations = result.get("citation", "[]")
            try:
                citation_list = json.loads(citations) if isinstance(citations, str) else citations
            except:
                citation_list = []
            
            # For contexts, we'll use citation info as proxy
            # In a real scenario, you might want to extract the actual retrieved docs
            contexts = [
                f"{c.get('source', 'Unknown')} (seq: {c.get('seq_num', 'N/A')})"
                for c in citation_list
            ] if citation_list else ["No contexts retrieved"]
            
            return {
                "question": query,
                "answer": answer,
                "contexts": contexts,
                "citations": citation_list,
                "raw_result": result
            }
            
        except Exception as e:
            print(f"⚠️  Error running agent on query: {str(e)}")
            return {
                "question": query,
                "answer": f"Error: {str(e)}",
                "contexts": [],
                "citations": [],
                "raw_result": None
            }
    
    def evaluate_agent(
        self, 
        test_cases: List[Dict[str, Any]] = None,
        test_file: str = None
    ) -> Dict[str, Any]:
        """
        Full evaluation: run agent on test questions and measure quality.
        
        Args:
            test_cases: Explicit test cases (overrides test_file)
            test_file: Path to test cases JSON file
        
        Returns:
            Comprehensive evaluation results
        """
        # Load test cases
        if test_cases is None:
            test_cases = load_test_dataset(test_file)
        
        print(f"\n🧪 Running {len(test_cases)} test cases through agent...")
        
        # Run agent on each test case
        agent_results = []
        for i, test_case in enumerate(test_cases, 1):
            print(f"  [{i}/{len(test_cases)}] {test_case['question'][:50]}...", end=" ")
            result = self.run_agent_on_query(test_case["question"])
            agent_results.append(result)
            print("✓")
        
        # Prepare data for Ragas evaluation
        ragas_test_cases = []
        for result in agent_results:
            # Find corresponding ground truth if available
            ground_truth = next(
                (tc.get("ground_truth", "") for tc in test_cases 
                 if tc.get("question") == result["question"]),
                ""
            )
            
            ragas_test_cases.append({
                "question": result["question"],
                "answer": result["answer"],
                "contexts": result["contexts"],
                "ground_truth": ground_truth
            })
        
        # Run Ragas evaluation
        print("\n📊 Running Ragas evaluation metrics...")
        metrics = [
            "faithfulness",
            "answer_relevancy",
            "context_relevance",   # ragas 0.4.x: renamed from context_relevancy
            "context_precision",
        ]
        
        ragas_results = self.ragas_evaluator.evaluate(ragas_test_cases, metrics=metrics)
        
        # Combine results
        full_results = {
            "timestamp": str(__import__('datetime').datetime.now()),
            "test_cases_count": len(test_cases),
            "ragas_metrics": ragas_results,
            "agent_outputs": [
                {
                    "question": r["question"],
                    "answer_preview": r["answer"][:200] + "..." if len(r["answer"]) > 200 else r["answer"],
                    "citations": r["citations"]
                }
                for r in agent_results
            ]
        }
        
        return full_results
    
    def evaluate_from_log(
        self,
        log_file: str = None,
        output: str = None,
    ) -> Dict[str, Any]:
        """
        Evaluate directly from qa_context_log.json without re-running the agent.
        Uses synthetic ground truth generated by a powerful LLM.

        Args:
            log_file: path to qa_context_log.json (default: project root)
            output:   path for the JSON report

        Returns:
            Ragas evaluation results dict
        """
        return eval_from_logs(
            log_file=log_file or str(LOG_FILE),
            output=output,
        )

    def print_report(self, results: Dict[str, Any]) -> None:
        """Print formatted evaluation report."""
        print("\n" + "="*60)
        print("📈 TRAVEL ADVISOR RAGAS EVALUATION REPORT")
        print("="*60)
        
        print(f"\n📅 Timestamp: {results['timestamp']}")
        print(f"🧪 Test Cases: {results['test_cases_count']}")
        
        metrics = results.get("ragas_metrics", {})
        print(f"\n📊 Overall Aggregate Score: {metrics.get('aggregate_score', 0):.4f}")
        
        print("\n🎯 Metric Breakdown:")
        for metric, score in metrics.items():
            if metric not in ["aggregate_score", "per_question"]:
                status = "✓" if score >= 0.7 else "⚠" if score >= 0.5 else "✗"
                print(f"  {status} {metric}: {score:.4f}")
        
        print("\n" + "="*60)


def main():
    """Main evaluation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Travel Advisor with Ragas")
    parser.add_argument(
        "--test-file",
        help="Path to test cases JSON file",
        default="activity_planner/test_cases.json"
    )
    parser.add_argument(
        "--output",
        help="Output file for evaluation report",
        default="ragas_evaluation_report.json"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick evaluation with subset of test cases"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = TravelAdvisorEvaluator()
    
    # Load test cases
    test_cases = load_test_dataset(args.test_file)
    
    # Use subset if quick mode
    if args.quick:
        test_cases = test_cases[:2]
        print("⚡ Quick mode: using subset of test cases")
    
    # Run evaluation
    results = evaluator.evaluate_agent(test_cases=test_cases)
    
    # Print report
    evaluator.print_report(results)
    
    # Save results
    output_file = save_evaluation_report(results, args.output)
    print(f"\n✅ Report saved to {output_file}")


if __name__ == "__main__":
    main()
