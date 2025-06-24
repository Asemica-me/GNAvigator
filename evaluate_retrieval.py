import asyncio
import os
import json
import time
import pandas as pd
import numpy as np
from llm_handler import RAGOrchestrator
from dotenv import load_dotenv

load_dotenv()
METRICS_DIR = os.path.join("data", "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)

class RAGEvaluator:
    def __init__(self, mistral_api_key: str):
        self.orchestrator = RAGOrchestrator(mistral_api_key=mistral_api_key)
        self.metrics = []
    
    async def evaluate_retrieval(self, test_file: str, top_k: int = 5):
        """Run evaluation on test dataset"""
        with open(test_file, encoding="utf-8") as f:
            test_data = json.load(f)
        
        for item in test_data:
            result = await self._evaluate_query(
                item["question"],
                set(item["relevant_docs"]),
                top_k
            )
            self.metrics.append(result)
        
        return self._aggregate_metrics()
    
    async def _evaluate_query(self, question: str, relevant_docs: set, top_k: int):
        """Evaluate a single query"""
        # Time the retrieval
        start_time = time.time()
        retrieved_docs = await self.orchestrator.retrieve_docs(question, top_k)
        retrieval_time = time.time() - start_time
        
        # Get document IDs
        #print("Sample retrieved_docs:", retrieved_docs)
        retrieved_ids = {doc["id"] for doc in retrieved_docs}
        
        # Calculate metrics
        relevant_retrieved = len(retrieved_ids & relevant_docs)
        precision = relevant_retrieved / top_k
        recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0
        
        # Calculate MRR
        reciprocal_rank = 0
        for rank, doc in enumerate(retrieved_docs, 1):
            if doc["id"] in relevant_docs:
                reciprocal_rank = 1 / rank
                break
        
        # Token usage (assuming tokenize method exists)
        input_tokens = len(self.orchestrator.tokenize(question))
        
        return {
            "question": question,
            "precision@k": precision,
            "recall@k": recall,
            "mrr": reciprocal_rank,
            "retrieval_time": retrieval_time,
            "input_tokens": input_tokens,
            "retrieved_docs": list(retrieved_ids),
            "relevant_docs": list(relevant_docs)
        }
    
    def _aggregate_metrics(self):
        """Calculate aggregate statistics"""
        df = pd.DataFrame(self.metrics)
        
        return {
            "num_queries": len(df),
            "avg_precision@k": df["precision@k"].mean(),
            "avg_recall@k": df["recall@k"].mean(),
            "avg_mrr": df["mrr"].mean(),
            "avg_retrieval_time": df["retrieval_time"].mean(),
            "total_input_tokens": df["input_tokens"].sum(),
            "by_question": df.to_dict(orient="records")
        }
    
    def save_reports(self, metrics, prefix="evaluation"):
        """Save all reports to metrics directory"""
        # Save aggregate metrics
        metrics_file = os.path.join(METRICS_DIR, f"{prefix}_metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in metrics.items() if k != "by_question"}, f, indent=2)
        
        # Save detailed results as CSV
        details_file = os.path.join(METRICS_DIR, f"{prefix}_details.csv")
        df = pd.DataFrame(metrics["by_question"])
        df.to_csv(details_file, index=False)
        
        # Save error analysis
        errors_file = os.path.join(METRICS_DIR, f"{prefix}_errors.csv")
        error_df = df[(df["precision@k"] < 0.3) | (df["recall@k"] < 0.3)]
        if not error_df.empty:
            error_df.to_csv(errors_file, index=False)
        
        print(f"Reports saved to {METRICS_DIR}:")
        print(f"- {prefix}_metrics.json (aggregate metrics)")
        print(f"- {prefix}_details.csv (per-query results)")
        if not error_df.empty:
            print(f"- {prefix}_errors.csv ({len(error_df)} problematic queries)")

async def main():
    evaluator = RAGEvaluator(mistral_api_key=os.getenv("MISTRAL_API_KEY"))
    test_file = os.path.join("data", "test_dataset.json")
    metrics = await evaluator.evaluate_retrieval(test_file, top_k=5)
    
    print("\n=== Evaluation Results ===")
    print(f"Queries evaluated: {metrics['num_queries']}")
    print(f"Avg Precision@5: {metrics['avg_precision@k']:.2f}")
    print(f"Avg Recall@5: {metrics['avg_recall@k']:.2f}")
    print(f"Mean Reciprocal Rank: {metrics['avg_mrr']:.2f}")
    print(f"Avg Retrieval Time: {metrics['avg_retrieval_time']:.3f}s")
    print(f"Total Input Tokens: {metrics['total_input_tokens']}")
    
    evaluator.save_reports(metrics)

if __name__ == "__main__":
    asyncio.run(main())