import asyncio
import os
import shutil
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from create_chunks_dict import *
import torch
import gc
import logging
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment setup
os.environ["SPACY_WARNING_IGNORE"] = "W008"
os.environ["NLTK_DATA"] = "/tmp/nlp_data/nltk"

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
FAISS_DIR = ".faiss_db"
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
METADATA_PATH = os.path.join(FAISS_DIR, "metadata.pkl")
EVALUATION_DIR = "retrieval_evaluation"
EVALUATION_RESULTS_PATH = os.path.join(EVALUATION_DIR, "results.json")
EVALUATION_PLOTS_PATH = os.path.join(EVALUATION_DIR, "metrics_plot.png")

# Memory management constants
MAX_CACHE_SIZE = 1000  # Max cached embeddings
CLEANUP_INTERVAL = 10  # Cleanup every 10 queries
GPU_CACHE_THRESHOLD = 80  # GPU memory usage threshold for offloading

# Create evaluation directory
os.makedirs(EVALUATION_DIR, exist_ok=True)

class VectorDatabaseManager:
    # ... (rest of the class remains the same until the query method)

    def query(self, question: str, top_k: int = 5) -> list:
        """Perform similarity search with memory management"""
        try:
            # ... (existing query code remains the same)

            # Format results - ADD CHUNK ID TO RESULTS
            results = []
            for i, score in zip(indices[0], scores[0]):
                if i < 0:  # FAISS returns -1 for invalid indices
                    continue
                try:
                    item = self.metadata_db[i]
                    results.append({
                        "chunk_id": item["id"],  # Add chunk ID for evaluation
                        "score": float(score),
                        "title": item["metadata"]["title"],
                        "source": item["metadata"]["source"],
                        "content": item["document"][:200] + "..." if item["document"] else ""
                    })
                except IndexError:
                    logger.warning(f"Invalid index {i} in metadata database")
            
            return results
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return []
        finally:
            # Clean up after each query
            self._clear_memory()

    # ... (rest of the class remains the same)

    def evaluate_retrieval(
        self, 
        test_set: list, 
        k_values: list = [1, 3, 5, 10],
        save_results: bool = True
    ) -> dict:
        """
        Evaluate retrieval performance using standard metrics.
        
        Args:
            test_set: List of test examples in format:
                [{
                    "query": str,
                    "relevant_chunks": list[str]  # List of chunk IDs
                }]
            k_values: List of k values to evaluate at
            save_results: Whether to save results to file
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Starting retrieval evaluation with {len(test_set)} test queries")
        
        # Initialize results structure
        results = {
            "config": {
                "model": EMBEDDING_MODEL,
                "test_set_size": len(test_set),
                "k_values": k_values,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "per_query": {},
            "aggregate": {}
        }
        
        # Initialize metrics storage
        for k in k_values:
            results["aggregate"][f"k={k}"] = {
                "recall": [],
                "precision": [],
                "f1": [],
                "mrr": [],
                "average_precision": []
            }
        
        # Process each test query
        for idx, test_case in enumerate(tqdm(test_set, desc="Evaluating queries")):
            query = test_case["query"]
            relevant_ids = set(test_case["relevant_chunks"])
            
            # Get top-k results for max k value
            max_k = max(k_values)
            retrieved = self.query(query, top_k=max_k)
            
            # Store per-query results
            query_results = {
                "query": query,
                "relevant_chunks": list(relevant_ids),
                "retrieved_chunks": [r["chunk_id"] for r in retrieved],
                "metrics": {}
            }
            
            # Calculate metrics for each k
            for k in k_values:
                k_results = self._calculate_metrics(
                    query, 
                    retrieved[:k], 
                    relevant_ids, 
                    k
                )
                query_results["metrics"][f"k={k}"] = k_results
                
                # Store for aggregate calculation
                for metric in ["recall", "precision", "f1", "mrr", "average_precision"]:
                    results["aggregate"][f"k={k}"][metric].append(k_results[metric])
            
            results["per_query"][f"query_{idx}"] = query_results
        
        # Calculate aggregate metrics
        for k in k_values:
            k_agg = results["aggregate"][f"k={k}"]
            for metric in k_agg.keys():
                k_agg[f"mean_{metric}"] = np.mean(k_agg[metric])
                k_agg[f"std_{metric}"] = np.std(k_agg[metric])
                k_agg[f"min_{metric}"] = np.min(k_agg[metric])
                k_agg[f"max_{metric}"] = np.max(k_agg[metric])
        
        # Save results
        if save_results:
            self._save_evaluation_results(results)
            self._plot_metrics(results)
        
        logger.info("Retrieval evaluation completed")
        return results

    def _calculate_metrics(
        self, 
        query: str, 
        retrieved: list, 
        relevant_ids: set, 
        k: int
    ) -> dict:
        """Calculate evaluation metrics for a single query"""
        retrieved_ids = [r["chunk_id"] for r in retrieved]
        
        # Calculate binary relevance vector
        y_true = [1 if chunk_id in relevant_ids else 0 for chunk_id in retrieved_ids]
        y_pred = [1] * len(retrieved_ids)  # All retrieved are considered positive
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        # MRR (Mean Reciprocal Rank)
        rr = 0.0
        for rank, item in enumerate(retrieved, 1):
            if item["chunk_id"] in relevant_ids:
                rr = 1.0 / rank
                break
        
        # Average Precision
        ap = 0.0
        relevant_count = 0
        for rank, item in enumerate(retrieved, 1):
            if item["chunk_id"] in relevant_ids:
                relevant_count += 1
                ap += relevant_count / rank
        
        # Avoid division by zero if no relevant docs
        ap = ap / len(relevant_ids) if relevant_ids else 0.0
        
        return {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "mrr": rr,
            "average_precision": ap
        }

    def _save_evaluation_results(self, results: dict):
        """Save evaluation results to JSON file"""
        try:
            with open(EVALUATION_RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Evaluation results saved to {EVALUATION_RESULTS_PATH}")
        except Exception as e:
            logger.error(f"Error saving evaluation results: {str(e)}")

    def _plot_metrics(self, results: dict):
        """Create visualization of evaluation metrics"""
        try:
            k_values = results["config"]["k_values"]
            metrics = ["recall", "precision", "f1", "mrr", "average_precision"]
            
            # Prepare data
            data = []
            for k in k_values:
                agg = results["aggregate"][f"k={k}"]
                for metric in metrics:
                    data.append({
                        "k": k,
                        "metric": metric,
                        "value": agg[f"mean_{metric}"]
                    })
            
            # Create DataFrame for plotting
            import pandas as pd
            df = pd.DataFrame(data)
            
            # Plot
            plt.figure(figsize=(12, 8))
            sns.set_theme(style="whitegrid")
            ax = sns.lineplot(
                data=df, 
                x="k", 
                y="value", 
                hue="metric", 
                marker="o",
                markersize=8
            )
            
            plt.title("Retrieval Performance Metrics", fontsize=16)
            plt.xlabel("Top-k Results", fontsize=12)
            plt.ylabel("Score", fontsize=12)
            plt.xticks(k_values)
            plt.ylim(0, 1.05)
            plt.legend(title="Metrics", loc="best")
            plt.tight_layout()
            
            # Save plot
            plt.savefig(EVALUATION_PLOTS_PATH)
            plt.close()
            logger.info(f"Metrics plot saved to {EVALUATION_PLOTS_PATH}")
        except Exception as e:
            logger.error(f"Error creating metrics plot: {str(e)}")

# ... (rest of the file remains the same until the main function)

def generate_test_set(database_manager, num_queries=50):
    """
    Generate a test set by sampling queries from the database.
    For real evaluation, you should create a curated test set.
    """
    test_set = []
    
    # Sample some documents to create queries
    sample_docs = database_manager.metadata_db[:1000:20]  # Every 20th doc
    
    for doc in sample_docs:
        # Create a query based on document content
        content = doc["document"]
        words = content.split()[:10]  # First 10 words as query
        query = " ".join(words)
        
        # Consider this document as relevant
        test_set.append({
            "query": query,
            "relevant_chunks": [doc["id"]]
        })
        
        if len(test_set) >= num_queries:
            break
    
    return test_set

async def main():
    # Initialize vector database manager
    db_manager = VectorDatabaseManager()

    # Process and store chunks only if no existing database
    if not db_manager.metadata_db:
        await db_manager.process_and_store_chunks(SITEMAP_PATH, BASE_DOMAIN)
    else:
        print("Using existing FAISS database")

    # Generate or load test set
    TEST_SET_PATH = os.path.join(EVALUATION_DIR, "test_set.json")
    if os.path.exists(TEST_SET_PATH):
        print("Loading existing test set")
        with open(TEST_SET_PATH, "r") as f:
            test_set = json.load(f)
    else:
        print("Generating new test set")
        test_set = generate_test_set(db_manager, num_queries=100)
        with open(TEST_SET_PATH, "w") as f:
            json.dump(test_set, f, indent=2)
    
    # Run evaluation
    results = db_manager.evaluate_retrieval(
        test_set=test_set,
        k_values=[1, 3, 5, 10, 20],
        save_results=True
    )
    
    # Print summary results
    print("\nRetrieval Evaluation Summary:")
    print(f"{'k':<5} {'Precision':<10} {'Recall':<10} {'F1':<10} {'MRR':<10} {'AvgP':<10}")
    for k, metrics in results["aggregate"].items():
        print(f"{k.split('=')[1]:<5} "
              f"{metrics['mean_precision']:.4f}     "
              f"{metrics['mean_recall']:.4f}     "
              f"{metrics['mean_f1']:.4f}     "
              f"{metrics['mean_mrr']:.4f}     "
              f"{metrics['mean_average_precision']:.4f}")

if __name__ == "__main__":
    asyncio.run(main())