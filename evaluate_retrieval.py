class VectorDatabaseManager:
    # ... existing code ...
    
    def evaluate_retrieval(self, test_set: List[Dict], k_values: List[int] = [1, 3, 5, 10]):
        results = {}
        for k in k_values:
            results[k] = {
                'recall': [],
                'precision': [],
                'mrr': []
            }
        
        for test_case in test_set:
            query = test_case["query"]
            relevant_ids = set(test_case["relevant_chunks"])
            
            # Get top-k results
            retrieved = self.query(query, top_k=max(k_values))
            retrieved_ids = [item['chunk_id'] for item in retrieved]
            
            for k in k_values:
                top_k = retrieved_ids[:k]
                
                # Recall@k
                relevant_found = len(relevant_ids & set(top_k))
                recall = relevant_found / len(relevant_ids) if relevant_ids else 0
                results[k]['recall'].append(recall)
                
                # Precision@k
                precision = relevant_found / k
                results[k]['precision'].append(precision)
                
                # MRR@k
                for rank, chunk_id in enumerate(top_k, 1):
                    if chunk_id in relevant_ids:
                        results[k]['mrr'].append(1 / rank)
                        break
                else:
                    results[k]['mrr'].append(0)
        
        # Aggregate metrics
        metrics_summary = {}
        for k in k_values:
            metrics_summary[k] = {
                "recall": np.mean(results[k]['recall']),
                "precision": np.mean(results[k]['precision']),
                "mrr": np.mean(results[k]['mrr'])
            }
        return metrics_summary