# Load test dataset
with open("retrieval_test_set.json") as f:
    test_set = json.load(f)

# Initialize vector DB
db = VectorDatabaseManager()

# Run evaluation
metrics = db.evaluate_retrieval(test_set, k_values=[1, 3, 5, 10])

print("Retrieval Evaluation Results:")
for k, scores in metrics.items():
    print(f"\nk={k}:")
    print(f"  Recall:    {scores['recall']:.3f}")
    print(f"  Precision: {scores['precision']:.3f}")
    print(f"  MRR:       {scores['mrr']:.3f}")

# Analyze embeddings
all_embeddings = db.get_all_embeddings()  # Need to implement this
analyze_embeddings(all_embeddings)