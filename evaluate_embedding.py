import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

def analyze_embeddings(embedding_matrix: np.ndarray):
    # Dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=30)
    embeddings_2d = tsne.fit_transform(embedding_matrix)
    
    # Clustering
    kmeans = KMeans(n_clusters=10)
    clusters = kmeans.fit_predict(embedding_matrix)
    
    # Visualization
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                c=clusters, cmap='tab20', alpha=0.6)
    plt.title("Embedding Space Clustering")
    plt.show()
    
    # Cluster quality (higher = better separation)
    silhouette_score = metrics.silhouette_score(embedding_matrix, clusters)
    print(f"Silhouette Score: {silhouette_score:.2f}")