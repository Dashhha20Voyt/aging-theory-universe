# src/utils/clustering_fast.py

from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import numpy as np
import pandas as pd
import time

def fast_cluster_papers(df_papers, batch_size=1000):
    """
    Fast clustering of papers using SentenceTransformer + UMAP + HDBSCAN
    Optimized for large datasets (10K+ papers)
    """
    start_time = time.time()
    
    # --- STEP 1: Generate embeddings in batches ---
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
    
    texts = []
    for _, row in df_papers.iterrows():
        text = row.get('abstract') or row.get('full_text') or ""
        texts.append(text)
    
    print(f"✅ Texts prepared: {len(texts)}")
    
    # Encode in batches to avoid OOM
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"📦 Encoding batch {i//batch_size + 1} / {len(texts)//batch_size + 1}")
        embeddings = model.encode(batch, show_progress_bar=True, batch_size=32)
        all_embeddings.append(embeddings)
    
    embeddings = np.vstack(all_embeddings)
    print(f"✅ Embeddings generated: {embeddings.shape}")
    
    # --- STEP 2: UMAP with optimization ---
    print("🚀 Reducing dimensionality with UMAP...")
    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=5,  # 5D instead of 2D → faster and better for clustering
        metric='cosine',
        random_state=42,
        verbose=True,
        n_jobs=-1  # Use all CPU cores
    )
    embedding_umap = reducer.fit_transform(embeddings)
    print(f"✅ UMAP completed: {embedding_umap.shape}")
    
    # --- STEP 3: HDBSCAN with optimization ---
    print("🧩 Clustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        metric='euclidean',
        cluster_selection_method='eom',
        core_dist_n_jobs=-1  # Parallelize core distance calculation
    )
    labels = clusterer.fit_predict(embedding_umap)
    print(f"✅ Clusters assigned: {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
    
    # --- STEP 4: Assign to DataFrame ---
    df_papers['cluster_id'] = labels
    df_papers['theory_id'] = range(1, len(df_papers) + 1)
    
    end_time = time.time()
    print(f"⏱️ Total clustering time: {end_time - start_time:.2f} seconds")
    
    return df_papers

# --- USAGE ---
if __name__ == "__main__":
    # Load your DataFrame
    df_papers = pd.read_csv('data/processed/extracted_answers.csv')
    
    # Run fast clustering
    df_papers = fast_cluster_papers(df_papers)
    
    # Save result
    df_papers.to_csv('data/processed/clustered_papers.csv', index=False)
    print("✅ Clustered data saved to data/processed/clustered_papers.csv")
