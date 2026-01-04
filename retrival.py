import os 
import json
import numpy as np
from embeddings import model


embeddings = np.load("storage/embeddings.npy")
texts = json.load(open("storage/texts.json", encoding="utf-8"))
metadata = json.load(open("storage/metadata.json", encoding="utf-8"))



def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def retrieve_similar_chunks(query, top_k=3):
    query_embedding = model.encode([query])[0]
    
    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_k_indices:
        results.append({
            "text": texts[idx],
            "metadata": metadata[idx],
            "similarity": similarities[idx]
        })
    
    return results

if __name__ == "__main__":
    while True:
        sample_query = input("Enter your query: ")

        if sample_query.lower() in ['exit', 'quit']:
            print("Exiting retrieval.")
            break

        results = retrieve_similar_chunks(sample_query, top_k=1)
        
        print("Top similar chunks for the query:")
        for i, res in enumerate(results):
            print(f"\nRank {i+1}:")
            print(f"Similarity Score: {res['similarity']:.4f}")
            print(f"Text Chunk: {res['text'][:500]}...")
            print(f"Metadata: {res['metadata']}")