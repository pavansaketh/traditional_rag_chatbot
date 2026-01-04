import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from text_splitting import texts

os.makedirs("storage", exist_ok=True)

texts_only = [doc.page_content for doc in texts]
metadata = [doc.metadata for doc in texts]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

texts_for_embedding = [doc.page_content for doc in texts]

embeddings = model.encode(texts_for_embedding)

np.save("storage/embeddings.npy", embeddings)

print("Embeddings generation completed.")
print("Sample embedding for first chunk: ", embeddings[0][:25]) 

print(f"Total number of embeddings generated: {len(embeddings)}")
print(f"shape of embeddings: {embeddings[0].shape}")

with open("storage/texts.json", "w", encoding="utf-8") as f:
    json.dump(texts_only, f, ensure_ascii=False, indent=2)

# Save metadata
with open("storage/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("✅ Embeddings, texts, and metadata saved locally")