import json
import numpy as np

embeddings = np.load("storage/embeddings.npy")

print("Embeddings loaded successfully.")
print("Sample embedding for first chunk: ", embeddings[0][:25])

texts = json.load(open("storage/texts.json", encoding="utf-8"))
print("Texts loaded successfully.")
print("Sample chunk content: ", texts[0][:500])

metadata = json.load(open("storage/metadata.json", encoding="utf-8"))
print("Metadata loaded successfully.")
print("Sample metadata for first chunk: ", metadata[0])