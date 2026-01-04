import os
from langchain_community.document_loaders import PyPDFLoader


PDF_DIR = "data/pdfs"

all_docs = []

for file in os.listdir(PDF_DIR):
    if file.endswith(".pdf"):
        file_path = os.path.join(PDF_DIR, file)

        loader = PyPDFLoader(file_path)
        docs = loader.load() 

        all_docs.extend(docs)

print(f"Total pages loaded: {len(all_docs)}")


print("Data ingestion completed.")
print("Sample page content: ", all_docs[0].page_content[:500])
print(all_docs[0].metadata)
print(f"Number of pages: {len(all_docs)}")