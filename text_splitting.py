from data_ingestion import all_docs as data
from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],
    length_function=len,
)

texts = text_splitter.split_documents(data)

print("Text splitting completed.")
print("Sample chunk content: ", texts[0].page_content[:500])
print(texts[0].metadata)

print(f"Number of chunks: {len(texts)}")