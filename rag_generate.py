import requests
from retrival import retrieve_similar_chunks


def build_context(results):
    context_blocks = []
    for r in results:
        block = f"""[Source: Page {r['metadata'].get('page')}]{r['text']}"""
        context_blocks.append(block.strip())
    return "\n\n".join(context_blocks)


def call_ollama(prompt, model="llama3.2:3b"):
    result = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return result.json()["response"]



def rag_answer(query, top_k=3):

    results = retrieve_similar_chunks(query, top_k=top_k)
    context = build_context(results)

    prompt = f"""
You are a helpful AI assistant.
Answer the question using ONLY the context below.
If the answer is not present in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
""".strip()

    # 4️⃣ Generate answer
    answer = call_ollama(prompt)

    return answer, results


# if __name__ == "__main__":

#     while True:
#         query = input("Enter your question: ")
        
#         if query.lower() in ['exit', 'quit']:
#             print("Exiting RAG generation.")
#             break

#         answer, sources = rag_answer(query, top_k=3)

#         print(" Answer:\n")
#         print(answer)

#         print("Sources:")
#         for s in sources:
#             print(f"- Page {s['metadata'].get('page')}")
