from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

result = embeddings.embed_query(
    "What is SD-WAN?"
)

print(len(result))
print(result)