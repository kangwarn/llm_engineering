import chromadb
from chromadb.utils import embedding_functions

# Define the settings for Chroma with the updated configuration
client = chromadb.PersistentClient(path="chroma_db")

# Define the OpenAI embedding function
OPENAI_BASE_URL = 'https://llm.ai.broadcom.net/api/v1'
OPENAI_API_KEY = "57fa6c09-20a4-4cc0-892e-23d0a37b26c2"
model = "BAAI/bge-en-icl"
# embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=model, api_base=OPENAI_BASE_URL)
embedding_function = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, api_base=OPENAI_BASE_URL, api_type="azure", model_name=model, api_version="v1")

# Create a collection with the new client
collection = client.get_or_create_collection(
    name="example_collection",
    embedding_function=embedding_function
)

# Sample data
texts = [
    "ChromaDB is a lightweight and fast vector database.",
    "OpenAI's API provides embeddings for semantic search.",
    "Vector databases are essential for managing embeddings."
]

# Add data to the collection
collection.add(
    documents=texts,
    ids=["doc1", "doc2", "doc3"],
    metadatas=[
        {"category": "tech", "source": "docs"},
        {"category": "AI", "source": "blog"},
        {"category": "DB", "source": "tutorial"}
    ]
)

# Query the collection
results = collection.query(
    query_texts=["What is ChromaDB?"],
    n_results=2
)

# Output results
for doc in results["documents"]:
    print("Retrieved document:", doc)



