import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    local_files_only=True
)

# Load FAISS index
index = faiss.read_index("vectorstore/faiss_index.bin")

# Load metadata
with open("vectorstore/metadata.pkl", "rb") as f:
    data = pickle.load(f)

documents = data["documents"]
metadata = data["metadata"]

while True:

    question = input("\nAsk Question: ")

    if question.lower() == "exit":
        break

    query_embedding = model.encode([question])

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        k=1
    )

    print("\nTop Results:\n")

    for i in indices[0]:

        print("=" * 80)

        print(documents[i][:500])

        print("\nSource:")
        print(metadata[i]['source'])

        print()