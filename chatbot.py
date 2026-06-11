import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ==========================
# GEMINI CONFIGURATION
# ==========================

API_KEY = "AQ.Ab8RN6LNaTs9JVIupqA3TXVhf9_ApYykw-s_ibSjhB2BIMcBWw"

genai.configure(api_key=API_KEY)

llm = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# ==========================
# LOAD EMBEDDING MODEL
# ==========================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    local_files_only=True
)

# ==========================
# LOAD FAISS INDEX
# ==========================

index = faiss.read_index(
    "vectorstore/faiss_index.bin"
)

with open(
    "vectorstore/metadata.pkl",
    "rb"
) as f:

    data = pickle.load(f)

documents = data["documents"]
metadata = data["metadata"]

# ==========================
# CHAT LOOP
# ==========================

while True:

    question = input("\nAsk Question: ")

    if question.lower() == "exit":
        break

    # --------------------------
    # Create Question Embedding
    # --------------------------

    query_vector = embedding_model.encode(
        [question]
    )

    query_vector = np.array(
        query_vector
    ).astype("float32")

    # --------------------------
    # Search FAISS
    # --------------------------

    distances, indices = index.search(
        query_vector,
        k=5
    )

    context = ""

    sources = set()

    # --------------------------
    # Distance Filtering
    # --------------------------

    for rank, idx in enumerate(indices[0]):

        distance = distances[0][rank]

        if distance > 1.5:
            continue

        context += documents[idx]
        context += "\n\n"

        sources.add(
            metadata[idx]["source"]
        )

    # --------------------------
    # Fallback if nothing found
    # --------------------------

    if len(context.strip()) == 0:

        context = """
No relevant document information found.
Use general knowledge.
"""

    # --------------------------
    # Prompt
    # --------------------------

    prompt = f"""
You are Mubashir AI Knowledge Assistant.

Developer:
Mubashir Azeem Abbasi

Rules:

1. Use document context first.

2. If answer exists in documents,
   answer from documents.

3. If document information is not available,
   answer using general knowledge.

4. If asked who developed this chatbot,
   answer:
   "This chatbot was developed by Mubashir Azeem Abbasi."

Document Context:
{context}

Question:
{question}
"""

    response = llm.generate_content(
        prompt
    )

    print("\n" + "=" * 60)

    print("\nAnswer:\n")

    print(response.text)

    print("\nSources:")

    if len(sources) == 0:

        print(
            "General Knowledge (Gemini)"
        )

    else:

        for source in sources:

            print(
                f"- {source}"
            )

    print("\n" + "=" * 60)