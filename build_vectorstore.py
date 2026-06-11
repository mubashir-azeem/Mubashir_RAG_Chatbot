import os
import pickle
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ====================================
# CONFIG
# ====================================

DATA_FOLDER = "data"
VECTORSTORE_FOLDER = "vectorstore"

# ====================================
# LOAD EMBEDDING MODEL
# ====================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    local_files_only=True
)

documents = []
metadata = []

# ====================================
# READ DOCUMENTS
# ====================================

for filename in os.listdir(DATA_FOLDER):

    filepath = os.path.join(DATA_FOLDER, filename)

    text = ""

    # ---------- PDF ----------

    if filename.lower().endswith(".pdf"):

        pdf = PdfReader(filepath)

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    # ---------- TXT ----------

    elif filename.lower().endswith(".txt"):

        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as f:

            text = f.read()

    else:
        continue

    # ====================================
    # CHUNKING WITH OVERLAP
    # ====================================

    chunk_size = 500
    chunk_overlap = 100

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        documents.append(chunk)

        metadata.append(
            {
                "source": filename
            }
        )

        start += (
            chunk_size
            - chunk_overlap
        )

# ====================================
# CREATE EMBEDDINGS
# ====================================

print(f"Total Chunks: {len(documents)}")

embeddings = model.encode(
    documents,
    show_progress_bar=True
)

embeddings = np.array(
    embeddings
).astype("float32")

# ====================================
# CREATE FAISS INDEX
# ====================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(
    embeddings
)

# ====================================
# SAVE VECTORSTORE
# ====================================

os.makedirs(
    VECTORSTORE_FOLDER,
    exist_ok=True
)

faiss.write_index(
    index,
    os.path.join(
        VECTORSTORE_FOLDER,
        "faiss_index.bin"
    )
)

with open(
    os.path.join(
        VECTORSTORE_FOLDER,
        "metadata.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        {
            "documents": documents,
            "metadata": metadata
        },
        f
    )

print("\n✅ Vector Store Created Successfully!")
print(f"✅ Total Chunks: {len(documents)}")