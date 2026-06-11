import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Mubashir AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

# =====================================
# GEMINI CONFIG
# =====================================

genai.configure(
    api_key=st.secrets["GEMINI_API_KEY"]
)

llm = genai.GenerativeModel(
    "gemini-pro-latest"
)

# =====================================
# LOAD EMBEDDING MODEL
# =====================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()

# =====================================
# LOAD VECTOR DATABASE
# =====================================

@st.cache_resource
def load_vectorstore():

    index = faiss.read_index(
        "vectorstore/faiss_index.bin"
    )

    with open(
        "vectorstore/metadata.pkl",
        "rb"
    ) as f:

        data = pickle.load(f)

    return (
        index,
        data["documents"],
        data["metadata"]
    )

index, documents, metadata = load_vectorstore()

# =====================================
# RETRIEVER
# =====================================

def retrieve_context(question):

    query_vector = embedding_model.encode(
        [question]
    )

    query_vector = np.array(
        query_vector
    ).astype("float32")

    distances, indices = index.search(
        query_vector,
        k=3
    )

    context = ""

    sources = []

    for idx in indices[0]:

        context += (
            documents[idx][:600]
            + "\n\n"
        )

        source = metadata[idx]["source"]

        if source not in sources:
            sources.append(source)

    best_distance = float(
        distances[0][0]
    )

    return (
        context,
        sources,
        best_distance
    )

# =====================================
# CHATBOT LOGIC
# =====================================

def ask_chatbot(question):

    context, sources, best_distance = retrieve_context(
        question
    )

    lower_question = question.lower()

    # Developer Questions

    if (
        "developer" in lower_question
        or "author" in lower_question
        or "creator" in lower_question
        or "who developed" in lower_question
        or "who created" in lower_question
    ):

        return (
            """
Mubashir Azeem Abbasi developed this chatbot.

Project:
RAG-Based Intelligent Document Assistant

Technology Stack:
• Gemini 3.5 Flash
• FAISS Vector Database
• all-MiniLM-L6-v2 Embeddings
• Streamlit

This chatbot uses Retrieval Augmented Generation (RAG)
to answer questions from uploaded documents and
general knowledge.
            """,
            ["Developer info.txt"]
        )

    # General Knowledge Mode

    if best_distance > 1.2:

        try:

            response = llm.generate_content(
                question
            )

            return (
                response.text,
                ["General Knowledge (Gemini)"]
            )

        except Exception as e:

            return (
                f"⚠️ Gemini Error:\n\n{str(e)}",
                []
            )

    # Document Mode

    prompt = f"""
You are Mubashir AI Knowledge Assistant.

Answer ONLY using the document context below.

Document Context:

{context}

Question:

{question}
"""

    try:

        response = llm.generate_content(
            prompt
        )

        return (
            response.text,
            sources
        )

    except Exception as e:

        return (
            f"⚠️ Gemini Error:\n\n{str(e)}",
            []
        )

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.title("🤖 Mubashir AI")

    st.markdown("---")

    st.subheader("📌 Project")

    st.write(
        "RAG-Based Intelligent Document Assistant"
    )

    st.markdown("---")

    st.subheader("👨‍💻 Developer")

    st.write(
        "Mubashir Azeem Abbasi"
    )

    st.markdown("---")

    st.subheader("⚙️ Technologies")

    st.write("• Gemini 3.5 Flash")
    st.write("• FAISS Vector Database")
    st.write("• all-MiniLM-L6-v2")
    st.write("• Streamlit")
    st.write("• Retrieval Augmented Generation")

    st.markdown("---")

    st.subheader("📚 Loaded Documents")

    unique_docs = sorted(
        set(
            item["source"]
            for item in metadata
        )
    )

    st.success(
        f"{len(unique_docs)} Documents Loaded"
    )

    for doc in unique_docs:

        st.write(
            f"📄 {doc}"
        )

    st.markdown("---")

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 Assalam-o-Alaikum! I am Mubashir AI Knowledge Assistant. Ask me anything from the uploaded documents or general knowledge."
            }
        ]

        st.rerun()

# =====================================
# HEADER
# =====================================

st.title(
    "🤖 Mubashir AI Knowledge Assistant"
)

st.caption(
    "AI-Powered RAG Chatbot | Developed by Mubashir Azeem Abbasi"
)

# =====================================
# SESSION STATE
# =====================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Assalam-o-Alaikum! I am Mubashir AI Knowledge Assistant. Ask me anything from the uploaded documents or general knowledge."
        }
    ]

# =====================================
# DISPLAY CHAT HISTORY
# =====================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

# =====================================
# CHAT INPUT
# =====================================

question = st.chat_input(
    "Ask anything about your documents..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            answer, sources = ask_chatbot(
                question
            )

        st.markdown(answer)

        if len(sources) > 0:

            st.markdown("### 📚 Sources")

            for source in sources:

                st.write(
                    f"📄 {source}"
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
