import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from git import Repo

from sentence_transformers import SentenceTransformer
from openai import OpenAI

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document  # <-- modern import

load_dotenv()

# --- CONFIG ---
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
INDEX_NAME = "repochat-rag"

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".tsx", ".jsx", ".ipynb", ".java",
    ".cpp", ".ts", ".go", ".rs", ".vue", ".swift", ".c", ".h",
}
IGNORED_DIRS = {
    "node_modules", "venv", "env", "dist", "build", ".git",
    "__pycache__", ".next", ".vscode", "vendor",
}

# --- CLIENTS ---
# Groq (OpenAI-compatible)
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

# Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(INDEX_NAME)

# Embeddings (HuggingFace)
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


# --- UTILS ---
@st.cache_resource(show_spinner=False)
def _load_sentence_transformer(model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    return SentenceTransformer(model_name)


def get_huggingface_embeddings(text: str):
    model = _load_sentence_transformer()
    return model.encode(text)


def existing_namespaces(index):
    try:
        stats = index.describe_index_stats()
        return list(stats.get("namespaces", {}).keys())
    except Exception as e:
        print(f"Error retrieving namespaces: {e}")
        return []


def is_repo_processed(namespace: str, index) -> bool:
    try:
        stats = index.describe_index_stats()
        return namespace in stats.get("namespaces", {})
    except Exception as e:
        print(f"Error checking Pinecone index: {e}")
        return False


def clone_repository(repo_url: str) -> str | None:
    repo_name = repo_url.rstrip("/").split("/")[-1]
    repo_path = Path(repo_name)

    if repo_path.exists():
        return str(repo_path)

    try:
        Repo.clone_from(repo_url, str(repo_path))
        return str(repo_path)
    except Exception as e:
        st.error(f"Error cloning repository: {e}")
        return None


def get_file_content(file_path: str, repo_path: str):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        rel_path = os.path.relpath(file_path, repo_path)
        return {"name": rel_path, "content": content}
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def get_main_files_content(repo_path: str):
    files_content = []
    try:
        for root, _, files in os.walk(repo_path):
            # skip ignored directories
            if any(ignored in root for ignored in IGNORED_DIRS):
                continue

            for filename in files:
                ext = os.path.splitext(filename)[1]
                if ext not in SUPPORTED_EXTENSIONS:
                    continue

                file_path = os.path.join(root, filename)
                item = get_file_content(file_path, repo_path)
                if item:
                    files_content.append(item)

    except Exception as e:
        print(f"Error reading repository: {e}")

    return files_content


def pinecone_setup(namespace: str, file_content: list[dict]):
    # Create vectorstore handle
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=hf_embeddings,
        namespace=namespace,
    )

    # Build Documents (store 'text' in metadata so retrieval works)
    documents: list[Document] = []
    for f in file_content:
        doc_text = f"{f['name']}\n{f['content']}"
        documents.append(
            Document(
                page_content=doc_text,
                metadata={"source": f["name"], "text": doc_text},
            )
        )

    # Upsert
    PineconeVectorStore.from_documents(
        documents=documents,
        embedding=hf_embeddings,
        index_name=INDEX_NAME,
        namespace=namespace,
    )

    return vectorstore

def perform_rag(query: str, namespace: str, max_context_chars: int = 18000):
    query_vec = get_huggingface_embeddings(query)

    top_matches = pinecone_index.query(
        vector=query_vec.tolist(),
        top_k=8,
        include_metadata=True,
        namespace=namespace,
    )

    matches = top_matches.get("matches", []) or []

    # Collect contexts, then truncate to max_context_chars
    contexts = []
    for item in matches:
        md = item.get("metadata") or {}
        txt = md.get("text")
        if txt:
            contexts.append(txt)

    joined = "\n\n-------\n\n".join(contexts)

    if len(joined) > max_context_chars:
        joined = joined[:max_context_chars] + "\n\n[...truncated context...]"

    augmented_query = (
        "<CONTEXT>\n"
        f"{joined if joined else 'No relevant context found.'}\n"
        "-------\n"
        "</CONTEXT>\n\n"
        "MY QUESTION:\n"
        f"{query}"
    )

    system_prompt = (
        "You are a Senior Software Engineer.\n"
        "Use ONLY the provided context. If insufficient, say what you need."
    )
    return system_prompt, augmented_query


# --- UI / APP ---
st.set_page_config(page_title="Ask Your Code", layout="wide")
st.title("Ask Your Code")

# Session state: chat threads
if "chats" not in st.session_state:
    st.session_state.chats = {"Chat 1": []}
    st.session_state.active_chat = "Chat 1"

with st.sidebar:
    st.title("GitHub Repo...")

    pasted = st.text_input(
        "Paste a GitHub link here:",
        placeholder="https://github.com/...",
    )

    all_namespaces = existing_namespaces(pinecone_index)

    option = st.selectbox(
        "Select an indexed repo:",
        options=all_namespaces if all_namespaces else ["Not available at this time"],
        index=None if all_namespaces else 0,
        placeholder="Select a namespace...",
    )

    st.divider()

    text_input = None
    if pasted:
        text_input = pasted.strip()
    elif option and option != "Not available at this time":
        text_input = option

    if text_input:
        st.write(f"Selected: {text_input}")
        st.divider()

        st.subheader("Chat Threads")
        for thread in list(st.session_state.chats.keys()):
            if st.button(thread):
                st.session_state.active_chat = thread

        if st.button("Add Chat"):
            new_chat_id = f"Chat {len(st.session_state.chats) + 1}"
            st.session_state.chats[new_chat_id] = []
            st.session_state.active_chat = new_chat_id

active_chat = st.session_state.active_chat
st.subheader(active_chat)
st.divider()

# Initialize message history per active chat (store in chats dict)
if active_chat not in st.session_state.chats:
    st.session_state.chats[active_chat] = []

if not st.session_state.chats[active_chat]:
    st.session_state.chats[active_chat].append({"role": "assistant", "content": "Welcome! How can I assist you today?"})

# Render messages
for message in st.session_state.chats[active_chat]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not text_input:
    st.warning("Please choose an indexed repo or paste in a url")
    st.stop()

# Index repo if needed
if not is_repo_processed(text_input, pinecone_index):
    with st.spinner("Processing repository..."):
        repo_path = clone_repository(text_input)
        if repo_path:
            file_content = get_main_files_content(repo_path)
            pinecone_setup(text_input, file_content)
        else:
            st.stop()

# Chat input
prompt = st.chat_input("What would you like to know?")
if prompt:
    st.session_state.chats[active_chat].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        system_prompt, augmented_query = perform_rag(prompt, text_input)

        llm_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query},
            ],
            stream=True,
        )

        response_text = st.write_stream(llm_response)

    st.session_state.chats[active_chat].append({"role": "assistant", "content": response_text})
