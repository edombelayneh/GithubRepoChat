
# import streamlit as st
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from langchain_pinecone import PineconeVectorStore
# # from langchain.embeddings import OpenAIEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from pinecone import Pinecone
# import os
# import tempfile
# from github import Github, Repository
# from git import Repo
# from openai import OpenAI
# from pathlib import Path
# from langchain.schema import Document
# import tempfile
# import shutil
# from dotenv import load_dotenv

# load_dotenv()

# # Functions and other initializations
# api_key = os.environ["PINECONE_API_KEY"]
# api_key_groq = os.environ["GROQ_API_KEY"]

# # Initialize Groq
# client = OpenAI(
#     base_url="https://api.groq.com/openai/v1",
#     api_key=api_key_groq
# )

# # Initialize Pinecone
# pc = Pinecone(api_key=api_key)
# pinecone_index = pc.Index("repochat-rag")


# def pinecone_setup(text_input, file_content):
#     vectorstore = PineconeVectorStore(index_name="repochat-rag", embedding=HuggingFaceEmbeddings())

#     # Insert the codebase embeddings into Pinecone
#     documents = []
#     for file in file_content:
#         doc = Document(
#             page_content=f"{file['name']}\n{file['content']}",
#             metadata={"source": file['name']}
#         )
#         documents.append(doc)

#     vectorstore = PineconeVectorStore.from_documents(
#         documents=documents,
#         embedding=HuggingFaceEmbeddings(),
#         index_name="repochat-rag",
#         namespace=text_input
#     )


# def is_repo_processed(repo_url, pinecone_index):
#     """Checks if a repository has already been processed and indexed in Pinecone.

#     Args:
#         repo_url: The URL of the GitHub repository.
#         pinecone_index: Pinecone index object.

#     Returns:
#         True if the repository has been processed, False otherwise.
#     """
#     namespace = repo_url
#     try:
#         response = pinecone_index.describe_index_stats()
#         if namespace in response.get('namespaces', {}):
#             return True
#     except Exception as e:
#         print(f"Error checking Pinecone index: {str(e)}")
#     return False


# def clone_repository(repo_url):
#     """Clones a GitHub repository if it hasn't been cloned already.

#     Args:
#         repo_url: The URL of the GitHub repository.

#     Returns:
#         The path to the cloned repository or a message if already cloned.
#     """
#     repo_name = repo_url.split("/")[-1]
#     repo_path = f"{repo_name}"

#     if os.path.exists(repo_path):
#         # st.write(f"Repository already exists at {repo_path}")
#         return repo_path

#     try:
#         Repo.clone_from(repo_url, repo_path)
#         # st.write(f"Repository cloned at {repo_path}")
#         return repo_path
#     except Exception as e:
#         st.write(f"Error cloning repository: {str(e)}")
#         return None

# def get_file_content(file_path, repo_path):
#     """
#     Get content of a single file.

#     Args:
#         file_path (str): Path to the file

#     Returns:
#         Optional[Dict[str, str]]: Dictionary with file name and content
#     """
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             content = f.read()

#         rel_path = os.path.relpath(file_path, repo_path)
#         return {
#             "name": rel_path,
#             "content": content
#         }
#     except Exception as e:
#         print(f"Error processing file {file_path}: {str(e)}")
#         return None


# def get_main_files_content(repo_path: str):
#     """
#     Get content of supported code files from the local repository.

#     Args:
#         repo_path: Path to the local repository

#     Returns:
#         List of dictionaries containing file names and contents
#     """
#     SUPPORTED_EXTENSIONS = {'.py', '.js', '.tsx', '.jsx', '.ipynb', '.java',
#                             '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h'}
#     IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.git',
#                     '__pycache__', '.next', '.vscode', 'vendor'}
#     files_content = []

#     try:
#         for root, _, files in os.walk(repo_path):
#             if any(ignored_dir in root for ignored_dir in IGNORED_DIRS):
#                 continue

#             for file in files:
#                 file_path = os.path.join(root, file)
#                 if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
#                     file_content = get_file_content(file_path, repo_path)
#                     if file_content:
#                         files_content.append(file_content)

#     except Exception as e:
#         print(f"Error reading repository: {str(e)}")

#     return files_content


# def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
#     model = SentenceTransformer(model_name)
#     return model.encode(text)


# def perform_rag(query, path):
#     raw_query_embedding = get_huggingface_embeddings(query)
#     top_matches = pinecone_index.query(
#         vector=raw_query_embedding.tolist(),
#         top_k=5,
#         include_metadata=True,
#         namespace=path
#     )
#     contexts = [item['metadata']['text'] for item in top_matches['matches']]
#     augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

#     system_prompt = """You are a Senior Software Engineer specializing in TypeScript.
#     Answer any questions I have about the codebase, based on the code provided."""
#     return system_prompt, augmented_query


# def existing_namespaces(pinecone_index):
#     """
#     Get all namespaces in a Pinecone index.

#     Args:
#         pinecone_index: Pinecone index object.

#     Returns:
#         List of all namespaces in the index.
#     """
#     try:
#         index_stats = pinecone_index.describe_index_stats()
#         namespaces = list(index_stats.get("namespaces", {}).keys())
#         return namespaces
#     except Exception as e:
#         print(f"Error retrieving namespaces: {str(e)}")
#         return []

# # --- CHAT FUNCTIONALITY STARTS HERE ---

# # Initialize session state for chats
# if "chats" not in st.session_state:
#     # Initialize chats
#     st.session_state.chats = {"Chat 1": []}
#     st.session_state.active_chat = "Chat 1"


# # Store the initial value of widgets in session state
# if "visibility" not in st.session_state:
#     st.session_state.visibility = "visible"
#     st.session_state.disabled = False

# # Sidebar for repository input and namespace selection
# with st.sidebar:
#     st.title("GitHub Repo...")
#     pasted = st.text_input(
#         "Paste a GitHub link here: 👇",
#         key="placeholder",
#         placeholder="https://github.com/...",
#         label_visibility=st.session_state.visibility,
#         disabled=st.session_state.disabled,
#     )

#     all_namespaces = existing_namespaces(pinecone_index)

#     # Create a selectbox with namespaces as options
#     option = st.selectbox(
#         "Select an indexed repo:",
#         options=all_namespaces if all_namespaces else ["Not available at this time"],  # Fallback if no namespaces
#         index=None,  # Set default index only if namespaces exist
#         placeholder="Select a namespace...",
#     )

#     st.divider()


#     if option or pasted:
#       text_input = pasted if pasted else option
#       with st.sidebar:
#         st.write(f"You selected: {text_input}")
#         st.divider()
#       # Select active chat or start a new one
#       st.subheader("Chat Threads")
#       chat_threads = list(st.session_state.chats.keys())
        
#       for thread in chat_threads:
#           if st.button(thread):
#               st.session_state.active_chat = thread
              

#       if st.button("Add Chat", icon="➕"):
#           new_chat_id = f"Chat {len(st.session_state.chats) + 1}"
#           st.session_state.chats[new_chat_id] = []
#           st.session_state.active_chat = new_chat_id

# # Main chat interface
# st.title("Ask Your Code")
# # st.divider()

# active_chat = st.session_state.active_chat

# if active_chat:
#     st.subheader(f"{active_chat}")
#     messages = st.session_state.chats[active_chat]
#     st.divider()

#     intro = "Welcome! How can I assist you today?"

#     if "messages" not in st.session_state:
#       st.session_state.messages = []
#       st.session_state.messages.append({"role": "assistant", "content": intro})

#     if messages:
#         for message in messages:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])


#     if pasted or option:
#         if pasted:
#           text_input = pasted
#         else:
#           text_input = option
#         if not is_repo_processed(text_input, pinecone_index):
#             with st.spinner("Processing repository..."):
#                 path = clone_repository(text_input)
#                 if path:
#                     # Embedding logic and Pinecone setup
#                     file_content = get_main_files_content(path)
#                     pinecone_setup(text_input, file_content)

#         if prompt:= st.chat_input("What would you like to know?"):
#             st.session_state.messages.append({"role": "user", "content": prompt})
#             with st.chat_message("user"):
#                 st.markdown(prompt)
    
#             with st.chat_message("assistant"):
#                 system_prompt, augmented_query = perform_rag(prompt, text_input)
    
#                 llm_response = client.chat.completions.create(
#                     model="llama-3.1-8b-instant",
#                     messages=[
#                         {"role": "assistant", "content": system_prompt},
#                         {"role": "user", "content": augmented_query}
#                       ],
#                     stream=True,
#                   )
#                 response = st.write_stream(llm_response)
#             st.session_state.messages.append({"role": "assistant", "content": response})
    
#                 # Save chat to Pinecone
#                 # save_chat_to_pinecone(active_chat, st.session_state.chats[active_chat])
#     else:
#         st.warning("Please choose an indexed repo or paste in a url")
# else:
#     st.warning("No active chat selected. Please select a chat or start a new one.")

# # --- CHAT FUNCTIONALITY ENDS HERE ---

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


def perform_rag(query: str, namespace: str):
    # Embed the query (same model used for indexing)
    query_vec = get_huggingface_embeddings(query)

    # Query Pinecone directly (your original style)
    top_matches = pinecone_index.query(
        vector=query_vec.tolist(),
        top_k=5,
        include_metadata=True,
        namespace=namespace,
    )

    matches = top_matches.get("matches", []) or []
    contexts = []
    for item in matches:
        md = item.get("metadata") or {}
        # we stored metadata["text"] during indexing
        if "text" in md:
            contexts.append(md["text"])

    context_block = "\n\n-------\n\n".join(contexts[:10]) if contexts else "No relevant context found."

    augmented_query = (
        "<CONTEXT>\n"
        f"{context_block}\n"
        "-------\n"
        "</CONTEXT>\n\n"
        "MY QUESTION:\n"
        f"{query}"
    )

    system_prompt = (
        "You are a Senior Software Engineer specializing in TypeScript.\n"
        "Answer questions about the codebase using ONLY the provided context.\n"
        "If the context is insufficient, say what you need."
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
