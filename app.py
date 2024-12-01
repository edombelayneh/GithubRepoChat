
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
import os
import tempfile
from github import Github, Repository
from git import Repo
from openai import OpenAI
from pathlib import Path
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

# Functions and other initializations
api_key = os.environ["PINECONE_API_KEY"]
api_key_groq = os.environ["GROQ_API_KEY"]

# Initialize Groq
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key_groq
)

# Initialize Pinecone
pc = Pinecone(api_key=api_key)
pinecone_index = pc.Index("codebase-rag")


def pinecone_setup(text_input, file_content):
    vectorstore = PineconeVectorStore(index_name="codebase-rag", embedding=HuggingFaceEmbeddings())

    # Insert the codebase embeddings into Pinecone
    documents = []
    for file in file_content:
        doc = Document(
            page_content=f"{file['name']}\n{file['content']}",
            metadata={"source": file['name']}
        )
        documents.append(doc)

    vectorstore = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=HuggingFaceEmbeddings(),
        index_name="codebase-rag",
        namespace=text_input
    )


def is_repo_processed(repo_url, pinecone_index):
    """Checks if a repository has already been processed and indexed in Pinecone.

    Args:
        repo_url: The URL of the GitHub repository.
        pinecone_index: Pinecone index object.

    Returns:
        True if the repository has been processed, False otherwise.
    """
    namespace = repo_url
    try:
        response = pinecone_index.describe_index_stats()
        if namespace in response.get('namespaces', {}):
            return True
    except Exception as e:
        print(f"Error checking Pinecone index: {str(e)}")
    return False


def clone_repository(repo_url):
    """Clones a GitHub repository if it hasn't been cloned already.

    Args:
        repo_url: The URL of the GitHub repository.

    Returns:
        The path to the cloned repository or a message if already cloned.
    """
    repo_name = repo_url.split("/")[-1]
    repo_path = f"{repo_name}"

    if os.path.exists(repo_path):
        st.write(f"Repository already exists at {repo_path}")
        return repo_path

    try:
        Repo.clone_from(repo_url, repo_path)
        st.write(f"Repository cloned at {repo_path}")
        return repo_path
    except Exception as e:
        st.write(f"Error cloning repository: {str(e)}")
        return None


def get_file_content(file_path, repo_path):
    """
    Get content of a single file.

    Args:
        file_path (str): Path to the file

    Returns:
        Optional[Dict[str, str]]: Dictionary with file name and content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        rel_path = os.path.relpath(file_path, repo_path)
        return {
            "name": rel_path,
            "content": content
        }
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None


def get_main_files_content(repo_path: str):
    """
    Get content of supported code files from the local repository.

    Args:
        repo_path: Path to the local repository

    Returns:
        List of dictionaries containing file names and contents
    """
    SUPPORTED_EXTENSIONS = {'.py', '.js', '.tsx', '.jsx', '.ipynb', '.java',
                            '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h'}
    IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.git',
                    '__pycache__', '.next', '.vscode', 'vendor'}
    files_content = []

    try:
        for root, _, files in os.walk(repo_path):
            if any(ignored_dir in root for ignored_dir in IGNORED_DIRS):
                continue

            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                    file_content = get_file_content(file_path, repo_path)
                    if file_content:
                        files_content.append(file_content)

    except Exception as e:
        print(f"Error reading repository: {str(e)}")

    return files_content


def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)


def perform_rag(query, path):
    raw_query_embedding = get_huggingface_embeddings(query)
    top_matches = pinecone_index.query(
        vector=raw_query_embedding.tolist(),
        top_k=5,
        include_metadata=True,
        namespace=path
    )
    contexts = [item['metadata']['text'] for item in top_matches['matches']]
    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

    system_prompt = """You are a Senior Software Engineer specializing in TypeScript.
    Answer any questions I have about the codebase, based on the code provided."""
    return system_prompt, augmented_query


def existing_namespaces(pinecone_index):
    """
    Get all namespaces in a Pinecone index.

    Args:
        pinecone_index: Pinecone index object.

    Returns:
        List of all namespaces in the index.
    """
    try:
        index_stats = pinecone_index.describe_index_stats()
        namespaces = list(index_stats.get("namespaces", {}).keys())
        return namespaces
    except Exception as e:
        print(f"Error retrieving namespaces: {str(e)}")
        return []


def save_chat_to_pinecone(chat_id, chat_history):
    """
    Save chat history to Pinecone.

    Args:
        chat_id (str): Unique identifier for the chat (e.g., chat name).
        chat_history (list): List of messages with roles and content.
    """
    vectorstore = PineconeVectorStore(index_name="msg-history", embedding=HuggingFaceEmbeddings())
    documents = [
        Document(page_content=msg["content"], metadata={
            "role": msg["role"],
            "chat_id": chat_id,
            "text": msg["content"]
        })
        for msg in chat_history
    ]
    vectorstore = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=HuggingFaceEmbeddings(),
        index_name="msg-history",
        namespace=chat_id
    )


def load_chats_from_pinecone(chat_id):
    """
    Load chat history from Pinecone for the specified chat namespace.

    Args:
        chat_id (str): The namespace or unique identifier for the chat.

    Returns:
        List of chat messages or an empty list if no chat is found.
    """
    try:
        # Fetch all vectors in the namespace (chat_id)
        response = pinecone_index.fetch(ids=[], namespace=chat_id)
        return [
            {"role": metadata["role"], "content": vector["page_content"]}
            for vector_id, vector in response.get("vectors", {}).items()
            for metadata in vector.get("metadata", [])
        ]
    except Exception as e:
        st.error(f"Error loading chat from Pinecone: {str(e)}")
        return []





# --- CHAT FUNCTIONALITY STARTS HERE ---

# Initialize session state for chats
if "chats" not in st.session_state:
    # Initialize chats
    st.session_state.chats = {"New Chat": []}
    st.session_state.active_chat = "New Chat"

    # Load existing chats from Pinecone
    for namespace in existing_namespaces(pinecone_index):
        chat_history = load_chats_from_pinecone(namespace)
        if chat_history:
            st.session_state.chats[namespace] = chat_history
        else:
            st.warning(f"No chat history found for namespace: {namespace}")

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

# Sidebar for repository input and namespace selection
with st.sidebar:
    st.title("GitHub Repo...")
    pasted = st.text_input(
        "Paste a GitHub link here: 👇",
        key="placeholder",
        placeholder="https://github.com/...",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
    )

    all_namespaces = existing_namespaces(pinecone_index)

    # Create a selectbox with namespaces as options
    option = st.selectbox(
        "Select a namespace:",
        options=all_namespaces if all_namespaces else ["No namespaces available"],  # Fallback if no namespaces
        index=None,  # Set default index only if namespaces exist
        placeholder="Select a namespace...",
    )

    st.divider()

    # Select active chat or start a new one
    st.subheader("Chat Threads")
    chat_threads = list(st.session_state.chats.keys())

    if option or pasted:
      text_input = pasted if pasted else option
      with st.sidebar:
        st.write(f"You selected: {text_input}")
        st.divider()
      for thread in chat_threads:
          if st.button(thread):
              st.session_state.active_chat = thread

      if st.button("Add Chat", icon="➕"):
          new_chat_id = f"Chat {len(st.session_state.chats) + 1}"
          st.session_state.chats[new_chat_id] = []
          st.session_state.active_chat = new_chat_id

# Main chat interface
st.title("Ask Your Code")
st.divider()

active_chat = st.session_state.active_chat

if active_chat:
    st.subheader(f"Active Chat: {active_chat}")
    messages = st.session_state.chats[active_chat]

    intro = "Welcome! How can I assist you today?"

    if "messages" not in st.session_state:
      st.session_state.messages = []
      st.session_state.messages.append({"role": "assistant", "content": intro})

    if messages:
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


    if pasted or option:
        if pasted:
          text_input = pasted
        else:
          text_input = option
        if not is_repo_processed(text_input, pinecone_index):
          st.write("Repository already processed in Pinecone!")
        else:
            path = clone_repository(text_input)
            if path:
                # Continue with processing the cloned repository
                st.write("Processing repository...")
                st.write(f"Here is the text_input: {text_input}\nHere is the path: {path}")

                # Embedding logic and Pinecone setup
                file_content = get_main_files_content(path)
                pinecone_setup(text_input, file_content)

        if prompt:= st.chat_input("What would you like to know?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                system_prompt, augmented_query = perform_rag(prompt, text_input)

                llm_response = client.chat.completions.create(
                  model="llama-3.1-8b-instant",
                  messages=[
                      {"role": "assistant", "content": system_prompt},
                      {"role": "user", "content": augmented_query}
                  ],
                  stream=True,
              )
                response = st.write_stream(llm_response)
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Save chat to Pinecone
            save_chat_to_pinecone(active_chat, st.session_state.chats[active_chat])

    else:
      st.warning("Please select a namespace or paste a GitHub link.")
else:
    st.warning("No active chat selected. Please select a chat or start a new one.")

# --- CHAT FUNCTIONALITY ENDS HERE ---
