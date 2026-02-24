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
from langchain_core.documents import Document

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
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(INDEX_NAME)

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
    except Exception:
        return []


def is_repo_processed(namespace: str, index) -> bool:
    try:
        stats = index.describe_index_stats()
        return namespace in stats.get("namespaces", {})
    except Exception:
        return False


def repo_stats(namespace: str, index) -> dict:
    """
    Best-effort namespace stats for sidebar status.
    Pinecone describe_index_stats() format varies by client versions.
    """
    try:
        stats = index.describe_index_stats()
        ns = (stats.get("namespaces") or {}).get(namespace) or {}
        # Common field name is "vector_count"
        return {
            "vector_count": ns.get("vector_count"),
        }
    except Exception:
        return {"vector_count": None}


def clone_repository(repo_url: str, branch: str | None = None) -> str | None:
    repo_name = repo_url.rstrip("/").split("/")[-1]
    repo_path = Path(repo_name)

    if repo_path.exists():
        return str(repo_path)

    try:
        if branch:
            Repo.clone_from(repo_url, str(repo_path), branch=branch, depth=1)
        else:
            Repo.clone_from(repo_url, str(repo_path), depth=1)
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
    except Exception:
        return None


def get_main_files_content(repo_path: str):
    files_content = []
    for root, _, files in os.walk(repo_path):
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

    return files_content


def pinecone_setup(namespace: str, file_content: list[dict]):
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=hf_embeddings,
        namespace=namespace,
    )

    documents: list[Document] = []
    for f in file_content:
        doc_text = f"{f['name']}\n{f['content']}"
        documents.append(
            Document(
                page_content=doc_text,
                metadata={"source": f["name"], "text": doc_text},
            )
        )

    PineconeVectorStore.from_documents(
        documents=documents,
        embedding=hf_embeddings,
        index_name=INDEX_NAME,
        namespace=namespace,
    )

    return vectorstore


def _confidence_from_scores(scores: list[float]) -> str:
    """
    Pinecone 'score' semantics depend on index type (cosine similarity vs distance).
    For typical similarity scores: closer to 1 is better.
    We'll use simple thresholds that work "good enough" for demo.
    """
    if not scores:
        return "Low"
    top = scores[0]
    if top >= 0.83:
        return "High"
    if top >= 0.73:
        return "Medium"
    return "Low"


def _mode_system_prompt(mode: str, response_style: str, prefer_uncertain: bool, show_citations: bool) -> str:
    common = (
        "You must ONLY use the provided <CONTEXT>.\n"
        "If the context is insufficient, say what you would need and do not guess.\n"
        "Be crisp, structured, and practical.\n"
    )
    if prefer_uncertain:
        common += "Prefer saying you are not sure over making assumptions.\n"

    # style
    style_line = {
        "Concise": "Keep answers short and to the point.",
        "Standard": "Balance brevity with clarity.",
        "Detailed": "Be thorough, include steps and edge cases.",
    }.get(response_style, "Balance brevity with clarity.")

    if mode == "PM":
        template = (
            "You are an AI Product Manager working with engineers on a codebase.\n"
            "Translate technical details into product + system understanding.\n"
            "Avoid implementation trivia unless necessary.\n"
            "Use this format:\n"
            "1) What it is (1-2 sentences)\n"
            "2) User / business flows (bullets)\n"
            "3) Key components in business terms (bullets)\n"
            "4) Risks / unknowns (bullets)\n"
            "5) What I’d validate next (bullets)\n"
        )
        if show_citations:
            template += "If you reference evidence, add a short 'Evidence' list with file paths.\n"
        return f"{template}\n{common}\n{style_line}\n"

    # Engineer
    template = (
        "You are a Senior Software Engineer.\n"
        "Be code-accurate and cite relevant file paths.\n"
        "Use this format:\n"
        "1) Direct answer\n"
        "2) Where to look (file paths)\n"
        "3) How it works (steps)\n"
        "4) Edge cases / gotchas\n"
        "5) Related files\n"
    )
    if show_citations:
        template += "When possible, include file paths that support each claim.\n"
    return f"{template}\n{common}\n{style_line}\n"


def perform_rag(
    query: str,
    namespace: str,
    mode: str,
    response_style: str,
    prefer_uncertain: bool,
    show_citations: bool,
    max_context_chars: int = 18000,
):
    query_vec = get_huggingface_embeddings(query)

    top_matches = pinecone_index.query(
        vector=query_vec.tolist(),
        top_k=8,
        include_metadata=True,
        namespace=namespace,
    )

    matches = top_matches.get("matches", []) or []

    # Collect contexts (and sources/scores)
    contexts = []
    sources = []
    scores = []

    for item in matches:
        md = item.get("metadata") or {}
        txt = md.get("text")
        src = md.get("source") or "unknown"
        score = item.get("score")
        if score is not None:
            scores.append(float(score))
        if txt:
            contexts.append(txt)
            sources.append(src)

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

    system_prompt = _mode_system_prompt(
        mode=mode,
        response_style=response_style,
        prefer_uncertain=prefer_uncertain,
        show_citations=show_citations,
    )

    confidence = _confidence_from_scores(scores)
    # For the Sources tab: keep a ranked list (best-effort)
    ranked = []
    for i, item in enumerate(matches):
        md = item.get("metadata") or {}
        ranked.append(
            {
                "rank": i + 1,
                "score": float(item.get("score") or 0.0),
                "source": md.get("source") or "unknown",
                "preview": ((md.get("text") or "")[:220] + "…") if (md.get("text") or "") else "",
            }
        )

    return system_prompt, augmented_query, confidence, ranked


def compute_local_insights(file_content: list[dict]) -> dict:
    """
    Cheap heuristics from scanned files (no heavy parsing).
    """
    lang_counts = {}
    top_folders = {}
    largest_files = []

    entry_candidates = []
    dep_candidates = []

    for f in file_content:
        name = f["name"]
        content = f["content"] or ""
        ext = os.path.splitext(name)[1].lower()

        lang_counts[ext] = lang_counts.get(ext, 0) + 1

        parts = name.split(os.sep)
        if len(parts) > 1:
            top = parts[0]
            top_folders[top] = top_folders.get(top, 0) + 1

        largest_files.append((name, len(content)))

        # entrypoint heuristics
        lowered = name.lower()
        if lowered.endswith(("main.py", "app.py", "server.py", "index.js", "index.ts", "index.tsx", "main.ts", "main.go")):
            entry_candidates.append(name)
        if lowered.endswith(("package.json", "pyproject.toml", "requirements.txt", "pom.xml", "go.mod", "cargo.toml")):
            dep_candidates.append(name)

    largest_files.sort(key=lambda x: x[1], reverse=True)
    top_folders_sorted = sorted(top_folders.items(), key=lambda x: x[1], reverse=True)

    return {
        "languages": sorted(lang_counts.items(), key=lambda x: x[1], reverse=True),
        "top_folders": top_folders_sorted[:8],
        "largest_files": largest_files[:10],
        "entrypoints": entry_candidates[:10],
        "dependency_files": dep_candidates[:10],
        "file_count": len(file_content),
    }


# --- UI / APP ---
st.set_page_config(page_title="RepoChat", layout="wide")
st.title("RepoChat")

# ---------- SESSION STATE ----------
if "chats" not in st.session_state:
    st.session_state.chats = {"Chat 1": []}
    st.session_state.active_chat = "Chat 1"

if "mode" not in st.session_state:
    st.session_state.mode = "PM"  # "PM" or "Engineer"

if "response_style" not in st.session_state:
    st.session_state.response_style = "Standard"

if "show_citations" not in st.session_state:
    st.session_state.show_citations = True

if "prefer_uncertain" not in st.session_state:
    st.session_state.prefer_uncertain = True

if "repo_branch" not in st.session_state:
    st.session_state.repo_branch = ""

if "active_namespace" not in st.session_state:
    st.session_state.active_namespace = None

if "local_repo_path" not in st.session_state:
    st.session_state.local_repo_path = None

if "local_file_content" not in st.session_state:
    st.session_state.local_file_content = None

if "local_insights" not in st.session_state:
    st.session_state.local_insights = None

if "last_ranked_sources" not in st.session_state:
    st.session_state.last_ranked_sources = {}  # by chat id

if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = {}  # by chat id


def ensure_welcome_message(chat_id: str):
    if not st.session_state.chats[chat_id]:
        st.session_state.chats[chat_id].append(
            {"role": "assistant", "content": "Welcome — pick a repo, choose PM/Engineer mode, and ask away."}
        )


def set_mode_defaults():
    # Mode-aware defaults (small but important)
    if st.session_state.mode == "Engineer":
        st.session_state.show_citations = True
    # PM mode can still show citations, but it’s optional


def handle_user_prompt(prompt_text: str, namespace: str):
    chat_id = st.session_state.active_chat
    st.session_state.chats[chat_id].append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    with st.chat_message("assistant"):
        system_prompt, augmented_query, confidence, ranked = perform_rag(
            query=prompt_text,
            namespace=namespace,
            mode=st.session_state.mode,
            response_style=st.session_state.response_style,
            prefer_uncertain=st.session_state.prefer_uncertain,
            show_citations=st.session_state.show_citations,
        )

        llm_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query},
            ],
            stream=True,
        )

        response_text = st.write_stream(llm_response)

        # Confidence line (simple UX win)
        st.caption(f"Confidence: {confidence}")

    st.session_state.chats[chat_id].append({"role": "assistant", "content": response_text})
    st.session_state.last_ranked_sources[chat_id] = ranked
    st.session_state.last_confidence[chat_id] = confidence


# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("Repo setup")

    pasted = st.text_input(
        "GitHub repo URL (public)",
        placeholder="https://github.com/owner/repo",
    )

    st.session_state.repo_branch = st.text_input(
        "Branch (optional)",
        value=st.session_state.repo_branch,
        placeholder="main",
    )

    all_namespaces = existing_namespaces(pinecone_index)
    option = st.selectbox(
        "Or select an indexed repo",
        options=all_namespaces if all_namespaces else ["Not available at this time"],
        index=None if all_namespaces else 0,
        placeholder="Select a namespace...",
    )

    # Resolve active namespace
    text_input = None
    if pasted:
        text_input = pasted.strip()
    elif option and option != "Not available at this time":
        text_input = option

    st.divider()

    # Mode switch
    st.header("Mode")
    st.session_state.mode = st.radio(
        "Choose a perspective",
        options=["PM", "Engineer"],
        index=0 if st.session_state.mode == "PM" else 1,
        horizontal=True,
    )
    set_mode_defaults()

    st.caption("PM = flows, risks, feature mapping. Engineer = architecture, file references, implementation.")

    st.divider()

    st.header("Output")
    st.session_state.response_style = st.selectbox(
        "Response style",
        options=["Concise", "Standard", "Detailed"],
        index=["Concise", "Standard", "Detailed"].index(st.session_state.response_style),
    )
    st.session_state.show_citations = st.toggle(
        "Show file references",
        value=st.session_state.show_citations,
    )
    st.session_state.prefer_uncertain = st.toggle(
        "Prefer “not sure” over guessing",
        value=st.session_state.prefer_uncertain,
    )

    st.divider()

    # Threads
    st.header("Chat threads")
    for thread in list(st.session_state.chats.keys()):
        if st.button(thread, use_container_width=True):
            st.session_state.active_chat = thread
            ensure_welcome_message(thread)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Add chat", use_container_width=True):
            new_chat_id = f"Chat {len(st.session_state.chats) + 1}"
            st.session_state.chats[new_chat_id] = []
            st.session_state.active_chat = new_chat_id
            ensure_welcome_message(new_chat_id)

    with col_b:
        if st.button("Rewrite last", use_container_width=True):
            # “Rewrite last answer in this mode”
            chat_id = st.session_state.active_chat
            # Find last user message to re-ask with mode shift
            last_user = None
            for m in reversed(st.session_state.chats[chat_id]):
                if m["role"] == "user":
                    last_user = m["content"]
                    break
            if last_user and st.session_state.active_namespace:
                handle_user_prompt(f"Rewrite your answer in {st.session_state.mode} mode: {last_user}", st.session_state.active_namespace)

    st.divider()

    # Repo status + actions
    st.header("Repo status")
    if text_input:
        st.session_state.active_namespace = text_input
        processed = is_repo_processed(text_input, pinecone_index)
        ns_stats = repo_stats(text_input, pinecone_index)

        st.write(f"**Selected:** {text_input}")
        st.write(f"**Indexed:** {'Yes' if processed else 'No'}")
        if ns_stats.get("vector_count") is not None:
            st.write(f"**Vectors:** {ns_stats['vector_count']}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Index / Re-index", use_container_width=True):
                # always clone locally for indexing
                with st.spinner("Cloning + indexing..."):
                    repo_path = clone_repository(text_input, st.session_state.repo_branch or None)
                    if not repo_path:
                        st.stop()
                    file_content = get_main_files_content(repo_path)
                    pinecone_setup(text_input, file_content)
                    st.session_state.local_repo_path = repo_path
                    st.session_state.local_file_content = file_content
                    st.session_state.local_insights = compute_local_insights(file_content)
                    st.success("Indexed.")

        with col2:
            if st.button("Load locally (Insights)", use_container_width=True):
                with st.spinner("Cloning repo for Insights..."):
                    repo_path = clone_repository(text_input, st.session_state.repo_branch or None)
                    if not repo_path:
                        st.stop()
                    file_content = get_main_files_content(repo_path)
                    st.session_state.local_repo_path = repo_path
                    st.session_state.local_file_content = file_content
                    st.session_state.local_insights = compute_local_insights(file_content)
                    st.success("Loaded locally.")

        with st.expander("Repo context"):
            ins = st.session_state.local_insights
            if not ins:
                st.info("Load locally to see Insights context (no fancy UI — fast heuristics).")
            else:
                st.write(f"**Supported files scanned:** {ins['file_count']}")
                st.write("**Top languages (by extension):**")
                st.write(", ".join([f"{k} ({v})" for k, v in ins["languages"][:8]]) or "N/A")
                st.write("**Top folders:**")
                st.write(", ".join([f"{k} ({v})" for k, v in ins["top_folders"]]) or "N/A")
                st.write("**Likely entry points:**")
                st.write("\n".join([f"- {p}" for p in ins["entrypoints"]]) or "N/A")
                st.write("**Dependency files:**")
                st.write("\n".join([f"- {p}" for p in ins["dependency_files"]]) or "N/A")
    else:
        st.info("Paste a repo URL or pick an indexed namespace.")


# ---------- MAIN AREA ----------
active_chat = st.session_state.active_chat
ensure_welcome_message(active_chat)

# Top header row (simple, Streamlit-friendly)
left, right = st.columns([3, 1])
with left:
    if st.session_state.active_namespace:
        st.subheader(f"{active_chat} • {st.session_state.active_namespace}")
    else:
        st.subheader(active_chat)
with right:
    st.write(f"**Mode:** {st.session_state.mode}")

# If no namespace, stop
namespace = st.session_state.active_namespace
if not namespace:
    st.warning("Choose an indexed repo or paste a URL to start.")
    st.stop()

# If not indexed, index automatically (your original behavior)
if not is_repo_processed(namespace, pinecone_index):
    with st.spinner("Processing repository (first time indexing)..."):
        repo_path = clone_repository(namespace, st.session_state.repo_branch or None)
        if repo_path:
            file_content = get_main_files_content(repo_path)
            pinecone_setup(namespace, file_content)
            st.session_state.local_repo_path = repo_path
            st.session_state.local_file_content = file_content
            st.session_state.local_insights = compute_local_insights(file_content)
        else:
            st.stop()

tabs = st.tabs(["Chat", "Insights", "Sources"])

# ---------- TAB: CHAT ----------
with tabs[0]:
    # Starter prompts (button-driven, mode-aware)
    st.caption("Starter prompts")
    if st.session_state.mode == "PM":
        starters = [
            "Explain what this system does in plain English.",
            "Map the user journey end-to-end based on what’s in the repo.",
            "Identify the highest-risk areas and why.",
            "List key features implemented vs missing (based on evidence).",
            "What questions should a PM ask the team about this repo?",
        ]
    else:
        starters = [
            "Give me the architecture overview.",
            "Where is authentication handled? Cite file paths.",
            "Trace the request flow for the main API entry point.",
            "How is database access structured?",
            "List key modules and their responsibilities.",
        ]

    c1, c2, c3, c4, c5 = st.columns(5)
    cols = [c1, c2, c3, c4, c5]
    for i, s in enumerate(starters[:5]):
        if cols[i].button(f"Prompt {i+1}", use_container_width=True):
            handle_user_prompt(s, namespace)

    st.divider()

    # Render messages
    for message in st.session_state.chats[active_chat]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about the repo…")
    if prompt:
        handle_user_prompt(prompt, namespace)


# ---------- TAB: INSIGHTS ----------
with tabs[1]:
    ins = st.session_state.local_insights
    if not ins:
        st.info("Click “Load locally (Insights)” in the sidebar to populate this tab.")
    else:
        # Keep it simple: expanders instead of complex visuals
        with st.expander("Executive summary (quick heuristic)", expanded=True):
            st.write(
                "This section is meant to be lightweight. Ask in Chat for an AI-generated summary; "
                "here we show what we can infer fast from repository structure."
            )
            st.write(f"- Scanned supported files: **{ins['file_count']}**")
            st.write(f"- Dependency files found: **{len(ins['dependency_files'])}**")
            st.write(f"- Likely entry points found: **{len(ins['entrypoints'])}**")

        with st.expander("Repo map", expanded=True):
            st.write("**Top folders**")
            st.write("\n".join([f"- {k} ({v} files)" for k, v in ins["top_folders"]]) or "N/A")
            st.write("\n**Top languages (by extension)**")
            st.write("\n".join([f"- {k}: {v}" for k, v in ins["languages"][:12]]) or "N/A")

        with st.expander("Entry points & dependencies"):
            st.write("**Likely entry points**")
            st.write("\n".join([f"- {p}" for p in ins["entrypoints"]]) or "N/A")
            st.write("\n**Dependency files**")
            st.write("\n".join([f"- {p}" for p in ins["dependency_files"]]) or "N/A")

        with st.expander("Hotspots (largest files)"):
            st.write("\n".join([f"- {name} ({size} chars)" for name, size in ins["largest_files"]]) or "N/A")

        # Mode-aware “guided” buttons
        st.divider()
        st.caption("Guided actions")
        if st.session_state.mode == "PM":
            a, b, c, d, e = st.columns(5)
            if a.button("Executive summary", use_container_width=True):
                handle_user_prompt("Generate an executive summary of this repo for a PM (include flows, risks, unknowns).", namespace)
            if b.button("Feature map", use_container_width=True):
                handle_user_prompt("Create a feature map: capabilities -> evidence (file paths).", namespace)
            if c.button("Risks & tech debt", use_container_width=True):
                handle_user_prompt("Identify risks / tech debt areas with evidence from file paths.", namespace)
            if d.button("Questions to ask", use_container_width=True):
                handle_user_prompt("List the top questions a PM should ask the team about this codebase.", namespace)
            if e.button("User journey", use_container_width=True):
                handle_user_prompt("Infer the end-to-end user journey(s) supported by this repo, with evidence.", namespace)
        else:
            a, b, c, d, e = st.columns(5)
            if a.button("Architecture", use_container_width=True):
                handle_user_prompt("Give an architecture overview with file paths and responsibilities.", namespace)
            if b.button("Trace a flow", use_container_width=True):
                handle_user_prompt("Trace a likely request flow from entry point to handlers; cite file paths.", namespace)
            if c.button("Where is X?", use_container_width=True):
                handle_user_prompt("Where is the primary business logic implemented? Cite file paths.", namespace)
            if d.button("Test plan", use_container_width=True):
                handle_user_prompt("Generate a practical test plan for the most critical modules; cite file paths.", namespace)
            if e.button("Related files", use_container_width=True):
                handle_user_prompt("List related files for the most important module(s) and explain coupling.", namespace)


# ---------- TAB: SOURCES ----------
with tabs[2]:
    chat_id = st.session_state.active_chat
    ranked = st.session_state.last_ranked_sources.get(chat_id)

    if not ranked:
        st.info("Ask a question in Chat to populate retrieved Sources.")
    else:
        st.write(f"**Last answer confidence:** {st.session_state.last_confidence.get(chat_id, 'N/A')}")
        # Streamlit table from list of dicts
        st.dataframe(
            ranked,
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("How to interpret this"):
            st.write(
                "- Higher scores generally mean closer semantic match.\n"
                "- Use the file paths to navigate quickly.\n"
                "- If the relevant file isn’t here, ask a more specific question or re-index the repo."
            )