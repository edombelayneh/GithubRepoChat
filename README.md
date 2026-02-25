# GitHub Repository Analysis with Pinecone and LangChain – RepoChat
<img src="https://img.shields.io/badge/-Solo Project-f2336f?&style=for-the-badge&logoColor=white" />

## Objective

RepoChat is an AI-powered repository intelligence tool designed to help Product Managers and Engineers quickly understand unfamiliar codebases.

Instead of manually navigating dozens of files, users can:

* Ask structured questions about a repository
* View evidence-backed responses
* Explore architectural and product insights
* Identify risks, flows, and system boundaries

RepoChat leverages vector search (Pinecone), semantic embeddings (HuggingFace), and LLM-based reasoning (Groq/OpenAI-compatible models) to transform raw source code into structured, contextual answers.

This project blends **technical implementation** with **product thinking**, enabling faster onboarding, due diligence, and feature discovery.

**To view the app, <a href="https://edom-repochat.streamlit.app/"> Click-here</a>.** 

# Product Vision

Modern teams frequently face these challenges:

* PMs inheriting legacy systems with little documentation
* Engineers onboarding into large repositories
* Investors or technical stakeholders performing due diligence
* Cross-functional teams needing shared understanding

RepoChat acts as a **codebase co-pilot**, translating repository structure into:

* System-level understanding
* User journey mapping
* Feature inventory
* Risk analysis
* Architectural traceability

It reduces the time-to-understanding from hours (or days) to minutes.

# Target Users

### Primary Users

* Product Managers
* Technical Program Managers
* Startup Founders
* Engineering Managers

### Secondary Users

* Software Engineers
* Technical Recruiters
* Investors conducting technical evaluation

# Core Value Proposition

RepoChat enables users to:

* Understand what a system does without reading every file
* Switch between PM mode and Engineer mode
* Trace features to file-level evidence
* Identify technical risks and unknowns
* Generate structured executive summaries

It bridges the gap between **code implementation** and **business understanding**.

# Key Features

## 1. Dual Perspective Modes

### PM Mode

Transforms technical details into product-level insight using a structured format:

* What it is
* User / business flows
* Key components
* Risks / unknowns
* What to validate next

### Engineer Mode

Focuses on technical accuracy and file-level traceability:

* Direct answer
* Relevant file paths
* Step-by-step logic
* Edge cases
* Related modules


## 2. GitHub Repository Cloning

* Clones public repositories locally
* Supports optional branch selection
* Prevents redundant re-processing

## 3. Semantic Embedding + Vector Search

* Supported files are embedded using HuggingFace sentence transformers
* Stored in Pinecone under unique namespaces
* Enables fast similarity search across large codebases

## 4. Evidence-Backed Retrieval (RAG)

* Retrieves top-matching files
* Injects structured context into LLM prompt
* Provides ranked sources with similarity scores
* Displays confidence level (High / Medium / Low)

This ensures answers are grounded in actual repository content.

## 5. Insights Tab (Heuristic Repo Intelligence)

Without heavy parsing, RepoChat surfaces:

* File count
* Top folders
* Language distribution
* Likely entry points
* Dependency files
* Largest files (hotspots)

This acts as a lightweight architecture overview.

## 6. Chat Thread Management

* Multiple concurrent chat threads
* Mode-aware prompt rewriting
* Thread-based context retention


# Workflow

### 1. Select or Paste a Repository

Input a public GitHub repository URL.

### 2. Clone & Index

* Files are scanned
* Supported extensions are extracted
* Content is embedded
* Stored in Pinecone namespace

### 3. Ask Structured Questions

Choose PM or Engineer mode and ask:

* “Map the user journey.”
* “Where is authentication handled?”
* “Identify tech debt risks.”

### 4. Review Evidence

Check:

* Ranked source files
* Confidence score
* Related modules

# Technical Architecture

### Frontend

* Streamlit (interactive web interface)

### Backend Logic

* Python
* GitPython (repo cloning)
* Sentence Transformers (embeddings)
* Pinecone (vector storage)
* Groq-compatible OpenAI client (LLM inference)

### Retrieval-Augmented Generation (RAG)

1. Query → embedding
2. Pinecone similarity search
3. Context injection
4. LLM structured response
5. Confidence scoring

# Skills Demonstrated

### Technical Skills

* End-to-end RAG implementation
* Vector database integration
* Namespace management in Pinecone
* Context window optimization
* LLM prompt engineering
* Streamlit app state management
* Heuristic repository analysis

### Product & PM Skills

* Mode-based persona design (PM vs Engineer)
* Structured response frameworks
* Risk identification logic
* Confidence communication (UX clarity)
* Feature prioritization thinking
* User segmentation
* Information hierarchy design
* Guided prompt workflows

# Screenshots

Here is what you see when you first load the web app:
<img width="1436" height="774" alt="Screenshot 2026-02-24 at 7 21 13 PM" src="https://github.com/user-attachments/assets/bd5652ba-32e2-40e9-92b3-b98100d7a2e4" />


# Demo
## PM Version

https://github.com/user-attachments/assets/63d018b6-c02e-4eca-af6d-349e04c8b453

## Engineer Version

https://github.com/user-attachments/assets/812fc65c-9975-4384-9561-d405fcd61703


# Future Enhancements

### Product Enhancements

* Private repo authentication (OAuth integration)
* Repo comparison mode (diff two systems)
* Architecture diagram generation
* Feature coverage mapping
* Technical due diligence report export (PDF)

### Technical Enhancements

* Smarter chunking strategies
* AST-based code parsing
* Hybrid search (keyword + semantic)
* Streaming source highlighting
* Multi-repo memory graph

# Why This Project Matters

RepoChat demonstrates more than technical integration.

It shows the ability to:

* Translate complex systems into structured understanding
* Design for multiple personas
* Combine product thinking with AI architecture
* Build tools that reduce cognitive load for technical teams
