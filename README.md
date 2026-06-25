# Project RADAR: Local Retrieval-Augmented Generation (RAG) System

Project RADAR is a lightweight, serverless, and fully local Retrieval-Augmented Generation (RAG) system. It is designed to ingest and index documentation, knowledge transfer (KT) files, and meeting keynotes, allowing team members to ask natural language questions and receive accurate answers with cited source files.

## Key Features

- **No External API Dependencies**: Runs completely locally for privacy and security.
- **Serverless Vector Database**: Uses **Qdrant** in local filesystem mode (no Docker required).
- **Fast Local Embeddings**: Uses **FastEmbed** with the `BAAI/bge-small-en-v1.5` model for high-speed, local CPU vector encoding.
- **Local Large Language Model**: Utilizes **Ollama** running the `qwen3:4b` model to generate context-grounded responses.
- **Smart Document Chunking**: Implements a custom text splitter that preserves word and sentence boundaries to maintain semantic integrity.
- **Source Citation with Scores**: Displays exact document sources used to construct answers along with similarity relevance percentages.
- **Robust Connection Lock Management**: Automatic lock release immediately after vector search to allow safe concurrent/sequential runs.

---

## Project Structure

```
RAG System RADAR/
├── .venv/                 # Python virtual environment (created on setup)
├── data/
│   ├── documents/         # Drop raw PDFs, DOCX, TXT, or MD files here to ingest
│   └── qdrant_storage/    # Local directory where Qdrant saves vector indexes
├── config.py              # Configuration settings (paths, chunk size, models)
├── ingest.py              # Parsing, chunking, and database indexing script
├── query.py               # Search retrieval and Ollama LLM prompt generation CLI
├── requirements.txt       # Python package dependencies
├── test_rag.py            # Automated end-to-end integration test runner
└── README.md              # Project documentation (this file)
```

---

## Prerequisites

Before setting up the project, make sure you have the following installed on your machine:

1. **Python 3.10+** (verified on Python 3.14)
2. **Ollama** (Download from [ollama.com](https://ollama.com/))
   - Download the required model by running the following command in your terminal:
     ```bash
     ollama pull qwen3:4b
     ```
   - Ensure the Ollama server is running (usually runs in the background automatically).

---

## Setup & Installation

Follow these steps to set up the project on your machine:

### 1. Clone or Open the Directory
Open your terminal (PowerShell, Command Prompt, or Git Bash) and navigate to the project directory:
```bash
cd "RAG System RADAR"
```

### 2. Create a Virtual Environment
Create an isolated Python virtual environment named `.venv`:
```bash
python -m venv .venv
```

### 3. Activate and Install Dependencies
Install all required packages from `requirements.txt`.

- **On Windows (PowerShell/CMD):**
  ```powershell
  .venv\Scripts\pip install -r requirements.txt
  ```
- **On macOS/Linux:**
  ```bash
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

---

## How to Use

### Step 1: Add Your Documentation
Place your source documents inside the `data/documents/` folder. The system supports:
- **PDF files** (`.pdf`)
- **Word files** (`.docx`)
- **Plain Text files** (`.txt`)
- **Markdown files** (`.md`)

### Step 2: Ingest and Index Documents
Run the ingestion script to process the documents, chunk them, convert them to vector embeddings, and save them to the Qdrant database:
```bash
# On Windows
.venv\Scripts\python ingest.py

# On macOS/Linux (if venv is activated)
python ingest.py
```
> [!NOTE]
> On your very first run, the system will automatically download the local embedding model (`BAAI/bge-small-en-v1.5`) from Hugging Face (~100 MB). This is cached locally for all subsequent runs.
>
> To wipe the database and index everything fresh, add the `--reset` flag:
> ```bash
> .venv\Scripts\python ingest.py --reset
> ```

### Step 3: Query the Database
You can query the RAG system in two ways:

#### A. Interactive Loop Mode (Recommended)
Launch the query client without arguments to enter an interactive shell where you can ask consecutive questions:
```bash
.venv\Scripts\python query.py
```
Output:
```
==================================================
Radar RAG CLI System (Qdrant + Qwen3:4b)
Type 'exit' or 'quit' to close.
==================================================
Ask a question > 
```

#### B. Single Question Mode
Ask a direct question by passing it as a command line argument:
```bash
.venv\Scripts\python query.py "What is the token expiration duration?"
```

---

## Running Integration Tests

To verify that the database search and Ollama generation work properly, run the automated integration tests:
```bash
.venv\Scripts\python test_rag.py
```
This script will:
1. Create temporary mock documents inside `data/documents/`.
2. Ingest them into a clean vector collection.
3. Query Qdrant to verify vector matching similarity.
4. Perform an end-to-end question-answering query against Ollama and stream the results.

---

## Troubleshooting

- **Database Lock Errors (`portalocker.exceptions.AlreadyLocked`)**: 
  - Qdrant local storage acts like a single-access file (similar to SQLite). Only one client can connect to it at a time.
  - The codebase has been designed to invoke `client.close()` immediately after vector extraction. If you run into locking issues, check if you have another terminal process running `test_rag.py` or another ingestion script in the background.
- **Ollama Timeout or Connection Issues**:
  - Verify that the Ollama app is running on your system tray or run `ollama list` in a separate command line to check its status.
  - Verify you pulled the correct model tag: `ollama pull qwen3:4b`.
