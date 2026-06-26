import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "documents"
QDRANT_DIR = DATA_DIR / "qdrant_storage"

# Ensure directories exist
DOCS_DIR.mkdir(parents=True, exist_ok=True)
QDRANT_DIR.mkdir(parents=True, exist_ok=True)

# Qdrant configurations
QDRANT_COLLECTION = "radar_knowledge_base"
# Default embedding model for FastEmbed (BGE-small-en-v1.5)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Qdrant Cloud configurations (loaded from .env)
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
# "cloud" uses Qdrant Cloud; "local" uses local file storage (for dev)
QDRANT_MODE = "cloud" if QDRANT_URL and QDRANT_API_KEY else "local"

# Groq LLM configurations
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

# Legacy Ollama configurations (kept for reference)
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:4b"

# Chunking configurations
CHUNK_SIZE = 600      # length in characters
CHUNK_OVERLAP = 80    # overlap in characters
