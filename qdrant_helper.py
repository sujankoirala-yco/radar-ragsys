"""
Qdrant client factory — returns a cloud or local client based on config.QDRANT_MODE.
"""
from qdrant_client import QdrantClient
import config


def get_qdrant_client() -> QdrantClient:
    """
    Creates and returns a QdrantClient instance.

    - If QDRANT_MODE == "cloud", connects to Qdrant Cloud using URL + API key.
    - If QDRANT_MODE == "local", uses a local file-based storage directory.

    The FastEmbed model is set on the client automatically.
    """
    if config.QDRANT_MODE == "cloud":
        print(f"Connecting to Qdrant Cloud: {config.QDRANT_URL}")
        client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            timeout=60,  # generous timeout for cloud requests
        )
    else:
        print(f"Using local Qdrant storage: {config.QDRANT_DIR}")
        client = QdrantClient(path=str(config.QDRANT_DIR))

    # Set FastEmbed model for automatic embedding
    client.set_model(config.EMBED_MODEL)
    return client
