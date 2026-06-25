import os
from pathlib import Path
from qdrant_client import QdrantClient

# Import system modules
import config
import ingest
import query

def setup_test_documents():
    """Creates mock documentation files in the data/documents directory."""
    print("Setting up mock test documents...")
    docs_dir = Path(config.DOCS_DIR)
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create a KT document about authentication
    kt_auth_content = (
        "Knowledge Transfer: authentication and Authorization in Radar Project\n"
        "=======================================================================\n"
        "We implement authentication using OAuth2 Protocol. Our token format is JWT (JSON Web Tokens).\n"
        "The authentication service is called 'RadarAuthService' and runs on port 8081.\n"
        "JWT tokens expire exactly 2 hours (7200 seconds) after generation.\n"
        "To rotate keys, the Security Team coordinates a rotation every 90 days. The environment variable\n"
        "is JWT_SIGNING_KEY which is fetched from AWS Secrets Manager at startup.\n"
    )
    with open(docs_dir / "kt_authentication.txt", "w", encoding="utf-8") as f:
        f.write(kt_auth_content)
        
    # 2. Create a meeting keynote markdown file
    keynotes_content = (
        "# Meeting Keynotes: Project Radar Sync - June 2026\n"
        "## Core Discussions\n"
        "- **Qdrant Integration**: We decided to run Qdrant in-memory/local-filesystem mode for developer setups to bypass Docker dependencies. The storage path is config.QDRANT_DIR.\n"
        "- **LLM Selected**: We selected the Qwen3:4b model for local generation because it runs comfortably on standard development workstations.\n"
        "- **Project Lead**: Sarah Connor is the official Project Lead for Radar, replacing John Bennett.\n"
        "- **Release Date**: The target alpha release date is July 15, 2026.\n"
    )
    with open(docs_dir / "keynotes_radar_sync.md", "w", encoding="utf-8") as f:
        f.write(keynotes_content)

    print("Mock documents created successfully.")

def run_tests():
    # Setup test files
    setup_test_documents()
    
    # Run Ingestion
    print("\n--- Running Ingestion ---")
    # Reset collection so we start clean
    ingest.ingest_documents(reset_db=True)
    
    # Verify vector search
    print("\n--- Verifying Qdrant Vector Search ---")
    client = QdrantClient(path=str(config.QDRANT_DIR))
    client.set_model(config.EMBED_MODEL)
    
    test_question = "Who is the Project Lead for Radar?"
    results = client.query(
        collection_name=config.QDRANT_COLLECTION,
        query_text=test_question,
        limit=2
    )
    
    print(f"Results for vector search on '{test_question}':")
    for idx, point in enumerate(results, 1):
        doc_name = point.metadata.get("document_name")
        print(f"  [{idx}] Source: {doc_name} | Score: {point.score * 100:.1f}%")
        # Print a snippet of the text
        snippet = point.metadata.get("document") or point.payload.get("document", "")
        print(f"      Snippet: {snippet[:120].replace('\n', ' ')}...")
        
    # Check if the correct file was fetched
    top_match = results[0].metadata.get("document_name")
    assert top_match == "keynotes_radar_sync.md", f"Expected keynotes_radar_sync.md as top match, got {top_match}"
    print("Vector Search Validation: PASSED!")
    client.close()
    
    # Run complete RAG query against LLM
    print("\n--- Running End-to-End RAG Query (FastEmbed + Qdrant + Qwen3:4b) ---")
    print(f"Question: 'What is the token expiration duration and the port of the Auth service?'")
    query.run_rag_query("What is the token expiration duration and the port of the Auth service?")
    
    print("\n--- Running Second RAG Query ---")
    print(f"Question: '{test_question}'")
    query.run_rag_query(test_question)
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    run_tests()
