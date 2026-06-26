import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Import configurations and helpers
import config
from qdrant_helper import get_qdrant_client

def format_score(score: float) -> str:
    """Formats similarity score as percentage."""
    return f"{score * 100:.1f}%"

def run_rag_query(query_str: str, top_k: int = 3, stream: bool = True):
    """Executes the RAG pipeline: retrieves context, formats prompt, and gets LLM answer."""
    # Connect to Qdrant (cloud or local based on config)
    client = get_qdrant_client()
    
    if not client.collection_exists(config.QDRANT_COLLECTION):
        print(f"Error: Collection '{config.QDRANT_COLLECTION}' does not exist.")
        print("Please run ingest.py to parse and index your documents first.")
        return
        
    print(f"\nSearching for: '{query_str}'...")
    
    # Search Qdrant for matching chunks
    # client.query auto-embeds the query string and retrieves matches
    search_results = client.query(
        collection_name=config.QDRANT_COLLECTION,
        query_text=query_str,
        limit=top_k
    )
    
    if not search_results:
        print("No matching information found in the document database.")
        return

    # Process and build context
    context_parts = []
    sources = []
    
    for idx, point in enumerate(search_results):
        # Qdrant client.query returns elements where document contains the text 
        # and metadata contains metadata dict. Let's support both point.payload and point.metadata.
        text = getattr(point, "document", None) or point.payload.get("document", "")
        metadata = getattr(point, "metadata", None) or point.payload or {}
        score = point.score
        
        doc_name = metadata.get("document_name", "Unknown Source")
        chunk_idx = metadata.get("chunk_index", 0)
        
        context_parts.append(f"Source: {doc_name} (Chunk {chunk_idx})\nContent: {text}")
        
        # Keep track of unique sources for summary output
        sources.append({
            "name": doc_name,
            "score": score
        })

    # Combine context
    context_text = "\n\n=== CONTEXT SPLIT ===\n\n".join(context_parts)
    
    # Build System prompt with instructions and context
    system_prompt = (
        "You are a helpful AI assistant. You answer user queries based on the provided context documents.\n"
        "Instructions:\n"
        "1. Base your answer strictly on the retrieved context below.\n"
        "2. If the context does not contain enough information to answer the question, respond with: "
        "'I am sorry, but the provided documents do not contain information to answer this question.'\n"
        "3. Keep your answers accurate, well-structured, and factual.\n\n"
        "Retrieved Context:\n"
        "--------------------------------------------------\n"
        f"{context_text}\n"
        "--------------------------------------------------\n"
    )

    print("\n" + "="*50)
    print("AI RESPONSE:")
    print("="*50)

    # Call Groq LLM
    try:
        load_dotenv()
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        # Create chat completion request
        response = groq_client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query_str}
            ],
            temperature=0.2,
            stream=stream
        )
        if stream:
            # Streaming response yields chunks
            for chunk in response:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                # Skip None content (e.g. role-only or finish chunks)
                if content:
                    print(content, end="", flush=True)
            print()
        else:
            answer = response.choices[0].message.content
            print(answer)
    except Exception as e:
        print(f"\nError communicating with Groq API: {e}")
        print("Please ensure that the GROQ_API_KEY is set correctly and the model name is valid.")
        return

    # Print source documents summary
    print("\n" + "="*50)
    print("SOURCES USED:")
    print("="*50)
    
    # Deduplicate sources while keeping the highest score for each
    unique_sources = {}
    for s in sources:
        name = s["name"]
        if name not in unique_sources or s["score"] > unique_sources[name]:
            unique_sources[name] = s["score"]
            
    for idx, (name, score) in enumerate(unique_sources.items(), 1):
        print(f"{idx}. {name} (Relevance Score: {format_score(score)})")
    print("="*50 + "\n")

def interactive_loop():
    print("==================================================")
    print("Radar RAG CLI System (Qdrant Cloud + Groq)")
    print("Type 'exit' or 'quit' to close.")
    print("==================================================")
    
    while True:
        try:
            query = input("Ask a question > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
                
            run_rag_query(query)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}\n")

if __name__ == "__main__":
    # Check if query was passed as command-line arguments
    if len(sys.argv) > 1:
        query_str = " ".join(sys.argv[1:])
        run_rag_query(query_str)
    else:
        interactive_loop()
