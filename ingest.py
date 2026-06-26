import os
import sys
import argparse
from pathlib import Path
import pypdf
import docx
from qdrant_client.http import models

# Import configurations and helpers
import config
from qdrant_helper import get_qdrant_client

def parse_txt_or_md(file_path: Path) -> str:
    """Reads a text or markdown file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading text file {file_path.name}: {e}")
        return ""

def parse_pdf(file_path: Path) -> str:
    """Extracts text from a PDF file."""
    try:
        reader = pypdf.PdfReader(file_path)
        text = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n\n".join(text)
    except Exception as e:
        print(f"Error reading PDF file {file_path.name}: {e}")
        return ""

def parse_docx(file_path: Path) -> str:
    """Extracts text from a Word document (DOCX)."""
    try:
        doc = docx.Document(file_path)
        text = [paragraph.text for paragraph in doc.paragraphs]
        return "\n".join(text)
    except Exception as e:
        print(f"Error reading DOCX file {file_path.name}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Splits text into chunks, respecting word/sentence boundaries where possible."""
    if not text or not text.strip():
        return []
        
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # If we aren't at the end of the text, try to find a natural boundary
        if end < text_len:
            boundary_found = False
            # Look backwards up to 50 characters for a sentence or word end
            for i in range(end, max(end - 50, start), -1):
                if text[i] in ['.', '!', '?', '\n']:
                    end = i + 1
                    boundary_found = True
                    break
            
            if not boundary_found:
                # If no sentence end, look for a space to avoid cutting a word in half
                for i in range(end, max(end - 20, start), -1):
                    if text[i] == ' ':
                        end = i + 1
                        break
                        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        # Move start point back by the overlap
        start = end - chunk_overlap
        if start >= text_len or end >= text_len:
            break
            
    return chunks

def get_file_parser(suffix: str):
    """Maps file extension to corresponding parser function."""
    suffix = suffix.lower()
    if suffix in [".txt", ".md"]:
        return parse_txt_or_md
    elif suffix == ".pdf":
        return parse_pdf
    elif suffix == ".docx":
        return parse_docx
    return None

def ingest_documents(reset_db: bool = False):
    # Scan documents folder
    docs_path = Path(config.DOCS_DIR)
    supported_extensions = {".txt", ".md", ".pdf", ".docx"}
    
    files_to_process = [
        f for f in docs_path.iterdir()
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]
    
    if not files_to_process:
        print(f"No documents found in '{config.DOCS_DIR}'.")
        print("Please place some .txt, .md, .pdf, or .docx files in that directory.")
        return

    print("Initializing Qdrant client...")
    client = get_qdrant_client()
    
    collection_name = config.QDRANT_COLLECTION
    
    # Reset collection if requested
    if reset_db and client.collection_exists(collection_name):
        print(f"Resetting collection: {collection_name}")
        client.delete_collection(collection_name)
        
    # Create collection if it doesn't exist
    if not client.collection_exists(collection_name):
        print(f"Creating Qdrant collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=client.get_fastembed_vector_params()
        )
    
    print(f"Found {len(files_to_process)} document(s) to process.")
    
    for file_path in files_to_process:
        filename = file_path.name
        print(f"\nProcessing: {filename}...")
        
        parser = get_file_parser(file_path.suffix)
        if not parser:
            print(f"Skipping unsupported file type: {filename}")
            continue
            
        # Parse text
        text = parser(file_path)
        if not text.strip():
            print(f"Skipping empty or unreadable file: {filename}")
            continue
            
        # Chunk text
        chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        if not chunks:
            print(f"No chunks created for: {filename}")
            continue
            
        print(f"Created {len(chunks)} chunks for {filename}.")
        
        # If database is NOT reset, delete existing records for this file to avoid duplicates
        if not reset_db:
            try:
                # Delete old chunks for this file
                client.delete(
                    collection_name=collection_name,
                    points_selector=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_name",
                                match=models.MatchValue(value=filename)
                            )
                        ]
                    )
                )
            except Exception as e:
                # If collection is empty, delete might warn/fail, which is fine
                pass

        # Prepare records for ingestion
        chunk_texts = []
        chunk_metadatas = []
        
        for idx, chunk in enumerate(chunks):
            chunk_texts.append(chunk)
            chunk_metadatas.append({
                "document_name": filename,
                "file_path": str(file_path),
                "chunk_index": idx,
                "total_chunks": len(chunks)
            })
            
        # Insert chunks to Qdrant (auto-embeds using FastEmbed)
        print(f"Uploading vectors for {filename} to Qdrant...")
        client.add(
            collection_name=collection_name,
            documents=chunk_texts,
            metadata=chunk_metadatas
        )
        print(f"Successfully indexed {filename}!")
        
    print("\nIngestion process completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant vector database.")
    parser.add_argument(
        "--reset", 
        action="store_true", 
        help="Reset/clear the database collection before ingesting."
    )
    args = parser.parse_args()
    ingest_documents(reset_db=args.reset)
