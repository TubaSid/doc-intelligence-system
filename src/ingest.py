import PyPDF2
from pathlib import Path

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF"""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        
        chunks.append({
            "text": chunk_text,
            "char_start": start,
            "char_end": end,
            "chunk_id": len(chunks)
        })
        
        start += (chunk_size - overlap)
    
    return chunks

if __name__ == "__main__":
    pdf_file = "data/raw/tesla_10k.pdf"  # Your PDF here
    
    text = extract_text_from_pdf(pdf_file)
    print(f"Extracted {len(text)} characters")
    print(f"\nFirst 500 chars:\n{text[:500]}")
    print("="*50)
    
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")
    print(f"\nChunk 0:\n{chunks[0]['text'][:200]}...")