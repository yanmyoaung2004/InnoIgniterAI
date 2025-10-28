from rag_db import add_documents
import PyPDF2

def extract_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into chunks of chunk_size with overlap.
    Returns a list of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    pdf_path = "data/MyanmarLaw.pdf" 
    text = extract_pdf_text(pdf_path)
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    ids = [f"{pdf_path}_chunk_{i}" for i in range(len(chunks))]
    add_documents(chunks, ids)
    print(f"Added {len(chunks)} chunks from {pdf_path} to the vector DB.")