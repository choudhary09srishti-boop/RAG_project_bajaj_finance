from pypdf import PdfReader

# -----------------------------
# STEP 1: LOAD PDF
# -----------------------------
pdf_path = "data/bajaj_terms.pdf"  # Make sure your PDF is inside 'data' folder
reader = PdfReader(pdf_path)

# Extract all text
full_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text + "\n"

print("Total length of document:", len(full_text))

# -----------------------------
# STEP 2: CHUNKING
# -----------------------------
chunk_size = 500   # characters per chunk
overlap = 50       # overlap between chunks

chunks = []
start = 0
while start < len(full_text):
    end = start + chunk_size
    chunk = full_text[start:end]
    chunks.append(chunk)
    start += chunk_size - overlap

print("Total chunks created:", len(chunks))
print("\n--- Sample Chunk ---\n")
print(chunks[0])
