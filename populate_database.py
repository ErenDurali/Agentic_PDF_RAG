import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
import pypdf

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    ## Check if the chroma directory exists and should be cleared
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help = "Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("Clearing the database.")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

###Loading Part###
# This is for loading the pdfs
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

###Splitting Part###
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

def preprocess_math_content(text):
    """
    Pre-process text to better handle mathematical content.
    This can help clean up common OCR errors in math formulas.
    """
    # Replace common OCR errors in math notation
    replacements = {
        '≈': '~',        # Approximately 
        '×': 'x',        # Multiplication
        '÷': '/',        # Division
        '−': '-',        # Minus
        '∑': 'sum',      # Summation
        '∫': 'integral', # Integral
        '∞': 'infinity', # Infinity
        '∈': 'in',       # Element of
        '∀': 'for all',  # For all
        '∃': 'there exists', # There exists
        # Add more replacements as needed
    }
    
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    
    return text

# Then in your split_documents function
def split_documents(documents: list[Document]):
    # Preprocess documents to handle math better
    for doc in documents:
        doc.page_content = preprocess_math_content(doc.page_content)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
        length_function = len,
        is_separator_regex = False
    )
    return text_splitter.split_documents(documents)

###Database Part###
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma

def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory = CHROMA_PATH, embedding_function = get_embedding_function()
    )

    # Calculate the Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or update the chunks in the database.
    existing_items = db.get(include=[]) # IDs are always included by default.
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in database: {len(existing_ids)}")

    # Only add documents that don't exist in the database.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents to the database: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids = new_chunk_ids)
        db.persist()
    else:
        print("No new documents to add.")

def calculate_chunk_ids(chunks):

    # This is for create IDs for the chunks
    # Page source : page number : chunk index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        #Calculate the chunk ID
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add the ID to the metadata.
        chunk.metadata["id"] = chunk_id
    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()
