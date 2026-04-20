import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_PATH = BASE_DIR / "documents"


def load_documents():
    all_docs = []

    for file in os.listdir(DOCUMENTS_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(str(DOCUMENTS_PATH / file))
            docs = loader.load()

            # ✅ IMPORTANT: attach source metadata
            for doc in docs:
                doc.metadata["source"] = file

            all_docs.extend(docs)

    return all_docs


def create_vector_store():
    documents = load_documents()

    # DEBUG (you can remove later)
    print("Loaded documents:", len(documents))
    if documents:
        print("Sample text:", documents[0].page_content[:200])

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)

    return db