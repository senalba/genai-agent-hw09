import logging
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from settings import settings

# Instantiate the embeddings model once at the module level to be reused.
# This is more efficient than creating a new instance on every function call.
embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

def load_and_split_pdf(pdf_path: str):
    """
    Loads a PDF and splits it into document chunks.
    """
    logging.info(f"Loading and splitting PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    logging.info(f"Split PDF into {len(split_docs)} chunks.")
    return split_docs

def clear_vector_store():
    """
    Deletes the existing ChromaDB collection.
    """
    try:
        logging.info(f"Attempting to delete collection: '{settings.CHROMA_COLLECTION_NAME}'")
        client.delete_collection(name=settings.CHROMA_COLLECTION_NAME)
        logging.info(f"Successfully deleted collection: '{settings.CHROMA_COLLECTION_NAME}'")
    except ValueError:
        # ChromaDB throws ValueError if the collection does not exist.
        logging.info(f"Collection '{settings.CHROMA_COLLECTION_NAME}' did not exist, proceeding to create it.")
    except Exception as e:
        logging.error(f"An unexpected error occurred while trying to delete collection: {e}", exc_info=True)
        # Re-raise the exception to be handled by the calling endpoint.
        raise

def get_vector_store(documents=None) -> Chroma:
    """
    Initializes and returns a Chroma vector store instance.
    If documents are provided, it creates the store from them.
    Otherwise, it loads the existing persistent store.
    """
    if documents:
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            client=client,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )
    
    return Chroma(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        client=client,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )
