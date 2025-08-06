import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class PDFChromaTool:
    def __init__(self, collection_name="pdf_collection", model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chroma_client = chromadb.Client(Settings(persist_directory="./chroma_db"))
        if collection_name not in [c.name for c in self.chroma_client.list_collections()]:
            self.collection = self.chroma_client.create_collection(collection_name)
        else:
            self.collection = self.chroma_client.get_collection(collection_name)

    def load_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        self.chunks = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                self.chunks.append(text.strip())

    def build_index(self):
        embeddings = self.model.encode(self.chunks)
        ids = [str(i) for i in range(len(self.chunks))]
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=self.chunks,
            ids=ids
        )

    def query(self, question, top_k=3):
        q_emb = self.model.encode([question])[0]
        results = self.collection.query(query_embeddings=[q_emb.tolist()], n_results=top_k)
        # Повертаємо уривки, якщо є
        return results['documents'][0] if results['documents'] else []
