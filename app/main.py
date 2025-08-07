import logging
import tempfile
import os
from typing import Literal
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from pdf_utils import load_and_split_pdf, clear_vector_store, get_vector_store
from agent import create_agent, get_session_history
import state



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GenAI PDF Agent API",
    description="API for interacting with a RAG agent for PDF documents.",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str
    session_id: str

@app.get("/health", summary="Check API Health")
def health_check():
    return {"status": "ok"}

@app.post("/index_pdf", summary="Upload and Index a PDF")
async def index_pdf(
    pdf_file: UploadFile = File(...),
    mode: Literal["replace", "add"] = Form(...)
):
    """
    Accepts a PDF file and an indexing mode.
    - 'replace': Clears the existing context and indexes the new PDF.
    - 'add': Adds the new PDF to the existing context.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await pdf_file.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.info(f"PDF file '{pdf_file.filename}' saved to temporary path: {tmp_path}")
        split_docs = load_and_split_pdf(tmp_path)

        if mode == "replace":
            logger.info("Mode: 'replace'. Clearing existing vector store and creating a new one.")
            clear_vector_store()
            state.vector_store = get_vector_store(documents=split_docs)
        elif mode == "add":
            logger.info("Mode: 'add'. Adding documents to vector store.")
            if state.vector_store is None:
                logger.info("No existing vector store in memory, loading from disk.")
                state.vector_store = get_vector_store()

            state.vector_store.add_documents(split_docs)
            logger.info(f"Added {len(split_docs)} new document chunks.")

        # Always recreate the agent to use the updated retriever from the vector store
        state.agent_executor = create_agent(state.vector_store)

        # This state is now ambiguous as it only tracks the last uploaded PDF.
        # The tools relying on it will only work for the last file.
        # This is a known limitation of the current tool design.
        state.indexed_pdf_path = tmp_path

        logger.info("PDF indexed and agent updated successfully.")
        return {"message": f"PDF '{pdf_file.filename}' indexed successfully in '{mode}' mode."}
    except Exception as e:
        logger.error(f"Error indexing PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to index PDF: {e}")
    finally:
        # Ensure the temporary file is cleaned up
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.info(f"Cleaned up temporary file: {tmp_path}")

@app.post("/query", summary="Query the Agent")
async def query_agent(request: QueryRequest):
    """
    Receives a question and a session_id, passes it to the agent,
    and returns the agent's response.
    """
    if not all([request.question, request.session_id]):
        raise HTTPException(status_code=400, detail="Both 'question' and 'session_id' are required.")

    if state.agent_executor is None:
        return {"answer": "The agent is not ready. Please upload and index a PDF file first."}

    try:
        logger.info(f"Querying agent for session_id: {request.session_id}")
        response = state.agent_executor.invoke(
            {"input": request.question},
            config={"configurable": {"session_id": request.session_id}}
        )
        return {"answer": response.get("output", "No valid output from agent.")}
    except Exception as e:
        logger.error(f"Error during agent execution for session {request.session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during agent execution: {e}")
