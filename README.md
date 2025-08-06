# GenAI PDF Agent

This project implements a sophisticated Retrieval-Augmented Generation (RAG) agent designed to interact with PDF documents. Users can upload one or more PDFs and ask questions about their content. The agent is equipped with several custom tools to provide detailed information beyond simple text retrieval.

The application is built with a decoupled architecture using FastAPI for the backend API and Streamlit for the user interface, all containerized with Docker for easy setup and deployment.

## Features

- **PDF-based Q&A**: Upload a PDF and ask the agent questions about its content.
- **Dynamic Context Management**: Choose to either **replace** the agent's knowledge with a new PDF or **add** a new PDF to its existing context, allowing for multi-document conversations.
- **Advanced Custom Tools**: The agent can use specialized tools to:
    - Retrieve the **table of contents**.
    - Extract **metadata** (author, title, creation date, etc.).
    - Count the total number of **pages**.
- **Conversational Memory**: The agent remembers the context of the conversation for each user session.
- **Multi-User Support**: Separate chat histories are maintained for each user session.
- **Web UI**: An easy-to-use interface built with Streamlit.

## How It Works

The agent's workflow is designed for robustness and scalability:

1.  **UI (Streamlit)**: The user uploads a PDF file and selects an **Indexing Mode**:
    - **Replace existing context**: Clears all previously loaded documents and starts a fresh session with the new PDF.
    - **Add to existing context**: Adds the new PDF's content to the agent's existing knowledge base.
2.  **API Interaction**: The UI sends the PDF file and the selected mode directly to a FastAPI endpoint. This decoupled approach is more robust than relying on a shared file system.
3.  **Backend (FastAPI)**:
    - The backend receives the PDF and saves it to a temporary location.
    - Based on the selected mode, it either clears the ChromaDB vector collection or prepares to add to it.
    - It uses `langchain` to load the PDF, split it into chunks, and create vector embeddings using an OpenAI model.
    - These embeddings are indexed in a persistent `ChromaDB` vector store.
    - A new agent instance is created or updated with a retriever that has access to the latest document set.
4.  **Conversational Agent**:
    - The user asks a question in the UI.
    - The UI sends the question and a unique `session_id` to the backend's query endpoint.
    - The LangChain agent receives the question. It can choose to use one of its tools or answer directly based on the conversation history and the LLM's knowledge.
    - The final answer is returned to the user.

## Custom Tools Implemented

The agent has access to the following tools to answer user queries:

- **`pdf_document_retriever`**: The primary tool for semantic search. It retrieves relevant chunks of text from all indexed PDF documents based on the user's question.
- **`get_pdf_page_count`**: A custom tool that reports the total number of pages in the *last uploaded* PDF.
- **`get_pdf_metadata`**: A custom tool that extracts metadata (e.g., author, title, creation date) from the *last uploaded* PDF.
- **`get_pdf_toc`**: A custom tool that lists the table of contents (bookmarks) from the *last uploaded* PDF, if available.

*Note: The metadata, page count, and TOC tools currently operate only on the most recently uploaded document.*

## How to Run

1.  **Prerequisites**:
    - Docker and Docker Compose must be installed.

2.  **Configuration**:
    - Edit the `.env` file and add your OpenAI API key.

3.  **Build and Run**:
    - Run the application using Docker Compose:
      ```bash
      docker-compose up --build
      ```

4.  **Access the Application**:
    - **UI**: Open your browser and go to `http://localhost:8503`.
    - **API Docs**: The FastAPI documentation is available at `http://localhost:8000/docs`.

## Demonstration

You can interact with the agent by asking questions like:

- "What is this document about?"
- "How many pages are in the document?"
- "Who is the author of this PDF?"
- "Can you show me the table of contents?"
- (After uploading a second PDF in 'add' mode) "Compare the conclusions from the first and second documents."

## Demonstration

You can interact with the agent by asking questions like:

- "What is this document about?"
- "How many pages are in the document?"
- "Who is the author of this PDF?"
- "Can you show me the table of contents?"
- (After uploading a second PDF in 'add' mode) "Compare the conclusions from the first and second documents."
