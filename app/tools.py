import fitz  # PyMuPDF
import logging
from langchain_core.tools import Tool
# from app import state
import state


def get_pdf_page_count(query: str) -> str:
    """
    Counts the number of pages in the currently indexed PDF file.
    The input query is ignored, as the tool operates on the globally indexed file.
    """
    current_path = state.indexed_pdf_path
    logging.info(f"Page count tool called. PDF path: {current_path}")

    if not current_path:
        return "Error: No PDF has been indexed yet. Please upload a PDF first."
    try:
        with fitz.open(current_path) as doc:
            page_count = doc.page_count
        return f"The PDF document has {page_count} pages."
    except Exception as e:
        logging.error(f"Failed to open or process PDF for page count: {e}")
        return f"An error occurred while trying to count the pages: {e}"

def get_pdf_metadata(query: str) -> str:
    """
    Returns PDF metadata like title, author, etc.
    """
    current_path = state.indexed_pdf_path
    if not current_path:
        return "Error: No PDF has been indexed yet."
    try:
        with fitz.open(current_path) as doc:
            meta = doc.metadata
        meta_str = "\n".join([f"{k}: {v}" for k, v in meta.items() if v])
        return f"PDF Metadata:\n{meta_str}" if meta_str else "No metadata found."
    except Exception as e:
        return f"Error retrieving metadata: {e}"

def get_pdf_toc(query: str) -> str:
    """
    Returns the table of contents (bookmarks) of the PDF, if present.
    """
    current_path = state.indexed_pdf_path
    if not current_path:
        return "Error: No PDF has been indexed yet."
    try:
        with fitz.open(current_path) as doc:
            toc = doc.get_toc()
        if not toc:
            return "No table of contents found in this PDF."
        toc_str = "\n".join([f"Level {item[0]}: {item[1]} (page {item[2]})" for item in toc])
        return toc_str
    except Exception as e:
        return f"Error reading table of contents: {e}"

pdf_toc_tool = Tool.from_function(
    name="get_pdf_toc",
    description="Use this tool to list the PDF's table of contents or bookmarks.",
    func=get_pdf_toc,
)

pdf_metadata_tool = Tool.from_function(
    name="get_pdf_metadata",
    description="Use this tool to get metadata (author, title, etc.) of the current PDF.",
    func=get_pdf_metadata,
)

page_count_tool = Tool.from_function(
    name="get_pdf_page_count",
    description="Use this tool to find out the total number of pages in the provided PDF document.",
    func=get_pdf_page_count,
)
