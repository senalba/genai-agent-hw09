from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_message_histories import ChatMessageHistory

# from app.tools import page_count_tool
from tools import page_count_tool, pdf_metadata_tool, pdf_toc_tool
from settings import settings

# In-memory store for chat histories. In a production app, use a persistent store like Redis.
memory_store = {}

def get_session_history(session_id: str):
    if session_id not in memory_store:
        memory_store[session_id] = ChatMessageHistory()
    return memory_store[session_id]

def create_agent(vector_store):
    """Creates a conversational RAG agent."""
    llm = ChatOpenAI(
        model=settings.LLM_MODEL_NAME,
        temperature=0.7,
        api_key=settings.OPENAI_API_KEY
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    retriever_tool = create_retriever_tool(
        retriever,
        "pdf_document_retriever",
        "Searches and returns relevant information from the indexed PDF document.",
    )

    tools = [retriever_tool, page_count_tool, pdf_metadata_tool, pdf_toc_tool]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You are a helpful AI assistant that answers questions based on the provided PDF document. "
        "You have access to tools for searching the document, counting the number of pages, and retrieving PDF metadata (such as title, author, and creation date), as well as the table of contents."
        "Use these tools as needed to provide accurate and helpful responses."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])


    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    agent_with_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return agent_with_history