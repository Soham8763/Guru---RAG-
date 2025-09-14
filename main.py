import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import tool, AgentExecutor, create_react_agent
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory

load_dotenv()

CHROMA_PATH = "chroma"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@tool
def search_knowledge_base(query:str) -> str:
    """
    Searches and returns relevant information from the knowledge base.
    Use this tool to answer questions about the content of the uploaded documents.
    """

    embedding_func = HuggingFaceEmbeddings(model_name=MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_func)
    results = db.similarity_search(query, k=10)

    if not results:
        return "I could not find any relevant information for that query."

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    return context_text

REACT_PROMPT_TEMPLATE = """
You are a study assistant. Your goal is to help the user understand the content of their class notes and other uploaded documents.

To do this, you MUST use the `search_knowledge_base` tool to find relevant information.

- You should maintain a conversation with the user and ask clarifying questions to get more information about their query.
- Use the conversation history to inform your answers.
- After your research is complete, synthesize the information from the tool to provide a final, coherent answer.
- In your final answer, cite the specific ideas you are referencing from your search results.

TOOLS:
------
You have access to the following tools:
{tools}

To use a tool, please use the following format:
Thought: Do I need to use a tool? Yes
Action: The action to take, should be one of [{tool_names}]
Action Input: The input to the action
Observation: The result of the action

When you have a response to say to the user, or if you do not need to use a tool, you MUST use the following format:
Thought: Do I need to use a tool? No
Final Answer: [Your comprehensive and well-structured response here]
Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)
tools = [search_knowledge_base]
agent = create_react_agent(llm, tools, prompt)

memory = ConversationBufferMemory(memory_key="chat_history")

agent_executer = AgentExecutor(
    agent = agent,
    tools = tools,
    verbose = True,
    handle_parsing_errors = True,
    memory=memory
)

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask the agent")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The agent's answer to the question.")


app = FastAPI(
    title="Study Assistant RAG API",
    description="An API for getting information from uploaded documents.",
    version="1.0.0",
)


@app.post("/ask_agent", response_model=QueryResponse)
async def ask_agent_endpoint(request: QueryRequest) -> QueryResponse:
    response = await agent_executer.ainvoke({"input": request.question})
    return QueryResponse(answer=response["output"])