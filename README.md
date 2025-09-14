# 📚 Your Personal Study Assistant: Unleash the Power of Your Notes!

## Description

This project transforms your static PDF documents, like class notes, research papers, or textbooks, into an interactive, conversational study assistant. Leveraging a Retrieval-Augmented Generation (RAG) pipeline, you can ask questions about your documents and get intelligent, contextualized answers based directly on the content you provide.

This application is a flexible and powerful starting point for anyone looking to build their own agentic AI tools using a modern tech stack including LangChain, FastAPI, and a vector database like ChromaDB.

**Disclaimer:** This tool is for informational purposes only and is not a substitute for professional research or academic guidance. Always verify information with your original source material.

## Features

- **FastAPI Backend**: A robust and easy-to-use API for interacting with the study assistant.
- **LangChain Agent with Memory**: A conversational agent built with LangChain that can reason, use tools, and remember previous interactions to provide more relevant answers.
- **ChromaDB Vector Store**: A lightweight local vector database for efficiently storing and retrieving information from your documents.
- **Hugging Face Embeddings**: Uses open-source sentence-transformer models to create high-quality vector embeddings of your text.
- **Flexible Data Ingestion**: A script designed to process a single PDF file and add its content to the vector store, allowing you to easily update your knowledge base.

## How It Works

The application is built on a two-part system: data ingestion and the agentic API.

### Data Ingestion (process_documents.py):
- The script takes a PDF file as a command-line argument.
- It loads the PDF, splits the content into smaller, manageable chunks, and creates vector embeddings for each chunk using a pre-trained sentence-transformer model.
- These embeddings are then saved to a ChromaDB vector store. This process intelligently adds new document data to the existing database, so you can build your knowledge base over time without losing previous information.

### Agentic API (main.py):
- A FastAPI application hosts a single API endpoint: `/ask_agent`.
- A specialized tool, `search_knowledge_base`, is defined for the agent. This tool queries the ChromaDB vector store to find the most relevant information based on your question.
- A conversational agent is created with a prompt that instructs it to act as a study assistant. It uses the `search_knowledge_base` tool and conversation history to synthesize a final, well-structured answer.

## Project Structure

```
.
├── chroma/                 # Directory for the ChromaDB vector store
├── data/
│   └── your_notes.pdf     # Your PDF document(s)
├── .gitignore             # Files to be ignored by Git
├── process_documents.py   # Script for data ingestion
├── main.py                # The main FastAPI application
└── requirements.txt       # Python dependencies
```

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/study-assistant-agent.git
   cd study-assistant-agent
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies:**

   Create a `requirements.txt` file with the following content:
   ```
   fastapi
   uvicorn
   langchain
   langchain-google-genai
   langchain-community
   chromadb
   sentence-transformers
   python-dotenv
   pypdf
   ```

   Then, install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your Google API Key:**

   Create a `.env` file in the root of the project and add your Google API key:
   ```
   GOOGLE_API_KEY="your-google-api-key"
   ```

5. **Run the data ingestion script:**

   This will process your PDF file and create the ChromaDB vector store. You must provide the file path using the `--file` flag.
   ```bash
   python process_documents.py --file "path/to/your/notes.pdf"
   ```

## Usage

1. **Start the FastAPI server:**
   ```bash
   uvicorn main:app --reload
   ```

2. **Access the API documentation:**

   Once the server is running, you can access the interactive API documentation at `http://127.0.0.1:8000/docs`.

3. **Send a request:**

   Use the API documentation to send a POST request to the `/ask_agent` endpoint with a JSON body.

   **Example Request:**
   ```json
   {
     "question": "What are the main causes and symptoms of Addison's disease?"
   }
   ```

## Dependencies

- **fastapi**: For creating the API.
- **uvicorn**: For running the FastAPI server.
- **langchain**: The main framework for building the agent.
- **langchain-google-genai**: For using Google's generative models.
- **langchain-community**: For community-provided LangChain components like loaders and vector stores.
- **chromadb**: The vector store for storing and retrieving embeddings.
- **sentence-transformers**: For creating the text embeddings.
- **python-dotenv**: For loading environment variables.
- **pypdf**: For parsing and reading PDF documents.

## Future Improvements

- **Add a frontend**: Create a simple chat interface to make the agent more user-friendly.
- **Support multiple file types**: Expand the knowledge base by adding support for other file formats like .txt, .docx, or even web pages.
- **Advanced agent capabilities**: Experiment with more sophisticated agent types or prompts to improve conversational flow and reasoning.