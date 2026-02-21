# 🌍 Travel Advisor AI

A sophisticated, AI-powered travel research assistant that combines Large Language Models (LLM), Retrieval-Augmented Generation (RAG), and real-time tools to plan your perfect trip.

## 🚀 Overview

Travel Advisor AI uses a multi-agent system built with **LangGraph** to process user queries. It can ground its answers in travel-specific documentation using a **vector store (FAISS)** and fetch real-time data for flights and hotels using **SerpApi** and the **Model Context Protocol (MCP)**.

---

## ✨ Key Features

- **Smart Trip Planning**: Generates detailed itineraries, including activities, accommodation types, and transportation.
- **Real-time Data Integration**:
  - ✈️ **Flight Search**: Fetches live flight options (airline, price, duration) via SerpApi.
  - 🏨 **Hotel Search**: Finds top hotels with prices, ratings, and amenities using SerpApi via MCP.
  - 📍 **Auto-Location**: Detects your starting point using IP-based location services.
- **Grounded Knowledge (RAG)**: Uses a local FAISS index of travel guides (WikiVoyage) to provide historically and culturally accurate information.
- **Intelligent Reranking**: Uses a Cross-Encoder reranker to ensure the most relevant context is provided to the LLM.
- **Structured Outputs**: Always responds in valid JSON for easy integration with frontend applications.

---

## 🛠️ Tech Stack

- **Framework**: [LangGraph](https://www.langchain.com/langgraph) & [LangChain](https://www.langchain.com/)
- **LLM**: Powered by **Groq** (using Qwen/Llama models)
- **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss)
- **Embeddings**: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
- **Reranker**: `BAAI/bge-reranker-base` via CrossEncoder
- **Real-time Tools**: SerpApi (Google Flights & Google Hotels via MCP)

---

## 📋 Prerequisites

- Python 3.10+
- API Keys:
  - **Groq API Key**: For LLM inference.
  - **SerpApi Key**: For live travel data.

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone <repository-url>
cd Travel_Advisor
```

### 2. Set up a Virtual Environment
It is recommended to use a virtual environment to manage dependencies:
```bash
python3 -m venv env
source env/bin/activate  # On macOS/Linux
# env\Scripts\activate   # On Windows
```

### 3. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the `activity_planner/` directory. This file is excluded from git for security.
```bash
touch activity_planner/.env
```
Add your API keys to `activity_planner/.env`:
```env
OPEN_API_KEY=your_groq_api_key
SERP_API_KEY=your_serpapi_key
```

### 5. Essential Downloads & Data Preparation
- **Models**: The first time you run the application, it will automatically download the embedding model (`all-MiniLM-L6-v2`) and the reranker model (`bge-reranker-base`). These are approximately 500MB total.
- **FAISS Index**: If you don't have a pre-built index, the system will attempt to create one from the data in `Data_preparation/`. Ensure your raw JSON data is placed in the correct directory as defined in `Faiss_indexing.py`.

---

## 🚀 Usage

### 🎨 Running with Streamlit UI (Recommended)
For a beautiful, interactive web interface:
```bash
streamlit run app.py
```
This will open a browser window with a chat-based interface featuring:
- 💬 Interactive chat conversation
- 🎨 Beautiful trip plan visualizations
- 📊 Formatted flight and hotel results
- 🔄 Persistent conversation history
- 🎯 Helpful tips and examples

### 🖥️ Running the CLI Assistant
To start the interactive command-line agent:
```bash
python3 activity_planner/Agents.py
```

### Updating Requirements
If you install new packages and want to update the `requirements.txt` file:
```bash
pip freeze > requirements.txt
```

### Creating the Search Index manually
If you want to force-rebuild the FAISS vector store index:
```bash
# Delete existing index if necessary
rm -rf faiss_index
# Run the indexing script (integrated into the initialization logic)
python3 activity_planner/Faiss_indexing.py
```

### **Example Queries:**
- *"Plan a 5-day trip to Tokyo in April."*
- *"Find a flight from my location to Paris and suggest some hotels."*
- *"What are the best cultural spots in Rome?"*

---

## 📂 Project Structure

- `app.py`: **Streamlit web interface** - Beautiful UI for interacting with the agent.
- `activity_planner/`
  - `Agents.py`: Main LangGraph logic, state definition, and agent nodes.
  - `Model.py`: LLM configuration and tool binding.
  - `tools.py`: Implementation of search and location tools.
  - `MCP_Client.py`: Client for interacting with SerpApi's Model Context Protocol.
  - `Faiss_indexing.py`: Vector store creation and loading.
  - `Data_loading.py`: Processing and chunking travel documents.
- `Data_preparation/`: (Optional) Scripts for processing raw data feeds.

---

## 🧠 Architecture

The system follows a cycles-based architecture:
1. **RAG Node**: Retrieves relevant local travel documents based on the query.
2. **LLM Node**: Processes the query + context and decides if any external tools are needed.
3. **Action Node**: Executes tools (flights, hotels, location) if the LLM requests them.
4. **Final Response**: Returns a structured JSON containing the complete trip plan.

---

## 📄 License

