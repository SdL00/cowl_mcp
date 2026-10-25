# 🧠 COWL-MCP Smart Home Ontology Assistant

This project implements a **client–server architecture** that enables **natural language querying** over an **OWL-based smart home ontology**.  
It integrates a **Language Model (LLM)** (via [Ollama](https://ollama.com/)) with a **Model Context Protocol (MCP)** server exposing ontology-related tools through REST endpoints.

The system allows users to ask questions such as:  
> 💡 “Which lights are on in the living room?”  
> 🌡️ “What is the current temperature in the bedroom?”  
> ⚙️ “List all appliances connected to the kitchen.”

The model translates these natural language queries into ontology-compliant URIs and tool calls, retrieves relevant axioms or entities from the knowledge graph, and generates contextually informed responses.


## ⚙️ System Requirements

- 🐍 **Python** ≥ 3.10  
- 🐳 **Docker** (for running the COWL ontology API container)  
- 🤖 **Ollama** installed and running locally  
- ⚡ **GPU (adopted)** — NVIDIA RTX A6000 or equivalent for embedding acceleration  


## Architecture Overview

- **`server.py`** → Starts a local **MCP server** exposing ontology operations (`get_all_entities`, `get_axioms_by_type`, etc.).  
  It loads the ontology into a **COWL API** instance running in a Docker container.  
- **`client.py`** → Connects to the MCP server via stdio transport, retrieves ontology entities, computes embeddings via **Ollama**, and runs an interactive chat interface for natural language querying.  
  Each query is processed independently — conversation history is **reset per turn** to ensure consistent evaluation.


## Core Components

| Component | Description |
|------------|-------------|
| **COWL API** | Ontology backend for storing and querying OWL graphs (exposed via REST). |
| **MCP Server** | Middleware converting REST API endpoints into callable tools for the client. |
| **Ollama Models** | Used for reasoning (`llama3.1`) and embeddings (`all-minilm`). |
| **Entity Embeddings** | Represent ontology entities as semantic vectors for similarity matching. |
| **TOP_K_ENTITIES** | Controls how many relevant entities are retrieved per user query. |


## Setup Instructions

### Start the Ontology Server (COWL API via Docker)

Run the official **COWL API** container on port `49201`:

```bash
sudo docker run --name cowl-api -p 49201:49201 -e COWL_PERSISTENT=false --rm cowl-api:latest
```

This command starts the ontology REST service required by the system.

### Start the Ollama Service

Start the Ollama server to provide embedding and chat capabilities:

```bash
ollama serve
```

### Start the MCP Components

Run both the client and server scripts to launch the MCP system:

```bash
python3 client.py server.py
```

Once running, the user can interact via the client's terminal interface.

The client will:

- Parse ontology entities from `home.owl`  
- Compute embeddings using the specified embedding model  
- Launch an interactive chat interface for ontology-based natural language queries  

