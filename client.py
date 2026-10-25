
# This file implements an MCP client that connects to the ontology server,
# extracts entities from the OWL ontology, embeds them using Ollama, and
# interacts with the user via chat to query the ontology.

# Import libraries
import asyncio
import sys
import json
import re
from ollama import AsyncClient as OllamaAsyncClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from typing import List, Dict
import numpy as np


#Configuration
OLLAMA_HOST = "http://localhost:11434"  # Ollama API host
OLLAMA_MODEL = "llama3.1"               # Model for reasoning
EMBEDDING_MODEL = "all-minilm"          # Model for embeddings
TOP_K_ENTITIES = 10                     # Number of entities to include per query

SYSTEM_PROMPT = """
You are a smart home assistant.
"""

# Client CLasss
class MCPClient:
    def __init__(self, ontology_path: str):
        # Session and transport management
        self.session = None
        self.stdio = None
        self.write = None
        self.exit_stack = AsyncExitStack()

        # Ollama client
        self.ollama_client = OllamaAsyncClient(host=OLLAMA_HOST)

        # Ontology and entity data
        self.ontology_path = ontology_path
        self.entities: Dict[str, str] = {}  # {entity_name: type}
        self.entity_embeddings: Dict[str, List[float]] = {}

    async def load_entities(self):
        """
        Parse OWL Functional Syntax file and extract:
        - Classes
        - NamedIndividuals
        Then compute embeddings for each entity for similarity search.
        """
        with open(self.ontology_path, "r", encoding="utf-8") as f:
            owl_text = f.read()

        # Extract Classes using regex
        class_matches = re.findall(r'# Class: (:[\w_]+)', owl_text)
        for cls in class_matches:
            self.entities[cls] = "Class"

        # Extract NamedIndividuals
        ind_matches = re.findall(r'# Individual: (:[\w_]+)', owl_text)
        for ind in ind_matches:
            self.entities[ind] = "NamedIndividual"

        # Compute embeddings for all entities
        for entity in self.entities.keys():
            emb_resp = await self.ollama_client.embeddings(EMBEDDING_MODEL, entity)
            self.entity_embeddings[entity] = emb_resp["embedding"]

        print(f"[INFO] Loaded {len(self.entities)} entities and computed embeddings.")
        print(self.entities)

    async def connect(self, server_script: str):
        """
        Connect to MCP server (server.py) via stdio transport.
        """
        server_params = StdioServerParameters(
            command="python3",
            args=[server_script],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        
        # List tools exposed by the server
        resp = await self.session.list_tools()
        tool_names = [tool.name for tool in resp.tools]
        print(f"[INFO] Connected to MCP server with tools: {tool_names}")

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Compute cosine similarity between two embedding vectors.
        """
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    async def select_relevant_entities(self, query: str) -> List[str]:
        """
        Compute embedding of query and select top-K most relevant ontology entities.
        """
        query_emb = (await self.ollama_client.embeddings(EMBEDDING_MODEL, query))["embedding"]
        sims = [(entity, self.cosine_similarity(query_emb, emb)) for entity, emb in self.entity_embeddings.items()]
        sims.sort(key=lambda x: x[1], reverse=True)
        top_entities = [entity for entity, _ in sims[:TOP_K_ENTITIES]]
        return top_entities

    async def chat(self):
        """
        Interactive chat loop between user and MCP-server assistant.
        """
        print("MCP client ready. Type your messages, or 'quit' to exit.")

        history = [{"role": "system", "content": SYSTEM_PROMPT}]

        while True:
            # print("INITIAL HISTORY: ", history)
            user_input = input("You: ").strip()
            if user_input.lower() in {"quit", "exit"}:
                break
            
            # Retrieve relevant ontology entities
            relevant_entities = await self.select_relevant_entities(user_input)
            entity_text = "\n".join([f"- {e} ({self.entities[e]})" for e in relevant_entities])

            # Construct enriched prompt for LLM with context
            prompt_with_entities = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Available entities relevant to this query:\n{entity_text}\n\n"
                f"Available tools:\n"
                "- get_axioms_by_entity(entity_type: str, iri: str)\n"
                "- get_entities_by_type(type: str)\n"
                "- get_axioms_by_type(axiom_type: str)\n"
                "- get_axioms_by_entity_and_type(entity_type: str, iri: str, axiom_type: str)\n\n"
                f"User query: {user_input}"
            )

            print(prompt_with_entities)

            history = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_with_entities}
            ]

            print("NEW HISTORY: ", history)

             # Get available tools from the server
            tool_resp = await self.session.list_tools()
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                }
                for tool in tool_resp.tools
            ]

            # Ask Ollama to respond (may call a tool)
            response = await self.ollama_client.chat(
                model=OLLAMA_MODEL,
                messages=history,
                tools=tools,
                stream=False
            )

            message = response.get("message", {})
            tool_calls = message.get("tool_calls", [])

            if tool_calls:
                # If LLM requested a tool call
                tool_call = tool_calls[0]
                tool_name = tool_call["function"]["name"]
                tool_args = tool_call["function"]["arguments"]

                print(f"[Tool Call] {tool_name}({tool_args})")

                # Execute tool on MCP server
                result = await self.session.call_tool(tool_name, tool_args)
                result_text = json.dumps(result.structuredContent)

                print(f"[Tool Result] {result_text}")

                # Append tool call and result to conversation history
                history.append({"role": "assistant", "tool_calls": [tool_call]})
                history.append({"role": "tool", "tool_call_id": tool_call.get("id", ""), "name": tool_name, "content": result_text})

                # print("HISTORY BEFORE FOLLOW-UP: ", history)

                # Ask LLM to follow up after tool result
                stream = await self.ollama_client.chat(model=OLLAMA_MODEL, messages=history, stream=True)
                reply = ""
                async for chunk in stream:
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        print(delta, end="", flush=True)
                        reply += delta
                print()
                history.append({"role": "assistant", "content": reply})
                # print("HISTORY AFTER FOLLOW-UP: ", history)

            else:
                # If no tool call, just print assistant's answer
                reply = message.get("content", "[No reply]")
                print(f"Assistant: {reply}")
                history.append({"role": "assistant", "content": reply})
                # print("HISTORY NO TOOL CALL: ", history)

    async def close(self):
        """
        Clean up all resources on shutdown.
        """
        if self.write:
            await self.write.aclose()
        if self.stdio:
            await self.stdio.aclose()
        await self.exit_stack.aclose()

# Main entry point
async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <mcp_server_script.py>")
        sys.exit(1)

    client = MCPClient(ontology_path="./home.owl")
    await client.load_entities()

    try:
        await client.connect(sys.argv[1])
        await client.chat()
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
