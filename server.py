
# This file sets up an MCP server that exposes ontology-related tools
# for querying and interacting with an OWL knowledge graph.

# Import libraries
import httpx
from mcp.server.fastmcp import FastMCP
import requests
from typing import Any


# Create MCP server instance with the name "cowl"
mcp = FastMCP("cowl") 


#Configuration
KG_BASE = "http://localhost:49201" # Base URL of the knowledge graph server
GRAPH = "home"  # Graph name (can be changed to load another ontology)
ONTOLOGY_PATH = "/home/stefanodileo/flc/home.owl" # Path to OWL ontology file
HEADERS = {"Accept": "application/json"} # Default HTTP headers for API calls


# Helper function to perform async GET requests and return JSON responses
async def get_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=HEADERS, params=params, timeout=20.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[ERROR] GET {url} failed: {e}")
            return None


# TOOLS (Ontology-agnostic functions exposed to MCP client)
@mcp.tool(
    name="get_all_entities",
    description="Retrieve all OWL entities in the ontology."
)
async def get_all_entities(graph_name: str) -> Any:
    # Always override graph name with configured GRAPH
    graph_name = GRAPH
    url = f"{KG_BASE}/{graph_name}/entities"
    return await get_json(url)


@mcp.tool(
    name="get_entities_by_type",
    description="Retrieve OWL entities filtered by their meta-type (e.g., NamedIndividual, Class, DataPropertyAssertion, ObjectPropertyAssertion)."
)
async def get_entities_by_type(type: str, graph_name: str) -> Any:
    graph_name = GRAPH
    url = f"{KG_BASE}/{graph_name}/entities"
    return await get_json(url, {"type": type})


@mcp.tool(
    name="get_all_axioms",
    description="Retrieve all axioms (facts, subclass relations, property assertions) in the ontology."
)
async def get_all_axioms(graph_name: str) -> Any:
    graph_name = GRAPH
    url = f"{KG_BASE}/{graph_name}"
    return await get_json(url)


@mcp.tool(
    name="get_axioms_by_type",
    description=(
        "Retrieve axioms filtered by type (e.g., DataPropertyAssertion, SubClassOf, ObjectPropertyAssertion, ...). "
        "DataPropertyAssertion useful to retrieve settings, readings, values and states/status of single, multiple or all instances, "
        "while ObjectPropertyAssertion to retrieve location of objects."
    )
)
async def get_axioms_by_type(axiom_type: str, graph_name: str) -> Any:
    graph_name = GRAPH
    url = f"{KG_BASE}/{graph_name}"
    return await get_json(url, {"type": axiom_type})


@mcp.tool(
    name="get_axioms_by_entity",
    description=(
        "Retrieve axioms involving a given and unique OWL entity. "
        "entity_type must be 'Class', 'NamedIndividual', ... , "
        "and iri must start with ':' (e.g., ':Entity')."
    )
)
async def get_axioms_by_entity(entity_type: str, iri: str, graph_name: str) -> Any:
    graph_name = GRAPH
    rany = f"{entity_type}({iri})"
    url = f"{KG_BASE}/{graph_name}?rany={rany}"
    return await get_json(url)


@mcp.tool(
    name="get_axioms_by_entity_and_type",
    description=(
        "Retrieve axioms of a given type involving a specific OWL entity. "
        "entity_type must be 'Class', 'NamedIndividual', ... , "
        "and iri must start with ':' (e.g., ':Entity')."
    )
)
async def get_axioms_by_entity_and_type(entity_type: str, iri: str, axiom_type: str, graph_name: str) -> Any:
    graph_name = GRAPH
    rany = f"{entity_type}({iri})"
    url = f"{KG_BASE}/{graph_name}?rany={rany}&type={axiom_type}"
    return await get_json(url)


# Load ontology graph
def load_graph():
    """
    Loads the ontology file from disk into the knowledge graph server.
    This must succeed before the MCP server is started.
    """
    try:
        with open(ONTOLOGY_PATH, "rb") as f:
            data = f.read()
        resp = requests.put(
            f"{KG_BASE}/{GRAPH}",
            data=data,
            headers={
                "Content-Type": "application/rdf+xml",
                "Accept": "application/json"
            }
        )
        resp.raise_for_status()
        print(f"[INFO] Loaded graph '{GRAPH}' from {ONTOLOGY_PATH}")
    except Exception as e:
        print(f"[FATAL] Failed to load ontology into graph '{GRAPH}': {e}")
        exit(1)


if __name__ == "__main__":
    load_graph()
    # Start MCP server over stdio
    mcp.run(transport="stdio")
