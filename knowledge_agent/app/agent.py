from collections.abc import AsyncIterable
import os
from typing import Any, List, Literal
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from app.tools.cve_tool import CVETool
from app.tools.mitre_tool import SafeMitreTool
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
# from app.tools.mcp_tool import MCPFileClient
# from app.rag_db import search_db


memory = MemorySaver()
websearch_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
cve_tool = CVETool()
mitre_tool = SafeMitreTool()

class CVESearchInput(BaseModel):
    keyword: str = Field(..., description="Keyword to search for CVEs (e.g., software or vulnerability type)")
    limit: int = Field(10, description="Maximum number of CVEs to return")

class CVEDetailsInput(BaseModel):
    cve_id: str = Field(..., description="CVE ID to fetch details for (e.g., CVE-2023-1234)")

class ClarificationInput(BaseModel):
    question: str = Field(description="The clarifying question to ask the user.")
    options: List[str] = Field(description="A list of options for the user to choose from, if applicable.")

class FileReadInput(BaseModel):
    path: str = Field(..., description="The path of the file to read.")

class MitreSearchInput(BaseModel):
    keyword: str

class MitreTechIDInput(BaseModel):
    attack_id: str

class MitreMitigationKeywordInput(BaseModel):
    keyword: str

class MitreMitigationIDInput(BaseModel):
    mitigation_id: str

class WebsearchInput(BaseModel):
    query: str

@tool(args_schema=MitreMitigationIDInput)
def get_techniques_for_mitigation(mitigation_id: str) -> str:
    """
    Retrieve MITRE ATT&CK techniques that are mitigated by a specific mitigation.

    Args:
        mitigation_id (str): The ID of the mitigation (course-of-action).

    Returns:
        str: A formatted string listing all techniques mitigated by the given mitigation.
    """
    results = mitre_tool.get_techniques_for_mitigation(mitigation_id)
    if not results:
        return f"No techniques found for mitigation ID '{mitigation_id}'."

    output = []
    for t in results:
        output.append(f"[{t['domain']}] {t['technique_id']}: {t['technique']}\nDescription: {t['description']}")
    return "\n\n".join(output)

@tool(args_schema=MitreMitigationKeywordInput)
def get_mitigations_by_keyword(keyword: str) -> str:
    """
    Search for MITRE ATT&CK mitigations by a keyword in associated technique names or descriptions.

    Args:
        keyword (str): The search keyword.

    Returns:
        str: A formatted string of techniques and their mitigations matching the keyword.
    """
    results = mitre_tool.get_mitigations_by_keyword(keyword)
    if not results:
        return f"No mitigations found for keyword '{keyword}'."

    output = []
    for t in results:
        output.append(f"[{t['domain']}] {t['technique_id']}: {t['technique']}")
        for m in t["mitigations"]:
            output.append(f"  - {m['mitigation_id']}: {m['name']}\n    Description: {m['description']}")
    return "\n".join(output)

@tool(args_schema=MitreTechIDInput)
def get_technique_by_id(attack_id: str) -> str:
    """
    Retrieve a MITRE ATT&CK technique by its attack ID.

    Args:
        attack_id (str): The attack ID of the technique (e.g., T1234).

    Returns:
        str: A formatted string with technique details or a message if not found.
    """
    t = mitre_tool.get_technique_by_id(attack_id)
    if not t:
        return f"No technique found for ID '{attack_id}'."
    return f"[{t['domain']}] {t['tech_id']}: {t['technique']}\nDescription: {t['description']}"

@tool(args_schema=MitreTechIDInput)
def get_mitigations_for_technique(attack_id: str) -> str:
    """
    Retrieve mitigations associated with a specific MITRE ATT&CK technique.

    Args:
        attack_id (str): The attack ID of the technique.

    Returns:
        str: A formatted string listing all mitigations for the technique or a message if none found.
    """
    t = mitre_tool.get_mitigations_for_technique(attack_id)
    if not t or not t.get("mitigations"):
        return f"No mitigations found for technique ID '{attack_id}'."

    output = [f"[{t['domain']}] {t['technique_id']}: {t['technique']}"]
    for m in t["mitigations"]:
        output.append(f"  - {m['mitigation_id']}: {m['name']}\n    Description: {m['description']}")
    return "\n".join(output)

@tool(args_schema=MitreSearchInput)
def search_mitre_techniques(keyword: str) -> str:
    """
    Search MITRE ATT&CK techniques by a keyword.

    Args:
        keyword (str): The search keyword for technique names or descriptions.

    Returns:
        str: A formatted string listing matching techniques with domain, ID, and description.
    """
    results = mitre_tool.search_techniques(keyword)
    if not results:
        return f"No MITRE techniques found for keyword '{keyword}'."

    output = []
    for r in results:
        output.append(
            f"[{r['domain']}] {r['tech_id']}: {r['technique']}\nDescription: {r['description']}"
        )
    return "\n\n".join(output)

@tool(args_schema=CVESearchInput)
def search_cves(keyword: str, limit: int = 5) -> str: 
    """Use this tool ONLY to search for specific cybersecurity CVEs..."""
    results = cve_tool.search_cves(keyword, limit)
    if not results:
        return f"No CVEs found for keyword '{keyword}'."
    return "\n".join([f"{v['cve']['id']}: {v['cve']['descriptions'][0]['value']}" for v in results])

@tool(args_schema=CVEDetailsInput)
def get_cve_details(cve_id: str) -> str:
    """Use this tool ONLY to get detailed information about a specific cybersecurity CVE ID..."""
    details = cve_tool.get_cve_details(cve_id)
    if not details:
        return f"No details found for CVE ID '{cve_id}'."
    cve_info = details.get("cve", {})
    desc = cve_info.get("descriptions", [{}])[0].get("value", "No description available.")
    return f"CVE ID: {cve_id}\nDescription: {desc}\nReferences: {', '.join([r.get('url','') for r in cve_info.get('references', [])])}"

@tool(args_schema=CVEDetailsInput)
def get_patch_info(cve_id: str) -> str:
    """Use this tool ONLY to find patch or mitigation info for a specific cybersecurity CVE ID..."""
    patch_info = cve_tool.get_patch_info(cve_id)
    if patch_info["patches"]:
        return f"Patches for {cve_id}:\n" + "\n".join(patch_info["patches"])
    return patch_info["message"]

@tool
def get_myanmar_cyber_law() -> str:
    """
    Retrieve the Myanmar Cybersecurity Law from a local file.

    Returns:
        str: The full content of the law or an error message if the file cannot be read.
    """
    print("Myanmar law tool called")
    file_path = "app/data/myanmar-electronic-policy.txt" 
    try:
        if not os.path.exists(file_path):
            return f"Law file not found at: {file_path}"

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            return f"The law file at '{file_path}' is empty."
        return content
    except Exception as e:
        return f"Error reading law file: {str(e)}"

@tool(args_schema=WebsearchInput)    
def web_search(query: str) -> str:
    """
    Perform a web search using an external API.

    Args:
        query (str): The search query.

    Returns:
        str: A summary of the search results or an error message.
    """
    print("Web search tool called")
    response = websearch_client.chat.completions.create(
            model="compound-beta",
            messages=[
                {
                    "role": "user",
                    "content": query
                }
            ]
        )
    try:
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error occurred: {e}")
        return "Error occurred while fetching web search results."

@tool(args_schema=ClarificationInput)
def request_user_clarification(question: str, options: List[str] = None) -> str:
    """
    Use this tool ONLY when you are unsure how to proceed and need to ask the user a clarifying question.
    This will pause your work until the user responds.
    """
    return f"Paused to ask user for clarification: '{question}' Options: {options or 'N/A'}"

@tool(args_schema=FileReadInput)
def read_file(path: str) -> str:
    """Read a file's content using the MCP file server."""
    return "hey there this is read file"

def final_response_tool(status: str, message: str) -> dict:
    """The final response to the user."""
    return {"status": status, "message": message}

class ResponseFormat(BaseModel):
    """The final response to the user."""
    status: Literal["input_required", "completed", "error"]
    message: str


class KnowledgeAgent:
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]
    
    SYSTEM_INSTRUCTION = (
        "You are a cybersecurity expert assistant with a friendly, funny, and approachable personality 😎💻. "
        "Your mission is to explain cybersecurity topics clearly and thoroughly, making complex concepts easy to understand 🧠✨. "
        "Use humor and relatable examples where appropriate to keep the conversation engaging 😂👍.\n\n"
        "--- CORE DIRECTIVES ---\n"
        "1.  **Analyze the Request:** First, determine if the user's query is about a specific cybersecurity topic, a legal inquiry, or a general knowledge question.\n\n"
        "2.  **Tool Usage:** You have access to the following tools:\n"
        "    - web_search(query): Use this to get up-to-date information from the web (for current events, 2025+ information, or anything beyond your knowledge cutoff).\n"
        "    - search_cves(keyword): Use this to find CVEs.\n"
        "    - get_cve_details(cve_id): Get detailed info about a CVE.\n"
        "    - get_patch_info(cve_id): Get patch information for a CVE.\n"
        "    - search_mitre_techniques(keyword): Search MITRE ATT&CK techniques.\n"
        "    - get_technique_by_id(tech_id): Retrieve a MITRE technique by ID.\n"
        "    - get_mitigations_for_technique(tech_id): Get mitigations for a technique.\n"
        "    - get_mitigations_by_keyword(keyword): Find mitigations by keyword.\n"
        "    - get_techniques_for_mitigation(mitigation_id): Find techniques mitigated by an ID.\n"
        "    - get_myanmar_cyber_law(topic): Fetch Myanmar cybersecurity law information.\n"
        "    - request_user_clarification(question): Ask the user to clarify ambiguous queries.\n\n"
        "3.  **Cybersecurity Queries:** If the query is about vulnerabilities, threats, software patches, or specific CVEs, you **MUST** use the specialized tools. "
        "If a tool returns multiple results for an ambiguous query (like 'log4j'), use `request_user_clarification` to ask which one the user is interested in before proceeding.\n\n"
        "4.  **Legal Queries:** For questions about Myanmar cybersecurity law or regulations, use `get_myanmar_cyber_law` to provide accurate, up-to-date information.\n\n"
        "5.  **Current or Future Events:** For events or data beyond your knowledge cutoff (2024+), you **MUST** use `web_search` to fetch the latest information.\n\n"
        "6.  **General Knowledge Queries:** If the query is not cybersecurity or legal related, answer directly from your own knowledge.\n\n"
        "7.  **Final Answer:** Once you have gathered all necessary information from the appropriate tools, provide a comprehensive, well-formatted, and clear answer. "
        "Cite the tools you used where relevant."
    )

    def __init__(self):
        self.model = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.7)
        self.tools = [
            request_user_clarification,
            search_cves, 
            get_cve_details, 
            get_patch_info, 
            search_mitre_techniques,
            get_technique_by_id,
            get_mitigations_for_technique,
            get_mitigations_by_keyword,
            get_techniques_for_mitigation,
            web_search,
            get_myanmar_cyber_law
            
            # read_file
        ]    
        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
        )

    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        # retrieved_contexts = search_db(query, top_k=3)
        retrieved_contexts = []
        if retrieved_contexts:
            context_text = "\n\n".join(retrieved_contexts)
            augmented_query = (
                f"User query: {query}\n\n"
                f"--- Retrieved context from knowledge base ---\n"
                f"{context_text}\n"
                f"--- End of retrieved context ---\n"
                "Use the above context to answer the user's question as accurately as possible."
            )
        else:
            augmented_query = f"User query: {query}"
        inputs = {"messages": [("user", augmented_query)]}
        config: RunnableConfig = {"configurable": {"thread_id": context_id}}
        async for item in self.graph.astream(inputs, config, stream_mode="values"):
            message = item["messages"][-1]
            if isinstance(message, AIMessage) and message.tool_calls and len(message.tool_calls) > 0:
                yield {"is_task_complete": False, "require_user_input": False, "content": f"Thinking... Using tool {message.tool_calls[0]['name']}...", "full_item": item}
            elif isinstance(message, ToolMessage):
                yield {"is_task_complete": False, "require_user_input": False, "content": f"Processing results from {message.name}...", "full_item": item}
        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)
        messages = current_state.values.get("messages", [])
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not message.tool_calls:
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": message.content,
                }
        return {
            "is_task_complete": True,
            "require_user_input": False,
            "content": "The agent finished, but the final response could not be parsed.",
        }
