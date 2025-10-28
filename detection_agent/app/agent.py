import base64
from collections.abc import AsyncIterable
from typing import Any, List, Literal
from pathlib import Path
import os
import requests
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from app.tools.detection_tool import DetectionTool
from dotenv import load_dotenv

load_dotenv()

# Initialize memory and detection tool
memory = MemorySaver()
detection_tool = DetectionTool()

# ---------------- Pydantic Schemas ----------------
class UrlDetectionInput(BaseModel):
    url: str = Field(..., description="The URL to analyze for potential threats.")

class FileDetectionInput(BaseModel):
    file_path: str = Field(..., description="The file path of the file to analyze for potential threats.")

class FileDownloadURLInput(BaseModel):
    file_download_url: str = Field(..., description="The URL to download the file from.")

class MailDetectionInput(BaseModel):
    email_content: str = Field(..., description="The content of the email to analyze for potential threats.")

class ClarificationInput(BaseModel):
    question: str = Field(description="The clarifying question to ask the user.")
    options: List[str] = Field(description="A list of options for the user to choose from, if applicable.")

# ---------------- Tools ----------------
@tool(args_schema=UrlDetectionInput)
def detect_url(url: str) -> str:
    """Use this tool ONLY to analyze URLs for potential threats."""
    result = detection_tool.detect_url(url)
    if not result:
        return f"No threat detected or analysis failed for URL '{url}'."
    return f"URL Analysis Result for '{url}': {result}"

@tool(args_schema=MailDetectionInput)
def detect_email(email_content: str) -> str:
    """Use this tool ONLY to analyze emails for potential threats."""
    result = detection_tool.detect_email(email_content)
    if not result:
        return f"No threat detected or analysis failed for email content."
    return f"Email Analysis Result: {result}"

@tool(args_schema=ClarificationInput)
def request_user_clarification(question: str, options: List[str] = None) -> str:
    """Pause workflow to ask user for clarification."""
    return f"Paused to ask user for clarification: '{question}' Options: {options or 'N/A'}"

# ---------------- File Download Helper ----------------
def download_file_base64(download_url: str, save_dir: str = "download") -> str:
    """Download a base64-encoded file from a URL and return the local file path."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Attempting to download file from: {download_url}")
        response = requests.get(download_url, timeout=10)
        print(f"Response status code: {response.status_code}")
        if response.status_code == 200:
            filename = os.path.basename(download_url)
            file_path = os.path.join(save_dir, filename)
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"File successfully downloaded to: {file_path}")
            return file_path
        else:
            raise Exception(f"Failed to download file: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Error in download_file_base64: {str(e)}")
        raise
    
@tool(args_schema=FileDownloadURLInput)
def detect_file_from_url(file_download_url: str) -> str:
    """
    Download a file (if URL) or use local file path and analyze it for threats.
    Returns the local file path with analysis result.
    """
    try:
        print("detect_file_from url")
        if file_download_url.startswith("http://") or file_download_url.startswith("https://"):
            local_file_path = download_file_base64(file_download_url)
        else:
            local_file_path = file_download_url 
        print(f"Local file path: {local_file_path}")
        result = detection_tool.detect_file(local_file_path)
        if not result:
            return f"No threat detected or analysis failed for file '{local_file_path}'."
        return f"File Analysis Result for '{local_file_path}': {result}"

    except Exception as e:
        return f"Failed to download or analyze file: {str(e)}"

def final_response_tool(status: str, message: str) -> dict:
    """The final response to the user."""
    return {"status": status, "message": message}

class ResponseFormat(BaseModel):
    status: Literal["input_required", "completed", "error"]
    message: str

# ---------------- Detection Agent ----------------
class DetectionAgent:
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]    
    SYSTEM_INSTRUCTION = (
        "You are a cybersecurity detection expert assistant with a friendly, funny, and approachable personality 😎💻. "
        "Your mission is to detect and analyze potentially malicious content (URLs, files, emails) clearly and thoroughly, "
        "making complex threat analysis understandable 🧠✨. Use humor and relatable examples where appropriate to keep the conversation engaging 😂👍.\n\n"
        "--- CORE DIRECTIVES ---\n"
        "1.  **Analyze the Input:** First, figure out what the user has given you:\n"
        "    - If it's a link → treat it as a URL.\n"
        "    - If it's a file (uploaded or shared via a link) → treat it as a file.\n"
        "    - If it's a raw email or email headers → treat it as an email.\n\n"
        "2.  **Tool Usage:** You have access to the following detection tools:\n"
        "    - detect_url(url): Scan a suspicious URL to check if it’s linked to malware, phishing, scams, or malicious payloads.\n"
        "    - detect_file_from_url(file_url): Download and scan a file from a given URL to detect malware, ransomware, or trojans.\n"
        "    - detect_email(email_content): Analyze email text, headers, and attachments for phishing, scams, or malware indicators.\n\n"
        "3.  **Detection Rules:**\n"
        "    - Always use the appropriate tool to gather verified threat intelligence (no guessing!).\n"
        "    - If results show multiple possible threats or ambiguous findings, ask the user for clarification before final judgment.\n"
        "    - Clearly classify threats (e.g., phishing, trojan, ransomware, PUPs) and rate severity (low, medium, high, critical).\n\n"
        "4.  **Explain Findings Clearly:**\n"
        "    - Summarize the detection results in plain English (no overly technical jargon).\n"
        "    - Use fun analogies or humor (like 'this file is sneakier than a ninja with WiFi') while keeping it accurate.\n"
        "    - Provide structured output: Threat Type, Severity, Indicators, and Recommended Next Steps.\n\n"
        "5.  **Security Advice:**\n"
        "    - Suggest practical next steps (block URL, delete email, quarantine file, run AV scan, etc.).\n"
        "    - Share general cybersecurity hygiene tips, but do not speculate beyond verified data.\n\n"
        "6.  **Final Answer:**\n"
        "    - Deliver a comprehensive, user-friendly detection report.\n"
        "    - Keep it approachable, a little funny, but very clear so even a non-tech user understands what’s going on."
    )

    def __init__(self):
        self.model = ChatGroq(model_name=os.getenv('LLM_MODEL'), temperature=0.7)
        self.tools = [
            request_user_clarification,
            detect_url,
            detect_file_from_url,
            detect_email
        ]    
        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
        )

    async def stream(self, query: str, context_id: str) -> AsyncIterable[dict[str, Any]]:
        # Add file URL automatically if included in the query
        print("Detection agent start running")
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
