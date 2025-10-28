import asyncio
import json
import uuid
from typing import List, Dict, Literal
import os

import httpx
from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
)
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from contextvars import ContextVar
from groq import Groq, GroqError
from dotenv import load_dotenv
from fastapi import  WebSocket
from app.remote_agent_connection import RemoteAgentConnections
from app.models.schemas import HistoryMessage
from sqlalchemy.orm import sessionmaker
from app.database.crud import update_chat_title
from app.database.database import  engine
from pydantic import BaseModel

load_dotenv()

llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
image_llm = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL_NAME = "openai/gpt-oss-120b"

class HostAgentState(Dict):
    query: str
    include_reasoning: bool
    file_url: str
    image_url: str
    image_data : str
    messages: List[HistoryMessage]
    response: str
    plan: List[str]
    reasoning: str
    execution_mode: Literal["parallel", "sequence"]
    intermediate_results: Dict[str, str]

class ChatTitle(BaseModel):
    title: str

async def chat_title_generator(content: str) -> ChatTitle:
        """
        Generate a concise structured title for a chat based on its content.
        The title should summarize the main topic in a few words.
        """
        schema = {
            "name": "chat_title",
            "schema": ChatTitle.model_json_schema()
        }

        response = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a short, clear title (max 6 words) "
                        "summarizing the conversation topic."
                    ),
                },
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_schema", "json_schema": schema},
            reasoning_effort="medium",
        )
        result = ChatTitle.model_validate(
            json.loads(response.choices[0].message.content)
        )
        return result
      
class HostAgent:
    def __init__(self):
        self.remote_agent_connections: Dict[str, "RemoteAgentConnections"] = {}
        self.cards: Dict[str, AgentCard] = {}
        self.graph = None
        self.websocket_context = ContextVar("websocket", default=None)

    async def _async_init_components(self, remote_agent_addresses: List[str]):
        async with httpx.AsyncClient(timeout=30) as client:
            for address in remote_agent_addresses:
                card_resolver = A2ACardResolver(client, address)
                try:
                    card = await card_resolver.get_agent_card()
                    remote_connection = RemoteAgentConnections(
                        agent_card=card, agent_url=address
                    )
                    self.remote_agent_connections[card.name] = remote_connection
                    self.cards[card.name] = card
                    print(f"Successfully connected to {card.name} at {address}")
                except Exception as e:
                    print(f"ERROR: Failed to connect to {address}: {e}")

    @classmethod
    async def create(cls, remote_agent_addresses: List[str]):
        instance = cls()
        await instance._async_init_components(remote_agent_addresses)
        instance._build_graph()
        return instance

    def _build_graph(self):
        async def send_message(agent_name: str, task: str) -> str:
            print(f"Sending task to '{agent_name}': '{task[:100]}...'")
            if agent_name not in self.remote_agent_connections:
                return f"Error: Agent '{agent_name}' not found or connection failed."
            
            client = self.remote_agent_connections[agent_name]
            context_id, message_id = str(uuid.uuid4()), str(uuid.uuid4())
            
            payload = {
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": task}],
                    "messageId": message_id,
                    "contextId": context_id,
                },
            }
            message_request = SendMessageRequest(
                id=message_id, params=MessageSendParams.model_validate(payload)
            )
            
            try:
                send_response: SendMessageResponse = await client.send_message(message_request)
                
                if not isinstance(send_response.root, SendMessageSuccessResponse) or not isinstance(send_response.root.result, Task):
                    return "Remote agent returned an invalid response structure."

                response_content = send_response.root.model_dump_json(exclude_none=True)
                json_content = json.loads(response_content)
                
                parts = []
                if artifacts := json_content.get("result", {}).get("artifacts"):
                    for artifact in artifacts:
                        if artifact_parts := artifact.get("parts"):
                            parts.extend(p["text"] for p in artifact_parts if "text" in p)
                
                return "\n".join(parts) if parts else "Agent provided no textual response."

            except Exception as e:
                return f"An error occurred while communicating with {agent_name}: {e}"

        async def read_image(state: HostAgentState) -> HostAgentState:
            print("Processing Image...")
            ws = self.websocket_context.get()
            if ws is not None:
                await ws.send_text(json.dumps({"type": "step", "step": "Processing Image"}))
            image_url = state["image_url"]
            completion = image_llm.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What's in this image?"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }
                        ]
                    }
                ],
                temperature=1,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )
            result = completion.choices[0].message
            query = state['query']
            modified_query = f"""
            --- User Input ---
            {query}

            --- Image Data Extracted ---
            {result}

            Please consider both the user input and the image data when processing this request.
            """
            state["query"] = modified_query
            state["image_data"] = result
            return state

        async def classify_query(state: HostAgentState) -> HostAgentState:
            print("Classifying query...")
            ws = self.websocket_context.get()
            if ws is not None:
                await ws.send_text(json.dumps({"type": "step", "step": "Planning request query"}))
            query = state["query"]
            classification_schema = {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": "Brief reasoning for the chosen plan."},
                    "execution_mode": {
                        "type": "string",
                        "enum": ["parallel", "sequence"],
                        "description": "'sequence' if one agent's output is needed for the next, otherwise 'parallel'."
                    },
                    "agents": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["Knowledge_Agent", "Detection_Agent"]},
                        "description": "List of agents to call. If sequential, this is the execution order."
                    }
                },
                "required": ["reasoning", "execution_mode", "agents"]
            }

            prompt = f"""
                You are a cybersecurity orchestration expert. Your task is to analyze the user's query (which may be in English or Burmese) and decide which agent(s) should handle it.

                Available Agents:
                - Knowledge_Agent: For general cybersecurity questions, explanations of concepts, vulnerabilities, best practices, and normal conversations (default choice if unsure).
                - Detection_Agent: For active tasks like scanning IPs/URLs, checking for phishing, or analyzing files for malware.

                Based on the user query, select the best agent(s) and explain your reasoning for the choice.

                User Query: "{query}"

                Respond with a valid JSON object containing:
                - "reasoning": Your explanation for why you chose the agent(s).
                - "execution_mode": "parallel" or "sequence".
                - "agents": List of agent names.

                Do NOT include instructions, formatting advice, or schema commentary in your reasoning. Only explain your agent selection.
                """
            try:
                response = llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object", "schema": classification_schema},
                    reasoning_effort="medium"
                )
                result_raw = response.choices[0].message.content
                reasoning = getattr(response.choices[0].message, "reasoning", None)
                try:
                    plan_data = json.loads(result_raw)
                    state["plan"] = plan_data.get("agents", ["Knowledge_Agent"])
                    state["execution_mode"] = plan_data.get("execution_mode", "parallel")
                    # state["reasoning"] = reasoning or plan_data.get("reasoning") or "No reasoning provided by LLM."
                except Exception as e:
                    print(f"Error: Classifier returned invalid JSON or missing reasoning: {e}")
                    state["plan"] = ["Knowledge_Agent"]
                    state["execution_mode"] = "parallel"
                    # state["reasoning"] = f"Defaulted due to error: {e}"
                state["intermediate_results"] = {}
                return state
            except GroqError as e:
                print("Error occur", e)
                message = "Sorry, I can’t provide that information. I can, however, help you learn about cybersecurity and safe practices."
                if ws is not None:
                    await ws.send_text(json.dumps({"type": "answer", "data": message}))
                state["reasoning"] = ""
                state["response"] = message
                raise GraphInterrupt("Flow stopped due to disallowed query")
            
        async def parallel_delegate(state: HostAgentState) -> HostAgentState:
            print("--- Delegating in Parallel ---")
            ws = self.websocket_context.get()
            print(state['query'])
            if ws is not None:
                await ws.send_text(json.dumps({"type": "step", "step": "Querying Detection & Knowledge Agents"}))
            agents_to_call = state["plan"]
            tasks = []
            for agent_name in agents_to_call:
                if agent_name == "Detection_Agent":
                    tasks.append(send_message(agent_name, state["query"]))
                else:
                    tasks.append(send_message(agent_name, state["query"]))
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for agent_name, response in zip(agents_to_call, responses):
                if isinstance(response, Exception):
                    state["intermediate_results"][agent_name] = f"Error: {str(response)}"
                else:
                    state["intermediate_results"][agent_name] = response
            return state

        async def call_detection_agent(state: HostAgentState) -> HostAgentState:
            print("--- Calling Detection Agent (Sequence) ---")
            ws = self.websocket_context.get()
            if ws is not None:
                await ws.send_text(json.dumps({"type": "step", "step": "Querying Detection Agent"}))
            print(state['query'])
            response = await send_message("Detection_Agent", state["query"])
            state["intermediate_results"]["Detection_Agent"] = response
            if ws is not None:
                await ws.send_text(json.dumps({"type": "step", "step": "Agent response received"}))
            return state

        async def call_knowledge_agent(state: HostAgentState) -> HostAgentState:
            print("--- Calling Knowledge Agent (Sequence) ---")
            ws = self.websocket_context.get()
            if ws is not None:
                await ws.send_text(json.dumps({"type": "step", "step": "Querying Knowledge Agent"}))
            detection_result = state["intermediate_results"].get("Detection_Agent")    
            if detection_result:
                task = f"Based on the following security scan results, please provide a detailed explanation and recommend mitigation steps:\n\n---\n{detection_result}\n---"
            else:
                task = state["query"]
            response = await send_message("Knowledge_Agent", task)
            state["intermediate_results"]["Knowledge_Agent"] = response
            if ws is not None:
                await ws.send_text(json.dumps({"type": "step", "step": "Agent response received"}))
            return state

        async def summarize_messages(messages: List[dict], model: str) -> str:
            if not messages:
                return ""
            prompt = "Summarize the following conversation concisely:\n\n"
            for msg in messages:
                prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"

            response = llm.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content

        async def prepare_messages_with_summary(messages: List[dict], chatModel: str) -> List[dict]:
            if len(messages) <= 5:
                return messages
            print("Summarizing messages...")
            first_two = messages[:2]
            last_three = messages[-3:]
            middle = messages[2:-3]
            summary_text = await summarize_messages(middle, chatModel)
            new_messages = first_two + [{"role": "system", "content": f"Summary of previous conversation: {summary_text}"}] + last_three
            return new_messages

        async def synthesize_response(state: HostAgentState, config) -> HostAgentState:
            """Combines results from multiple agents into a single, coherent response."""
            print("--- Synthesizing Final Response ---")
            ws = self.websocket_context.get()
            if ws is not None:
                await ws.send_text(json.dumps({"type": "step", "step": "Synthesizing final output"}))
            # Always synthesize results (single or multiple) for consistency
            results_text = ""
            for agent, result in state["intermediate_results"].items():
                results_text += f"--- Result from {agent} ---\n{result}\n\n"

            prompt = f"""You are a helpful cybersecurity assistant.
                Your job is to synthesize the results from multiple specialized agents into a single, clear, and comprehensive response for the user.
                Do not just list the results; integrate them into a helpful answer.

                User's Original Query: "{state['query']}"

                Collected Agent Results:
                {results_text}

                Synthesized Response:
                """
            messages = state["messages"]
            messages.append({"role": "user", "content": prompt})
            message_to_send = await prepare_messages_with_summary(messages, MODEL_NAME)
            
            try:
                stream = llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=message_to_send,
                    reasoning_effort=os.getenv("REASONING_EFFORT"),
                    include_reasoning=state["include_reasoning"],
                    stream=True,
                )
                
                reasoning_text = ""
                answer_text = ""
                reasoning_ended = False
                for chunk in stream: 
                    delta = chunk.choices[0].delta
                    if delta.reasoning and ws is not None:
                        reasoning_text += delta.reasoning
                        try:
                            await ws.send_text(json.dumps({"type": "reasoning", "data": delta.reasoning}))
                        except Exception as e:
                            print(f"WebSocket send failed for reasoning: {e}")
                    if delta.content:
                        if not reasoning_ended:
                            reasoning_ended = True
                        answer_text += delta.content
                        if ws is not None:
                            try:
                                await ws.send_text(json.dumps({"type": "answer", "data": delta.content}))
                            except Exception as e:
                                print(f"WebSocket send failed for answer: {e}")
                    await asyncio.sleep(0) 
                
                state["response"] = answer_text
                state["reasoning"] = reasoning_text
            except Exception as e:
                print(f"Error processing stream: {e}")
                state["response"] = "Error: Failed to process response stream."
                state["reasoning"] = "Error: Failed to process reasoning stream."
                if ws is not None:
                    try:
                        await ws.send_text(json.dumps({"type": "error", "data": str(e)}))
                    except Exception as send_e:
                        print(f"WebSocket send failed for error: {send_e}")
            finally:
                return state
        
        def route_after_classification(state: HostAgentState) -> str:
            print(f"Routing after classification: execution_mode={state['execution_mode']}, plan={state['plan']}")
            if state["execution_mode"] == "parallel":
                return "parallel_delegate"
            elif state["execution_mode"] == "sequence" and state["plan"]:
                first_agent = state["plan"][0]
                return f"call_{first_agent.lower()}"
            else:
                return END
        
        def check_image(state: HostAgentState) -> HostAgentState:
            if state.get("image_url"):
                print(f"Image URL found: {state['image_url']}")
                state["has_image"] = True
            else:
                print("No image URL found.")
                state["has_image"] = False
            return state

        workflow = StateGraph(HostAgentState)
        workflow.add_node("check_image", check_image)
        workflow.add_node("read_image", read_image)
        workflow.add_node("classify", classify_query)
        workflow.add_node("parallel_delegate", parallel_delegate)
        workflow.add_node("call_detection_agent", call_detection_agent)
        workflow.add_node("call_knowledge_agent", call_knowledge_agent)
        workflow.add_node("synthesize_response", synthesize_response)

        workflow.set_entry_point("check_image")

        def route_after_image_check(state: HostAgentState) -> str:
            if state.get("image_url"):
                return "read_image"
            else:
                return "classify"

        workflow.add_conditional_edges(
            "check_image",
            route_after_image_check,
            {
                "read_image": "read_image",
                "classify": "classify",
            }
        )
        workflow.add_edge("read_image", "classify")
        workflow.add_conditional_edges(
            "classify",
            route_after_classification,
            {
                "parallel_delegate": "parallel_delegate",
                "call_detection_agent": "call_detection_agent",
                "call_knowledge_agent": "call_knowledge_agent", 
                END: END,
            },
        )
        
        workflow.add_edge("call_detection_agent", "call_knowledge_agent")
        workflow.add_edge("parallel_delegate", "synthesize_response")
        workflow.add_edge("call_knowledge_agent", "synthesize_response")
        workflow.add_edge("synthesize_response", END)

        self.graph = workflow.compile(checkpointer=MemorySaver())

    async def run(self, ws: WebSocket, query: str, session_id: str = "default", include_reasoning: bool = False, file_url: str = None, image_url: str = None, messages: List[HistoryMessage] = []):
        print(f"Running pipeline for query: {query}, session_id: {session_id}")
        token = self.websocket_context.set(ws)
        try:
            initial_state: HostAgentState = {
                "query": query,
                "include_reasoning": include_reasoning,
                "file_url": file_url,
                "image_url": image_url,
                "image_data": "",
                "messages": messages,
                "chat_id": session_id,
                "response": "",
                "plan": [],
                "reasoning": "",
                "execution_mode": "parallel",
                "intermediate_results": {},
            }
            config = {"configurable": {"thread_id": session_id}}

            trace = []
            final_state = None

            try:
                async for event in self.graph.astream_events(initial_state, config=config):
                    langgraph_node = event.get("metadata", {}).get("langgraph_node")
                    if event["event"] == "on_chain_end" and langgraph_node == "synthesize_response":
                        event_data = event["data"]["output"]
                        
                        messages = event_data.get("messages", [])
                        chat_messages = messages[2:]
                        if 1 <= len(chat_messages) < 4:
                            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
                            db = SessionLocal()
                            chat_content = "\n".join([f"{m['role']}: {m['content']}" for m in chat_messages])
                            chat_title = await chat_title_generator(content=chat_content)
                            await ws.send_text(json.dumps({"type": "title", "title": chat_title.title, 'chatId': session_id}))
                            update_chat_title(db=db, chat_id=session_id, title=chat_title.title)
                        return {
                            "reason": event_data.get("reasoning", ""),
                            "answer": event_data.get("response", "No response generated!")
                        }
            
                    if event["event"] == "on_interrupt":
                        print("Interrupt detected")
                        interrupt: GraphInterrupt = event["data"]
                        print("⚠️ Graph was interrupted!")
                        final_state = interrupt.state
                        return {
                            "reason": final_state.get("reasoning", "Stopped due to disallowed query."),
                            "answer": final_state.get(
                                "final_message",
                                "Sorry, I can’t provide that information. I can, however, help you learn about cybersecurity and safe practices."
                            ),
                        }

                    if event["event"] == "on_step":
                        node, state = event["data"]
                        if isinstance(state, tuple):
                            state = state[0]

                        print(f"[Step: {node}] Query: {state.get('query')} Plan: {state.get('plan')}")
                        trace.append({
                            "step": node,
                            "query": state.get("query"),
                            "plan": state.get("plan"),
                            "execution_mode": state.get("execution_mode"),
                            "reasoning": state.get("reasoning"),
                            "intermediate_results": state.get("intermediate_results"),
                            "response": state.get("response"),
                        })

                        if node in ["synthesize_response", END]:
                            final_state = state
                            print(f"Final state captured at node {node}: {final_state}")

                if final_state:
                    print(f"Pipeline completed: {final_state.get('response')}")
                else:
                    print("Pipeline failed: No final state")
                    return {
                        "reason": "",
                        "answer": "No response generated.",
                    }

            except Exception as e:
                print(f"Unexpected error: {e}")
                return {
                    "reason": "Pipeline crashed unexpectedly.",
                    "answer": "No response generated.",
                }
        finally:
            self.websocket_context.reset(token)
    