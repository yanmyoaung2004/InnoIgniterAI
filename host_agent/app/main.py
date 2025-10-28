# File: app/main.py
import json
import os
import base64
import io
import tempfile
import hashlib
from typing import List
from groq import Groq
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.crud import create_new_chat, add_message_to_chat, get_chat_messages, get_user_chats, get_all_messages, get_chat_by_id, delete_chat, add_image_to_message
from app.database.deps import get_user_id_from_token_dependency, get_current_user
from app.models.models import User, Message, Chat
from app.models.schemas import ChatOut, MessageOut
from app.config.config import CORS_ORIGINS, DATABASE_URL
from app.routes import routes
from app.agent import HostAgent
from app.database.database import Base, engine, SessionLocal
from langchain_mcp_adapters.client import MultiServerMCPClient
from sqlalchemy.orm import selectinload

load_dotenv()

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

agent_urls = [
    "http://localhost:8002",
    "http://localhost:8001",
]

CACHE_DIR = "./tts_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

app = FastAPI(title="InnoIgnitorsAI", version="1.1")
app.state.root_agent = None  # placeholder
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")

# -----------------------
# Startup event
# -----------------------
@app.on_event("startup")
async def startup_event():
    print("Setting the server up ...")
    await init_db()  # Initialize database asynchronously
    app.state.root_agent = await HostAgent.create(remote_agent_addresses=agent_urls)
    print("All agents are initialized!")

# -----------------------
# Middleware
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONT_END_ORIGIN"), "http://103.115.214.70:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Routers
# -----------------------
app.include_router(routes.router)

# -----------------------
# Chat state
# -----------------------
active_chats: dict[int, str] = {}
chat_histories: dict[str, list[dict]] = {}

system_messages = [
    {
        "role": "system",
        "content": """You are InnoIgnitorsAI, developed by InnoIgnitors AI Developer Team. 
        Answer questions in a warm, chatty, and friendly way. 
        Use the provided context to be accurate and helpful."""
    },
    {
        "role": "system",
        "content": """CRUCIAL INSTRUCTION FOR REASONING:
        Only reason about the user's query content itself and its logical analysis."""
    },
]

@app.websocket("/chat")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    root_agent = ws.app.state.root_agent
    try:
        async with SessionLocal() as db:  # Use async session
            while True:
                prompt_json = await ws.receive_text()
                print(f"Received message: {prompt_json}")
                data = json.loads(prompt_json)
                query = data.get("query")
                include_reasoning = data.get("includeReasoning", False)
                chatId = data.get("currentChatId", None)
                fileUrl = data.get("fileUrl", None)
                imageUrl = data.get("imageUrl", None)
                token = data.get("token", None)
                user_id = await get_user_id_from_token_dependency(token, db=db)

                prev_chat = active_chats.get(user_id)

                # Load previous chat history
                if prev_chat != chatId:
                    if chatId and chatId not in chat_histories:
                        chat = await get_chat_by_id(db, chatId)
                        if chat:
                            messages = await get_all_messages(db, chat.id)
                            chat_histories[chatId] = [{"role": m.role, "content": m.content} for m in messages]
                    active_chats[user_id] = chatId

                # Handle new chat
                if chatId is None:
                    if token is None:
                        print("No account provided, skipping agent call.")
                        await ws.send_text(json.dumps({"type": "error", "data": "Please provide a token to start a chat."}))
                        continue
                    chat = await create_new_chat(db, user_id)
                    await ws.send_text(json.dumps({
                        "type": "new_chat",
                        "id": chat.id,
                        "unique_id": chat.unique_id,
                        "title": chat.title,
                    }))
                    chat_histories[chat.unique_id] = system_messages.copy()
                    active_chats[user_id] = chat.unique_id
                    await add_message_to_chat(db=db, chat_id=chat.id, role="user", content=query, reason=None)
                    if imageUrl:
                        await add_image_to_message(db=db, message_id=chat.id, image_url=imageUrl)
                    if fileUrl:
                        query += (
                            " Please use the 'detect_file_from_url' tool with the following JSON input. "
                            "Do not reveal the URL, token, or filename in your response; "
                            "refer only to it as 'the uploaded file'.\n"
                            f"{{'file_download_url': '{fileUrl}'}}"
                        )

                    result = await root_agent.run(
                        ws=ws,
                        query=query,
                        session_id=chat.unique_id,
                        include_reasoning=include_reasoning,
                        file_url=fileUrl,
                        image_url=imageUrl,
                        messages=chat_histories[chat.unique_id]
                    )
                    await add_message_to_chat(db=db, chat_id=chat.id, role="assistant", content=result["answer"], reason=result.get("reason"))
                    chat_histories[chat.unique_id].append({"role": "assistant", "content": result["answer"]})

                # Handle existing chat
                else:
                    chat = await get_chat_by_id(db, chatId)
                    if not chat:
                        await ws.send_text(json.dumps({"type": "error", "data": "Chat not found."}))
                        continue
                    await add_message_to_chat(db=db, chat_id=chat.id, role="user", content=query, reason=None)
                    if imageUrl:
                        await add_image_to_message(db=db, message_id=chat.id, image_url=imageUrl)
                    result = await root_agent.run(
                        ws=ws,
                        query=query,
                        session_id=chatId,
                        include_reasoning=include_reasoning,
                        file_url=fileUrl,
                        image_url=imageUrl,
                        messages=chat_histories.get(chatId, system_messages.copy())
                    )
                    await add_message_to_chat(db=db, chat_id=chat.id, role="assistant", content=result["answer"], reason=result.get("reason"))
                    chat_histories.setdefault(chatId, system_messages.copy()).append({"role": "assistant", "content": result["answer"]})

                await ws.send_text(json.dumps({"type": "done"}))
                print("Sent 'done' signal")

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        await ws.send_text(json.dumps({"type": "error", "data": f"Server error: {str(e)}"}))



async def get_chat_messages(db: AsyncSession, chat_id: int) -> list[Message]:
    """
    Fetch all messages for a chat, eagerly loading images to avoid async lazy-loading errors.
    """
    result = await db.execute(
        select(Message)
        .options(selectinload(Message.images))  # eagerly load images
        .where(Message.chat_id == chat_id)
        .order_by(Message.time_stamp)
    )
    messages = result.scalars().all()
    return messages


@app.get("/chats/{chat_id}", response_model=ChatOut)
async def get_chat(chat_id: str, current_user: User = Depends(get_current_user)):
    async with SessionLocal() as db:
        # 1️⃣ Fetch the chat belonging to current user
        result = await db.execute(
            select(Chat).where(Chat.unique_id == chat_id, Chat.user_id == current_user.id)
        )
        chat = result.scalar_one_or_none()
        if not chat:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

        # 2️⃣ Fetch messages safely
        messages = await get_chat_messages(db, chat.id)

        # 3️⃣ Transform messages for response
        messages_out = [
            MessageOut(
                id=m.id,
                role=m.role,
                content=m.content,
                reasoning=m.reason,
                imageUrl=m.images[0].image_url if m.images else None,
                time_stamp=str(m.time_stamp)
            )
            for m in messages
        ]

        # 4️⃣ Return the chat
        return ChatOut(
            id=chat.id,
            created_at=str(chat.created_at),
            messages=messages_out
        )

@app.delete("/chat/delete/{chat_id}", response_model=ChatOut)
async def delete_chat_endpoint(chat_id: str, current_user: User = Depends(get_current_user)):
    async with SessionLocal() as db:
        result = await db.execute(
            select(Chat).where(Chat.unique_id == chat_id, Chat.user_id == current_user.id)
        )
        chat = result.scalar_one_or_none()
        if not chat:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
        success = await delete_chat(db, chat_id)
        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"success": True, "message": "Chat deleted successfully"},
        )

@app.get("/chats")
async def list_chats(current_user: User = Depends(get_current_user)):
    async with SessionLocal() as db:
        chats = await get_user_chats(db, current_user.id)
        result = []
        for chat in chats:
            result_last_msg = await db.execute(
                select(Message)
                .where(Message.chat_id == chat.id)
                .order_by(Message.time_stamp.desc())
                .limit(1)
            )
            last_msg = result_last_msg.scalar_one_or_none()
            message_count = (await db.execute(
                select(func.count(Message.id)).where(Message.chat_id == chat.id)
            )).scalar_one()
            result.append({
                "id": chat.id,
                "title": chat.title,
                "unique_id": chat.unique_id,
                "lastMessage": last_msg.content if last_msg else "",
                "timestamp": last_msg.time_stamp.isoformat() if last_msg else chat.created_at.isoformat(),
                "messageCount": message_count
            })
        return result

mcp_client = MultiServerMCPClient({
    "file_server": {"url": MCP_SERVER_URL, "transport": "streamable_http"}
})

async def upload_to_mcp(file_bytes: bytes, filename: str):
    tools = await mcp_client.get_tools(server_name="file_server")
    upload_tool = next(t for t in tools if t.name == "upload_file")

    file_base64 = base64.b64encode(file_bytes).decode("utf-8")

    result_json = await upload_tool.ainvoke({
        "filename": filename,
        "content_base64": file_base64
    })
    return json.loads(result_json)

def get_cache_path(text: str) -> str:
    """Generate a hash filename for the given text."""
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{text_hash}.wav")

tts_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
@app.post("/tts")
async def text_to_speech(payload: dict):
    text = payload.get("message", "").strip()
    if not text:
        return {"error": "No text provided"}

    cache_path = get_cache_path(text)
    if os.path.exists(cache_path):
        print("there is data")
        with open(cache_path, "rb") as f:
            audio_bytes = io.BytesIO(f.read())
        audio_bytes.seek(0)
        return StreamingResponse(audio_bytes, media_type="audio/wav")
    
    print("no data")

    response = tts_client.audio.speech.create(
        model="playai-tts",
        voice="Fritz-PlayAI",
        input=text,
        response_format="wav"
    )

    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", dir=CACHE_DIR, delete=False)
    tmp_file.close()
    response.write_to_file(tmp_file.name)

    os.replace(tmp_file.name, cache_path)

    with open(cache_path, "rb") as f:
        audio_bytes = io.BytesIO(f.read())
    audio_bytes.seek(0)

    return StreamingResponse(audio_bytes, media_type="audio/wav")