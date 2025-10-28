# File: crud.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from datetime import datetime
from app.models.models import Chat, Message, Image

async def create_new_chat(db: AsyncSession, user_id: int) -> Chat:
    chat = Chat(user_id=user_id, created_at=datetime.utcnow())
    db.add(chat)
    await db.commit()
    await db.refresh(chat)
    return chat

async def update_chat_title(db: AsyncSession, chat_id: int, title: str) -> Chat | None:
    """
    Update the title of a chat.
    Returns the updated Chat object, or None if chat not found.
    """
    result = await db.execute(select(Chat).where(Chat.unique_id == chat_id))
    chat = result.scalar_one_or_none()
    if not chat:
        return None

    chat.title = title
    db.add(chat)
    await db.commit()
    await db.refresh(chat)
    return chat

async def add_message_to_chat(db: AsyncSession, chat_id: int, role: str, content: str, reason: str) -> Message:
    message = Message(chat_id=chat_id, role=role, content=content, reason=reason, time_stamp=datetime.utcnow())
    db.add(message)
    await db.commit()
    await db.refresh(message)
    return message

async def add_image_to_message(db: AsyncSession, message_id: int, image_url: str) -> Image:
    image = Image(message_id=message_id, image_url=image_url, time_stamp=datetime.utcnow())
    db.add(image)
    await db.commit()
    await db.refresh(image)
    return image

async def get_all_messages(db: AsyncSession, chat_id: int) -> list[Message]:
    result = await db.execute(
        select(Message)
        .where(Message.chat_id == chat_id)
        .order_by(Message.time_stamp.asc())
    )
    return result.scalars().all()

async def get_chat_by_id(db: AsyncSession, chat_id: int) -> Chat | None:
    result = await db.execute(select(Chat).where(Chat.unique_id == chat_id))
    return result.scalar_one_or_none()

async def get_chat_messages(db: AsyncSession, chat_id: int) -> list[Message]:
    result = await db.execute(
        select(Message)
        .options(joinedload(Message.images))
        .where(Message.chat_id == chat_id)
        .order_by(Message.time_stamp.asc())
    )
    return result.scalars().all()

async def get_user_chats(db: AsyncSession, user_id: int) -> list[Chat]:
    result = await db.execute(
        select(Chat)
        .where(Chat.user_id == user_id)
        .order_by(Chat.created_at.desc())
    )
    return result.scalars().all()

async def delete_chat(db: AsyncSession, chat_id: int) -> bool:
    """
    Delete a chat by unique_id. All related messages will be deleted automatically
    because of cascade settings.
    
    Returns True if deleted, False if not found.
    """
    result = await db.execute(select(Chat).where(Chat.unique_id == chat_id))
    chat = result.scalar_one_or_none()
    if not chat:
        return False

    await db.delete(chat)
    await db.commit()
    return True