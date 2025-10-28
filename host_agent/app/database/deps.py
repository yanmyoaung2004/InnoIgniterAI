from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional
from app.database.database import SessionLocal
from app.models.models import User
from app.utils.utils import decode_token

async def get_db():
    async with SessionLocal() as db:
        yield db

async def get_current_user(authorization: Optional[str] = Header(default=None), db: AsyncSession = Depends(get_db)) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    if payload.get("type") != "access":

        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Wrong token type")
    email = payload.get("sub")
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User no longer exists")
    return user

async def get_user_id_from_token_dependency(
    token: str,
    db: AsyncSession = Depends(get_db)
) -> int:
    """
    FastAPI dependency: extracts user ID from the JWT token in the Authorization header.
    Automatically gets a DB session from get_db().
    """
    try:
        payload = decode_token(token)
    except HTTPException:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    
    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing subject")
    
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User no longer exists")
    return user.id