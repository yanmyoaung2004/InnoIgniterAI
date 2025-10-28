# File: app/routes/routes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.schemas import SignupIn, LoginIn, TokenOut, UserOut, RefreshIn, OAuthLoginIn
from app.models.models import User
from app.utils.utils import hash_password, verify_password, create_token_pair, decode_token, generate_random_password
from app.database.deps import get_db

router = APIRouter(prefix="/auth", tags=[""])

@router.post("/signup", response_model=UserOut)
async def signup(body: SignupIn, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email))
    existing = result.scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    u = User(email=body.email, password_hash=hash_password(body.password))
    db.add(u)
    await db.commit()
    await db.refresh(u)
    return UserOut(id=u.id, email=u.email)

@router.post("/login", response_model=TokenOut)
async def login(body: LoginIn, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return create_token_pair(user.email)

@router.post("/oauth", response_model=TokenOut)
async def oauth_login(body: OAuthLoginIn, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if not user:
        random_password = generate_random_password()
        hashed_password = hash_password(random_password)
        user = User(
            email=body.email,
            password_hash=hashed_password
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
    return create_token_pair(user.email)

@router.post("/refresh", response_model=TokenOut)
async def refresh_tokens(body: RefreshIn, db: AsyncSession = Depends(get_db)):
    payload = decode_token(body.refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Wrong token type")
    email = payload.get("sub")
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return create_token_pair(email)