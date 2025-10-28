import bcrypt
import jwt
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, status
from app.config.config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS
from app.models.schemas import TokenOut
import secrets

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def generate_random_password() -> str:
    return secrets.token_hex(8) 

def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False

def create_token(sub: str, expires_delta: timedelta, token_type: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": sub,
        "type": token_type,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def create_token_pair(email: str) -> TokenOut:
    access = create_token(email, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES), "access")
    refresh = create_token(email, timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS), "refresh")
    return TokenOut(
        email=email,
        access_token=access,
        refresh_token=refresh,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
