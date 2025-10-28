from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, TypedDict

class SignupIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class OAuthLoginIn(BaseModel):
    email: EmailStr

class TokenOut(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    email : str

class UserOut(BaseModel):
    id: int
    email: EmailStr

class MessageIn(BaseModel):
    content: str = Field(..., max_length=10000)

class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    reasoning: Optional[str] = None
    imageUrl: Optional[str] = None
    time_stamp: str

class ChatOut(BaseModel):
    id: int
    created_at: str
    messages: List[MessageOut] = []

class RefreshIn(BaseModel):
    refresh_token: str


class HistoryMessage(TypedDict):
    role: str
    content: str