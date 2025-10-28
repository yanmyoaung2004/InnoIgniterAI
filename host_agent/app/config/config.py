
import os

SECRET_KEY = os.getenv("JWT_SECRET", "secret key")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_MIN", "15"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_DAYS", "7"))

# DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./auth.db")
# CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
# # File: app/config/config.py
DATABASE_URL = "sqlite+aiosqlite:///auth.db"  # Adjust path as needed
CORS_ORIGINS = ["http://localhost:5173"]  # Add your frontend origins
