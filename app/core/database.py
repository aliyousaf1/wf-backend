from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import MONGO_URI

# Optimized connection settings for MongoDB Atlas M0 (free tier)
client = AsyncIOMotorClient(
    MONGO_URI,
    maxPoolSize=20,              # Enough for concurrent requests
    minPoolSize=3,               # Keep minimum connections ready
    serverSelectionTimeoutMS=5000,  # 5 seconds to select server
    socketTimeoutMS=120000,      # 2 min timeout for bulk ops
)

db = client["whyfashion"]