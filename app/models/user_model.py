import time
from pydantic import BaseModel, Field
from typing import List

class UserModel(BaseModel):
    email: str
    phone_number: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    dob: str | None = None
    onboarded: bool = Field(default=False)
    email_verified: bool = Field(default=False)
    favorite_brands: List = []
    sizes: dict = {}
    gender: str | None = None
    created_at: int = Field(default_factory= lambda : int(time.time()))
    likes: List = []
    dislikes: List =[]
    productsViewed: List =[]
    fcm_tokens: List =[]
    deleted: bool = Field(default=False)
    deletedAt: int | None = None
    login_method: str | None = None
    last_seen_announcements_at: int = Field(default_factory= lambda : int(time.time()))
    notification_preferences: dict = Field(
    default_factory=lambda: {
        "appNotificationsMain": True,
        "appSpecialOffers": True,
        "appNewExclusive": True,
        "appStockAlerts": True,
        "emailNotificationsMain": True,
        "emailSpecialOffers": True,
        "emailNewExclusive": True,
        "emailStockAlerts": True,
    }
)
