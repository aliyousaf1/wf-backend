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
    sex: str | None = None
    genders: List[str] = []  # List of genders for filtering (e.g., ["men", "women", "unisex"])
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

    # ── Style Points & Tiers ──────────────────────────────────────────────────
    style_points: int = 0
    tier: str = "silver"                    # silver | gold | platinum | vip_black
    vip_black: bool = False                 # paid membership flag
    title: str = "Style Explorer"

    # ── Streak ────────────────────────────────────────────────────────────────
    streak_current: int = 0
    streak_last_active_date: str | None = None   # "YYYY-MM-DD"
    streak_freeze_used: int = 0
    streak_freeze_month: str | None = None       # "YYYY-MM" — resets monthly

    # ── One-time SP award flags ───────────────────────────────────────────────
    onboarding_sp_awarded: bool = False
    style_pref_sp_awarded: bool = False

    # ── Tier maintenance ──────────────────────────────────────────────────────
    tier_grace_until: str | None = None     # ISO date set when maintenance missed
