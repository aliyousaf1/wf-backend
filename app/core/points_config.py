# ─────────────────────────────────────────────────────────────────────────────
# WF Style Points — central configuration
# All caps, thresholds, multipliers, and streak milestones live here.
# ─────────────────────────────────────────────────────────────────────────────

# ── Action points ─────────────────────────────────────────────────────────────
ACTION_POINTS = {
    "app_open":              10,
    "checkin":               15,
    "first_swipe_session":   20,
    "swipe":                  1,
    "save":                   4,
    "product_view":           3,
    "share":                 10,
    "retailer_click":         8,
    "camera_search":         40,
    "camera_search_result":  20,   # bonus on top of camera_search when result generated
    "tryon":                 50,
    "tryon_save":            25,   # bonus when try-on result is saved
    "rate_recommendation":    5,
    "follow_brand":           8,
    "follow_category":        8,
    "build_collection":      20,
    "collection_5_items":    15,   # bonus when a collection reaches 5 items
    "style_review":          25,
    "onboarding_complete":  100,
    "style_preferences":     75,
    "referral_signup":      100,   # inviter gets this when referred user signs up
    "referral_onboarded":    50,   # inviter gets this when referred user completes onboarding
    "referral_7day_active": 150,   # inviter gets this when referred user is active for 7 days
    "purchase":             200,
    "purchase_high_value":  100,   # bonus on top of purchase when above threshold
}

# ── Daily caps (max times an action earns points per day) ────────────────────
DAILY_CAPS = {
    "app_open":              1,
    "checkin":               1,
    "first_swipe_session":   1,
    "swipe":               100,
    "save":                 20,
    "product_view":         30,
    "share":                 5,
    "retailer_click":       10,
    "camera_search":         3,
    "camera_search_result":  3,
    "tryon":                 2,
    "tryon_save":            2,
    "rate_recommendation":  20,
    "follow_brand":         10,
    "follow_category":       5,
    "build_collection":      2,
    "collection_5_items":    2,
    # style_review is weekly (3/week), handled separately
}

# ── Weekly caps ───────────────────────────────────────────────────────────────
WEEKLY_CAPS = {
    "style_review": 3,
}

# ── Swipe milestone bonus (every N swipes in a day → +bonus, max M times) ───
SWIPE_MILESTONE_EVERY  = 25    # award bonus every 25 swipes
SWIPE_MILESTONE_BONUS  = 10    # bonus SP per milestone
SWIPE_MILESTONE_MAX    = 4     # max milestones per day

# ── Tier thresholds ───────────────────────────────────────────────────────────
TIER_THRESHOLDS = {
    "silver":    0,
    "gold":   1500,
    "platinum": 5000,
    # vip_black is paid — handled by vip_black flag on user doc
}

# Tier maintenance: SP required per 30-day window to keep tier
TIER_MAINTENANCE = {
    "gold":     300,
    "platinum": 800,
}

TIER_GRACE_DAYS = 14    # grace period before downgrade

# ── Streak milestones → SP bonus ─────────────────────────────────────────────
STREAK_MILESTONES = {
     2:   20,
     3:   30,
     5:   50,
     7:  100,
    14:  175,
    21:  250,
    30:  400,
}

# Streak freeze allowance per tier (per month)
STREAK_FREEZE_ALLOWANCE = {
    "silver":    0,
    "gold":      1,
    "platinum":  2,
    "vip_black": 4,
}

# ── Multipliers ────────────────────────────────────────────────────────────────
VIP_BLACK_MULTIPLIER  = 1.5
WEEKEND_MULTIPLIER    = 1.2
DOUBLE_POINTS_DAYS: list[str] = []   # populated by admin: ["2026-04-10", ...]
FEATURE_SPOTLIGHT: dict = {}         # e.g. {"camera_search": {"multiplier": 2.0, "until": "2026-04-15"}}

# ── Bonus mechanics ────────────────────────────────────────────────────────────
FIRST_SESSION_BONUS       = 20    # extra SP for first meaningful session of day
COMEBACK_ABSENT_DAYS      = 7     # days absent before comeback bonus triggers
PURCHASE_HIGH_VALUE_THRESHOLD = 1000  # AED

# ── Weekly missions (Phase 1: 3 active at a time) ────────────────────────────
WEEKLY_MISSIONS = [
    {"name": "Discovery Sprint",     "requirement_key": "swipe",          "requirement_value": 200, "reward_sp": 120},
    {"name": "Stylist in Training",  "requirement_key": "camera_search",  "requirement_value":   3, "reward_sp": 100},
    {"name": "Virtual Fitting Room", "requirement_key": "tryon",          "requirement_value":   3, "reward_sp": 130},
    {"name": "Curator",              "requirement_key": "save",           "requirement_value":  25, "reward_sp":  90},
    {"name": "Brand Explorer",       "requirement_key": "follow_brand",   "requirement_value":   5, "reward_sp":  60},
    {"name": "Detail-Oriented",      "requirement_key": "product_view",   "requirement_value":  20, "reward_sp":  70},
    {"name": "Social Tastemaker",    "requirement_key": "share",          "requirement_value":   3, "reward_sp":  50},
    {"name": "Weekend Session",      "requirement_key": "weekend_checkin","requirement_value":   2, "reward_sp":  60},
]

# ── Monthly missions ──────────────────────────────────────────────────────────
MONTHLY_MISSIONS = [
    {"name": "Consistency Month",  "requirement_key": "checkin",         "requirement_value":  20, "reward_sp": 350},
    {"name": "Fashion Explorer",   "requirement_key": "swipe",           "requirement_value":1000, "reward_sp": 400},
    {"name": "Smart Discovery",    "requirement_key": "camera_search",   "requirement_value":  10, "reward_sp": 300},
    {"name": "Try-On Collector",   "requirement_key": "tryon",           "requirement_value":   8, "reward_sp": 350},
    {"name": "Luxury Hunter",      "requirement_key": "retailer_click",  "requirement_value":  20, "reward_sp": 250},
    {"name": "Style Architect",    "requirement_key": "build_collection","requirement_value":   5, "reward_sp": 250},
]

# ── Title system ──────────────────────────────────────────────────────────────
DEFAULT_TITLES = {
    "silver":    "Style Explorer",
    "gold":      "Trend Curator",
    "platinum":  "Fashion Insider",
    "vip_black": "House Member",
}
