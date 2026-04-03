from datetime import date, datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException
from app.core.database import db
from app.core.points_config import (
    ACTION_POINTS, DAILY_CAPS, WEEKLY_CAPS,
    SWIPE_MILESTONE_EVERY, SWIPE_MILESTONE_BONUS, SWIPE_MILESTONE_MAX,
    TIER_THRESHOLDS, TIER_MAINTENANCE, TIER_GRACE_DAYS,
    STREAK_MILESTONES, STREAK_FREEZE_ALLOWANCE,
    VIP_BLACK_MULTIPLIER, WEEKEND_MULTIPLIER,
    DOUBLE_POINTS_DAYS, FEATURE_SPOTLIGHT,
    PURCHASE_HIGH_VALUE_THRESHOLD,
    DEFAULT_TITLES, WEEKLY_MISSIONS, MONTHLY_MISSIONS,
)
from app.models.points_model import ActionRequest, CheckinRequest, MissionClaimRequest, StreakFreezeRequest
from bson import ObjectId
import random

points_route = APIRouter(prefix="/points", tags=["Style Points"])

usersCollection        = db["users"]
dailyActivityCollection = db["daily_activity"]
pointsLedgerCollection  = db["points_ledger"]
missionsCollection      = db["missions"]
userMissionsCollection  = db["user_missions"]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _today_str() -> str:
    return date.today().isoformat()   # "YYYY-MM-DD"

def _this_month_str() -> str:
    return date.today().strftime("%Y-%m")   # "YYYY-MM"

def _this_week_start() -> str:
    """ISO date of the Monday of the current week."""
    today = date.today()
    return (today - timedelta(days=today.weekday())).isoformat()

def _this_month_start() -> str:
    return date.today().replace(day=1).isoformat()

def _is_weekend() -> bool:
    return date.today().weekday() >= 5   # 5=Sat, 6=Sun


async def _get_user(email: str) -> dict:
    user = await usersCollection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


async def _get_or_create_daily_activity(email: str) -> dict:
    today = _today_str()
    doc = await dailyActivityCollection.find_one({"email": email, "date": today})
    if not doc:
        doc = {
            "email": email,
            "date": today,
            "app_open": 0,
            "checkin": 0,
            "first_swipe_session": 0,
            "swipe": 0,
            "save": 0,
            "product_view": 0,
            "share": 0,
            "retailer_click": 0,
            "camera_search": 0,
            "camera_search_result": 0,
            "tryon": 0,
            "tryon_save": 0,
            "rate_recommendation": 0,
            "follow_brand": 0,
            "follow_category": 0,
            "build_collection": 0,
            "collection_5_items": 0,
            "swipe_milestones_awarded": 0,
        }
        await dailyActivityCollection.insert_one(doc)
    return doc


async def _get_weekly_action_count(email: str, action: str) -> int:
    """Sum of an action across all daily_activity docs in the current week."""
    week_start = _this_week_start()
    pipeline = [
        {"$match": {"email": email, "date": {"$gte": week_start}}},
        {"$group": {"_id": None, "total": {"$sum": f"${action}"}}},
    ]
    result = await dailyActivityCollection.aggregate(pipeline).to_list(1)
    return result[0]["total"] if result else 0


def _calculate_multiplier(user: dict, action: str) -> float:
    multiplier = 1.0

    # VIP Black
    if user.get("vip_black"):
        multiplier *= VIP_BLACK_MULTIPLIER

    # Weekend bonus
    if _is_weekend():
        multiplier *= WEEKEND_MULTIPLIER

    # Double points day
    if _today_str() in DOUBLE_POINTS_DAYS:
        multiplier *= 2.0

    # Feature spotlight
    spotlight = FEATURE_SPOTLIGHT.get(action)
    if spotlight:
        until = spotlight.get("until", "")
        if _today_str() <= until:
            multiplier *= spotlight.get("multiplier", 1.0)

    return multiplier


async def _recalculate_tier(email: str, style_points: int, vip_black: bool) -> str:
    """Determine the correct tier from current SP total and vip_black flag."""
    if vip_black:
        new_tier = "vip_black"
    elif style_points >= TIER_THRESHOLDS["platinum"]:
        new_tier = "platinum"
    elif style_points >= TIER_THRESHOLDS["gold"]:
        new_tier = "gold"
    else:
        new_tier = "silver"

    new_title = DEFAULT_TITLES.get(new_tier, "Style Explorer")
    await usersCollection.update_one(
        {"email": email},
        {"$set": {"tier": new_tier, "title": new_title}}
    )
    return new_tier


async def _write_ledger(email: str, action: str, points: int, metadata: dict = {}):
    await pointsLedgerCollection.insert_one({
        "email": email,
        "action": action,
        "points": points,
        "timestamp": int(datetime.now(timezone.utc).timestamp()),
        "metadata": metadata,
    })


async def _update_mission_progress(email: str, action: str, increment: int = 1):
    """Increment progress on any active missions that track this action."""
    today = _today_str()
    week_start = _this_week_start()
    month_start = _this_month_start()

    # Find active missions for this action
    active_missions = await missionsCollection.find({
        "requirement_key": action,
        "period_end": {"$gte": today},
    }).to_list(length=None)

    for mission in active_missions:
        mission_id = str(mission["_id"])
        um = await userMissionsCollection.find_one(
            {"email": email, "mission_id": mission_id}
        )

        if um and um.get("completed"):
            continue  # already done

        new_progress = (um["progress"] if um else 0) + increment
        completed = new_progress >= mission["requirement_value"]

        await userMissionsCollection.update_one(
            {"email": email, "mission_id": mission_id},
            {"$set": {
                "email": email,
                "mission_id": mission_id,
                "progress": new_progress,
                "completed": completed,
                "claimed": um.get("claimed", False) if um else False,
            }},
            upsert=True,
        )


async def award_action(email: str, action: str, metadata: dict = {}) -> dict:
    """
    Core SP award engine. Called internally by all route handlers.
    Returns {"points_awarded": int, "new_total": int, "tier": str, "bonuses": list}
    """
    user = await _get_user(email)
    daily = await _get_or_create_daily_activity(email)

    base_points = ACTION_POINTS.get(action, 0)
    if base_points == 0:
        return {"points_awarded": 0, "new_total": user.get("style_points", 0), "tier": user.get("tier", "silver"), "bonuses": []}

    bonuses = []
    total_awarded = 0

    # ── Check daily cap ───────────────────────────────────────────────────────
    daily_cap = DAILY_CAPS.get(action)
    current_count = daily.get(action, 0)

    # Special: weekly cap for style_review
    if action in WEEKLY_CAPS:
        weekly_count = await _get_weekly_action_count(email, action)
        if weekly_count >= WEEKLY_CAPS[action]:
            return {"points_awarded": 0, "new_total": user.get("style_points", 0), "tier": user.get("tier", "silver"), "bonuses": []}
    elif daily_cap is not None and current_count >= daily_cap:
        return {"points_awarded": 0, "new_total": user.get("style_points", 0), "tier": user.get("tier", "silver"), "bonuses": []}

    # ── Feature guard: camera/tryon only on result ────────────────────────────
    if action in ("camera_search_result", "tryon_save"):
        if not metadata.get("result_generated"):
            return {"points_awarded": 0, "new_total": user.get("style_points", 0), "tier": user.get("tier", "silver"), "bonuses": []}

    # ── Calculate points with multipliers ─────────────────────────────────────
    multiplier = _calculate_multiplier(user, action)
    earned = round(base_points * multiplier)
    total_awarded += earned

    # ── Increment daily activity counter ──────────────────────────────────────
    await dailyActivityCollection.update_one(
        {"email": email, "date": _today_str()},
        {"$inc": {action: 1}}
    )

    # ── Swipe milestone bonus ─────────────────────────────────────────────────
    if action == "swipe":
        new_swipe_count = current_count + 1
        milestones_so_far = daily.get("swipe_milestones_awarded", 0)
        milestones_earned = new_swipe_count // SWIPE_MILESTONE_EVERY
        new_milestones = min(milestones_earned, SWIPE_MILESTONE_MAX) - milestones_so_far

        if new_milestones > 0:
            milestone_sp = new_milestones * SWIPE_MILESTONE_BONUS
            total_awarded += milestone_sp
            bonuses.append({"reason": f"swipe_milestone_x{new_milestones}", "points": milestone_sp})
            await dailyActivityCollection.update_one(
                {"email": email, "date": _today_str()},
                {"$inc": {"swipe_milestones_awarded": new_milestones}}
            )
            await _write_ledger(email, "swipe_milestone", milestone_sp, {"milestones": new_milestones})

    # ── Purchase high-value bonus ─────────────────────────────────────────────
    if action == "purchase":
        purchase_value = metadata.get("purchase_value", 0)
        if purchase_value >= PURCHASE_HIGH_VALUE_THRESHOLD:
            bonus_sp = round(ACTION_POINTS["purchase_high_value"] * multiplier)
            total_awarded += bonus_sp
            bonuses.append({"reason": "high_value_purchase", "points": bonus_sp})
            await _write_ledger(email, "purchase_high_value", bonus_sp, metadata)

    # ── Write main ledger entry ───────────────────────────────────────────────
    await _write_ledger(email, action, earned, metadata)

    # ── Update user style_points ──────────────────────────────────────────────
    await usersCollection.update_one(
        {"email": email},
        {"$inc": {"style_points": total_awarded}}
    )

    # ── Recalculate tier ──────────────────────────────────────────────────────
    new_total = user.get("style_points", 0) + total_awarded
    new_tier  = await _recalculate_tier(email, new_total, user.get("vip_black", False))

    # ── Update mission progress ───────────────────────────────────────────────
    await _update_mission_progress(email, action)

    return {
        "points_awarded": total_awarded,
        "new_total": new_total,
        "tier": new_tier,
        "bonuses": bonuses,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Streak logic
# ─────────────────────────────────────────────────────────────────────────────

async def _process_streak(email: str, user: dict) -> dict:
    """
    Called during check-in. Returns {"streak": int, "milestone_bonus": int, "freeze_used": bool}
    """
    today       = _today_str()
    yesterday   = (date.today() - timedelta(days=1)).isoformat()
    this_month  = _this_month_str()

    last_active  = user.get("streak_last_active_date")
    current      = user.get("streak_current", 0)
    freeze_used  = user.get("streak_freeze_used", 0)
    freeze_month = user.get("streak_freeze_month")
    tier         = "vip_black" if user.get("vip_black") else user.get("tier", "silver")
    freeze_allowance = STREAK_FREEZE_ALLOWANCE.get(tier, 0)

    # Reset monthly freeze counter if new month
    if freeze_month != this_month:
        freeze_used  = 0
        freeze_month = this_month

    freeze_consumed = False

    if last_active == today:
        # Already checked in today — no change
        return {"streak": current, "milestone_bonus": 0, "freeze_used": False}

    if last_active == yesterday:
        # Consecutive day
        current += 1
    elif last_active and (date.today() - date.fromisoformat(last_active)).days == 2:
        # Missed exactly 1 day — use a freeze if available
        if freeze_used < freeze_allowance:
            freeze_used    += 1
            freeze_consumed = True
            current        += 1
        else:
            current = 1
    else:
        # Gap too large or first ever — reset
        current = 1

    # Milestone bonus
    milestone_bonus = STREAK_MILESTONES.get(current, 0)

    update_fields = {
        "streak_current":         current,
        "streak_last_active_date": today,
        "streak_freeze_used":     freeze_used,
        "streak_freeze_month":    freeze_month,
    }
    await usersCollection.update_one({"email": email}, {"$set": update_fields})

    # Write milestone bonus to ledger and award SP
    if milestone_bonus > 0:
        await usersCollection.update_one({"email": email}, {"$inc": {"style_points": milestone_bonus}})
        await _write_ledger(email, f"streak_milestone_{current}d", milestone_bonus)

    return {"streak": current, "milestone_bonus": milestone_bonus, "freeze_used": freeze_consumed}


# ─────────────────────────────────────────────────────────────────────────────
# Tier maintenance check (call daily from a cron or on user login)
# ─────────────────────────────────────────────────────────────────────────────

async def check_tier_maintenance(email: str):
    """
    Checks whether a Gold/Platinum user has earned the required SP in the last 30 days.
    Applies grace period or downgrade as needed.
    """
    user = await _get_user(email)
    tier = user.get("tier", "silver")

    if tier not in TIER_MAINTENANCE:
        return  # Silver and VIP Black don't decay

    required   = TIER_MAINTENANCE[tier]
    cutoff     = (date.today() - timedelta(days=30)).isoformat()
    cutoff_ts  = int(datetime.fromisoformat(cutoff).replace(tzinfo=timezone.utc).timestamp())

    # Sum SP earned in the last 30 days from ledger
    pipeline = [
        {"$match": {"email": email, "timestamp": {"$gte": cutoff_ts}}},
        {"$group": {"_id": None, "total": {"$sum": "$points"}}},
    ]
    result = await pointsLedgerCollection.aggregate(pipeline).to_list(1)
    recent_sp = result[0]["total"] if result else 0

    if recent_sp >= required:
        # Maintenance met — clear any grace period
        await usersCollection.update_one({"email": email}, {"$unset": {"tier_grace_until": ""}})
        return

    # Maintenance missed
    grace_until = user.get("tier_grace_until")
    today_str   = _today_str()

    if not grace_until:
        # Start grace period
        grace_date = (date.today() + timedelta(days=TIER_GRACE_DAYS)).isoformat()
        await usersCollection.update_one({"email": email}, {"$set": {"tier_grace_until": grace_date}})
    elif today_str > grace_until:
        # Grace expired → downgrade
        new_tier  = "gold" if tier == "platinum" else "silver"
        new_title = DEFAULT_TITLES.get(new_tier, "Style Explorer")
        await usersCollection.update_one(
            {"email": email},
            {"$set": {"tier": new_tier, "title": new_title}, "$unset": {"tier_grace_until": ""}}
        )


# ─────────────────────────────────────────────────────────────────────────────
# Mission seeding helpers
# ─────────────────────────────────────────────────────────────────────────────

async def seed_weekly_missions():
    """Seed 3 random weekly missions for the current week. Idempotent."""
    week_start = _this_week_start()
    week_end   = (date.fromisoformat(week_start) + timedelta(days=6)).isoformat()

    existing = await missionsCollection.count_documents({"type": "weekly", "period_start": week_start})
    if existing >= 3:
        return

    chosen = random.sample(WEEKLY_MISSIONS, min(3, len(WEEKLY_MISSIONS)))
    docs = [
        {**m, "type": "weekly", "period_start": week_start, "period_end": week_end}
        for m in chosen
    ]
    await missionsCollection.insert_many(docs)


async def seed_monthly_missions():
    """Seed all monthly missions for the current month. Idempotent."""
    month_start = _this_month_start()
    month_end   = (date.fromisoformat(month_start).replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    month_end_str = month_end.isoformat()

    existing = await missionsCollection.count_documents({"type": "monthly", "period_start": month_start})
    if existing >= len(MONTHLY_MISSIONS):
        return

    docs = [
        {**m, "type": "monthly", "period_start": month_start, "period_end": month_end_str}
        for m in MONTHLY_MISSIONS
    ]
    await missionsCollection.insert_many(docs)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@points_route.post("/checkin")
async def daily_checkin(body: CheckinRequest):
    """
    Daily check-in. Awards check-in SP + app open SP, processes streak logic.
    Mobile should call this once per day when the user opens the app intentionally.
    """
    user = await _get_user(body.email)

    # Ensure missions are seeded
    await seed_weekly_missions()
    await seed_monthly_missions()

    # Check tier maintenance on every checkin
    await check_tier_maintenance(body.email)

    # Streak processing
    streak_result = await _process_streak(body.email, user)

    # Award app_open SP
    app_open_result = await award_action(body.email, "app_open")

    # Award checkin SP
    checkin_result = await award_action(body.email, "checkin")

    # Update mission progress for checkin
    await _update_mission_progress(body.email, "checkin")

    # Weekend session mission
    if _is_weekend():
        await _update_mission_progress(body.email, "weekend_checkin")

    total_awarded = app_open_result["points_awarded"] + checkin_result["points_awarded"] + streak_result["milestone_bonus"]

    return {
        "status": "SUCCESS",
        "points_awarded": total_awarded,
        "new_total": checkin_result["new_total"],
        "tier": checkin_result["tier"],
        "streak": streak_result["streak"],
        "streak_milestone_bonus": streak_result["milestone_bonus"],
        "freeze_used": streak_result["freeze_used"],
    }


@points_route.post("/action")
async def record_action(body: ActionRequest):
    """
    Generic action SP award. Mobile calls this for:
    share, follow_brand, follow_category, build_collection, rate_recommendation,
    style_review, purchase, camera_search_result, tryon_save, etc.

    For swipe/save/product_view/retailer_click/camera_search/tryon — those are
    triggered automatically from the feed routes.
    """
    result = await award_action(body.email, body.action, body.metadata or {})

    if result["points_awarded"] == 0:
        return {"status": "CAP_REACHED", "points_awarded": 0, "new_total": result["new_total"], "tier": result["tier"]}

    return {
        "status": "SUCCESS",
        "points_awarded": result["points_awarded"],
        "new_total": result["new_total"],
        "tier": result["tier"],
        "bonuses": result["bonuses"],
    }


@points_route.get("/summary")
async def get_summary(email: str):
    """
    Returns the user's full points summary for display in the mobile app.
    """
    user = await _get_user(email)
    daily = await _get_or_create_daily_activity(email)

    sp    = user.get("style_points", 0)
    tier  = user.get("tier", "silver")

    # Progress to next tier
    if tier == "silver":
        next_tier      = "gold"
        next_threshold = TIER_THRESHOLDS["gold"]
        progress_pct   = min(round((sp / next_threshold) * 100), 100)
        sp_to_next     = max(next_threshold - sp, 0)
    elif tier == "gold":
        next_tier      = "platinum"
        next_threshold = TIER_THRESHOLDS["platinum"]
        progress_pct   = min(round(((sp - TIER_THRESHOLDS["gold"]) / (next_threshold - TIER_THRESHOLDS["gold"])) * 100), 100)
        sp_to_next     = max(next_threshold - sp, 0)
    else:
        next_tier      = None
        next_threshold = None
        progress_pct   = 100
        sp_to_next     = 0

    # Caps remaining today
    caps_remaining = {}
    for action, cap in DAILY_CAPS.items():
        caps_remaining[action] = max(cap - daily.get(action, 0), 0)

    return {
        "status": "SUCCESS",
        "style_points": sp,
        "tier": tier,
        "title": user.get("title", DEFAULT_TITLES.get(tier, "Style Explorer")),
        "vip_black": user.get("vip_black", False),
        "streak": user.get("streak_current", 0),
        "last_active": user.get("streak_last_active_date"),
        "streak_freeze_remaining": max(
            STREAK_FREEZE_ALLOWANCE.get(tier, 0) - user.get("streak_freeze_used", 0), 0
        ),
        "next_tier": next_tier,
        "sp_to_next_tier": sp_to_next,
        "tier_progress_pct": progress_pct,
        "caps_remaining_today": caps_remaining,
        "tier_grace_until": user.get("tier_grace_until"),
    }


@points_route.get("/history")
async def get_history(email: str, limit: int = 20, offset: int = 0):
    """Paginated points ledger for the user."""
    await _get_user(email)   # 404 guard

    cursor = pointsLedgerCollection.find(
        {"email": email},
        {"_id": 0}
    ).sort("timestamp", -1).skip(offset).limit(limit)

    entries = await cursor.to_list(length=limit)

    return {
        "status": "SUCCESS",
        "entries": entries,
        "limit": limit,
        "offset": offset,
    }


@points_route.get("/missions")
async def get_missions(email: str):
    """
    Returns active weekly and monthly missions with this user's progress.
    """
    today = _today_str()

    active_missions = await missionsCollection.find(
        {"period_end": {"$gte": today}}
    ).to_list(length=None)

    result = []
    for m in active_missions:
        mission_id = str(m["_id"])
        um = await userMissionsCollection.find_one({"email": email, "mission_id": mission_id})
        result.append({
            "mission_id": mission_id,
            "name":              m["name"],
            "type":              m["type"],
            "requirement_key":   m["requirement_key"],
            "requirement_value": m["requirement_value"],
            "reward_sp":         m["reward_sp"],
            "period_end":        m["period_end"],
            "progress":          um["progress"] if um else 0,
            "completed":         um["completed"] if um else False,
            "claimed":           um.get("claimed", False) if um else False,
        })

    return {"status": "SUCCESS", "missions": result}


@points_route.post("/missions/claim")
async def claim_mission(body: MissionClaimRequest):
    """
    Claim the SP reward for a completed mission.
    """
    um = await userMissionsCollection.find_one(
        {"email": body.email, "mission_id": body.mission_id}
    )
    if not um:
        raise HTTPException(status_code=404, detail="Mission progress not found")
    if not um.get("completed"):
        raise HTTPException(status_code=400, detail="Mission not yet completed")
    if um.get("claimed"):
        raise HTTPException(status_code=400, detail="Mission already claimed")

    mission = await missionsCollection.find_one({"_id": ObjectId(body.mission_id)})
    if not mission:
        raise HTTPException(status_code=404, detail="Mission not found")

    reward_sp = mission["reward_sp"]

    await usersCollection.update_one({"email": body.email}, {"$inc": {"style_points": reward_sp}})
    await _write_ledger(body.email, f"mission_{mission['name'].lower().replace(' ', '_')}", reward_sp)
    await userMissionsCollection.update_one(
        {"email": body.email, "mission_id": body.mission_id},
        {"$set": {"claimed": True}}
    )

    user = await _get_user(body.email)
    new_total = user.get("style_points", 0)
    new_tier  = await _recalculate_tier(body.email, new_total, user.get("vip_black", False))

    return {
        "status": "SUCCESS",
        "reward_sp": reward_sp,
        "new_total": new_total,
        "tier": new_tier,
    }


@points_route.post("/streak/use-freeze")
async def use_streak_freeze(body: StreakFreezeRequest):
    """
    Manually consume a streak freeze. Mobile calls this when user taps 'protect streak'.
    """
    user      = await _get_user(body.email)
    tier      = "vip_black" if user.get("vip_black") else user.get("tier", "silver")
    allowance = STREAK_FREEZE_ALLOWANCE.get(tier, 0)
    this_month = _this_month_str()

    freeze_month = user.get("streak_freeze_month")
    freeze_used  = user.get("streak_freeze_used", 0)

    if freeze_month != this_month:
        freeze_used  = 0
        freeze_month = this_month

    if freeze_used >= allowance:
        raise HTTPException(status_code=400, detail="No streak freezes remaining this month")

    await usersCollection.update_one(
        {"email": body.email},
        {"$set": {
            "streak_freeze_used":  freeze_used + 1,
            "streak_freeze_month": freeze_month,
        }}
    )

    return {
        "status": "SUCCESS",
        "freezes_used": freeze_used + 1,
        "freezes_remaining": allowance - freeze_used - 1,
    }
