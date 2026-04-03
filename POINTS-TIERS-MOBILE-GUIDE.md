# WF Style Points & Tiers — Mobile Integration Guide

This doc covers everything you need to integrate the points and tiers system into the WhyFashion app. The backend handles all the logic — you just need to call the right endpoints at the right moments and display the data.

---

## Overview

The system has 4 moving parts:

| System | What it does |
|---|---|
| **Style Points (SP)** | The main currency. Earned from user actions. |
| **Tiers** | Silver → Gold → Platinum → VIP Black. Unlocked by SP total. |
| **Streaks** | Consecutive daily check-ins. Milestone bonuses at 2, 3, 5, 7, 14, 21, 30 days. |
| **Missions** | Weekly and monthly goals. Complete them to claim bonus SP. |

Most SP awards happen automatically (swipes, saves, product views). You only need to manually call a handful of endpoints.

---

## Endpoints

**Base URL:** same as the existing API (`https://app.whyfashion.com`)

All new endpoints are under `/points`.

---

## 1. Daily Check-in

**Call this once per day when the user opens the app.**

It handles: app open SP + check-in SP + streak processing + mission seeding + tier maintenance check.

```
POST /points/checkin
```

**Request body:**
```json
{
  "email": "user@example.com"
}
```

**Response:**
```json
{
  "status": "SUCCESS",
  "points_awarded": 45,
  "new_total": 320,
  "tier": "silver",
  "streak": 5,
  "streak_milestone_bonus": 50,
  "freeze_used": false
}
```

| Field | Description |
|---|---|
| `points_awarded` | Total SP earned from this check-in (app open + check-in + streak milestone if any) |
| `new_total` | User's new SP total |
| `tier` | Current tier: `silver`, `gold`, `platinum`, `vip_black` |
| `streak` | Current streak count in days |
| `streak_milestone_bonus` | Extra SP awarded if a streak milestone was hit (0 if none) |
| `freeze_used` | `true` if a streak freeze was automatically consumed to maintain streak |

**When to call:** On app foreground/open. Track locally with a date so you only call it once per calendar day. If called multiple times in a day, the backend ignores the duplicates — but avoid unnecessary calls.

**UI suggestion:** Show a check-in animation with the SP total when `points_awarded > 0`. If `streak_milestone_bonus > 0`, show a streak milestone celebration.

---

## 2. Record an Action

**Call this for any trackable action that isn't already automatic.**

```
POST /points/action
```

**Request body:**
```json
{
  "email": "user@example.com",
  "action": "share",
  "metadata": {}
}
```

**Response (points awarded):**
```json
{
  "status": "SUCCESS",
  "points_awarded": 10,
  "new_total": 330,
  "tier": "silver",
  "bonuses": []
}
```

**Response (daily cap reached):**
```json
{
  "status": "CAP_REACHED",
  "points_awarded": 0,
  "new_total": 330,
  "tier": "silver"
}
```

### Action reference table

| Action string | When to call | Points | Daily cap |
|---|---|---|---|
| `share` | User shares a product | 10 SP | 5/day |
| `follow_brand` | User follows a brand | 8 SP | 10/day |
| `follow_category` | User follows a category | 8 SP | 5/day |
| `build_collection` | User creates a new collection/board | 20 SP | 2/day |
| `collection_5_items` | A collection the user owns reaches 5 items | 15 SP | 2/day |
| `rate_recommendation` | User rates a recommendation | 5 SP | 20/day |
| `style_review` | User writes a style review or feedback | 25 SP | 3/week |
| `camera_search` | User triggers Search by Camera | 40 SP | 3/day |
| `camera_search_result` | Camera search returns a result | 20 SP | 3/day |
| `tryon` | User uses AI Try-On | 50 SP | 2/day |
| `tryon_save` | User saves a try-on result | 25 SP | 2/day |
| `retailer_click` | User taps "Buy" / clicks out to retailer | 8 SP | 10/day |
| `purchase` | User makes a purchase via the app | 200 SP | unlimited |

> **Note:** `swipe`, `save`, `product_view`, `app_open`, and `checkin` are **already automatic** — the backend awards them when you call the existing endpoints. Do not call `/points/action` for those.

### Camera search — call both actions in sequence

```
// 1. User opens camera and triggers search
POST /points/action  →  action: "camera_search"

// 2. Results load successfully
POST /points/action  →  action: "camera_search_result"
                        metadata: { "result_generated": true }
```

### AI Try-On — call both actions in sequence

```
// 1. User starts a try-on session
POST /points/action  →  action: "tryon"

// 2. User saves the try-on result/look
POST /points/action  →  action: "tryon_save"
                        metadata: { "result_generated": true }
```

> The `result_generated: true` metadata flag is required for `camera_search_result` and `tryon_save`. Without it, no points are awarded (anti-abuse guard).

### Purchase with high value bonus

```json
{
  "email": "user@example.com",
  "action": "purchase",
  "metadata": { "purchase_value": 1500 }
}
```

If `purchase_value >= 1000` (AED), the user automatically gets an extra 100 SP bonus on top of the 200 SP base.

---

## 3. Points Summary

**Call this to render the user's points UI — progress bar, tier badge, streak count, etc.**

```
GET /points/summary?email=user@example.com
```

**Response:**
```json
{
  "status": "SUCCESS",
  "style_points": 1240,
  "tier": "silver",
  "title": "Style Explorer",
  "vip_black": false,
  "streak": 7,
  "last_active": "2026-04-03",
  "streak_freeze_remaining": 0,
  "next_tier": "gold",
  "sp_to_next_tier": 260,
  "tier_progress_pct": 82,
  "caps_remaining_today": {
    "swipe": 63,
    "save": 14,
    "product_view": 22,
    "share": 5,
    "retailer_click": 10,
    "camera_search": 3,
    "tryon": 2
  },
  "tier_grace_until": null
}
```

| Field | Description |
|---|---|
| `style_points` | Total SP all-time |
| `tier` | `silver`, `gold`, `platinum`, `vip_black` |
| `title` | Fashion identity title (e.g. "Style Explorer", "Trend Curator") |
| `vip_black` | Boolean — paid VIP Black membership active |
| `streak` | Current streak in days |
| `streak_freeze_remaining` | How many streak freezes left this month |
| `next_tier` | Next tier name, `null` if at top |
| `sp_to_next_tier` | SP needed to reach next tier, `0` if at top |
| `tier_progress_pct` | 0–100 — use this to fill a progress bar |
| `caps_remaining_today` | How many more times each action will earn SP today |
| `tier_grace_until` | ISO date string if in grace period for maintenance, otherwise `null` |

**When to call:** On profile screen load, on points hub screen, and after any action that awards points (refresh the display).

---

## 4. Points History (Ledger)

**Paginated list of SP transactions for the user.**

```
GET /points/history?email=user@example.com&limit=20&offset=0
```

**Response:**
```json
{
  "status": "SUCCESS",
  "entries": [
    {
      "action": "swipe",
      "points": 1,
      "timestamp": 1743800000,
      "metadata": {}
    },
    {
      "action": "streak_milestone_7d",
      "points": 100,
      "timestamp": 1743790000,
      "metadata": {}
    }
  ],
  "limit": 20,
  "offset": 0
}
```

---

## 5. Missions

### Get active missions

```
GET /points/missions?email=user@example.com
```

**Response:**
```json
{
  "status": "SUCCESS",
  "missions": [
    {
      "mission_id": "abc123",
      "name": "Discovery Sprint",
      "type": "weekly",
      "requirement_key": "swipe",
      "requirement_value": 200,
      "reward_sp": 120,
      "period_end": "2026-04-06",
      "progress": 143,
      "completed": false,
      "claimed": false
    },
    {
      "mission_id": "def456",
      "name": "Virtual Fitting Room",
      "type": "weekly",
      "requirement_key": "tryon",
      "requirement_value": 3,
      "reward_sp": 130,
      "period_end": "2026-04-06",
      "progress": 3,
      "completed": true,
      "claimed": false
    }
  ]
}
```

| Field | Description |
|---|---|
| `requirement_key` | The action being tracked |
| `requirement_value` | Target count to complete the mission |
| `progress` | User's current count toward the target |
| `completed` | `true` when `progress >= requirement_value` |
| `claimed` | `true` after the reward has been collected |

**UI:** Show a progress bar per mission: `progress / requirement_value`. When `completed && !claimed`, show a "Claim" button.

### Claim a mission reward

```
POST /points/missions/claim
```

**Request body:**
```json
{
  "email": "user@example.com",
  "mission_id": "def456"
}
```

**Response:**
```json
{
  "status": "SUCCESS",
  "reward_sp": 130,
  "new_total": 1370,
  "tier": "silver"
}
```

After claiming, refresh the missions list and the summary.

---

## 6. Streak Freeze (Manual)

If you want to show a "Protect my streak" button in the UI, call this. It manually consumes a streak freeze without waiting for the automatic process.

```
POST /points/streak/use-freeze
```

**Request body:**
```json
{
  "email": "user@example.com"
}
```

**Response:**
```json
{
  "status": "SUCCESS",
  "freezes_used": 1,
  "freezes_remaining": 0
}
```

**Error (no freezes left):**
```
HTTP 400 — "No streak freezes remaining this month"
```

---

## Actions already handled automatically

These endpoints already award points behind the scenes. You do not need to call `/points/action` for any of these:

| Existing endpoint | SP awarded |
|---|---|
| `POST /feed/add_to_likes` | 1 SP (swipe) + first session bonus |
| `POST /feed/add_to_dislikes` | 1 SP (swipe) + first session bonus |
| `POST /feed/add_to_watched` | 3 SP (product view) |
| WebSocket `LIKE` event | 1 SP (swipe) |
| WebSocket `DISLIKE` event | 1 SP (swipe) |
| WebSocket `WATCHED` event | 3 SP (product view) |
| `POST /onboarding/create_user` | 10 SP (app open) |
| `PATCH /onboarding/update_user` with `onboarded: true` | 100 SP (one-time) |
| `PATCH /onboarding/update_user` with style prefs | 75 SP (one-time) |

---

## Tier reference

| Tier | SP required | Ongoing maintenance | Streak freezes/month |
|---|---|---|---|
| Silver | 0 (default) | None | 0 |
| Gold | 1,500 SP | 300 SP per 30 days | 1 |
| Platinum | 5,000 SP | 800 SP per 30 days | 2 |
| VIP Black | Paid subscription | Active subscription | 4 |

**Maintenance:** If a Gold or Platinum user misses their 30-day SP requirement, they get a 14-day grace period (`tier_grace_until` in the summary response will have a date). After the grace period expires their tier drops. Show this in the UI if `tier_grace_until` is not null.

**VIP Black sub-tiers** (internal prestige, same tier value = `vip_black`):
- VIP Black
- VIP Black Elite
- VIP Black Icon

---

## Streak milestone rewards

| Streak | Bonus SP | Badge / Unlock |
|---|---|---|
| 2 days | +20 SP | — |
| 3 days | +30 SP | — |
| 5 days | +50 SP | — |
| 7 days | +100 SP | Badge |
| 14 days | +175 SP | Profile flair |
| 21 days | +250 SP | Special theme unlock |
| 30 days | +400 SP | "Style Devotee" badge |

These are awarded automatically during `/points/checkin`. The `streak_milestone_bonus` field in the check-in response tells you if one was triggered so you can show a celebration UI.

---

## Suggested UI surfaces

### Home screen
- Streak count + flame icon
- Today's missions (compact view, 1–3)
- SP earned today

### Profile / Points hub screen
- Tier badge + tier name + title (e.g. "Trend Curator")
- Progress bar to next tier (`tier_progress_pct`, `sp_to_next_tier`)
- Total SP
- Streak count + freeze count
- Active missions with progress bars + claim buttons
- Points history list

### Swipe/feed screen
- Subtle swipe counter (e.g. "47 / 100 swipes today")
- Milestone flash at 25, 50, 75, 100 swipes

### AI Try-On screen
- Badge: "Earn 50 SP this session"
- Show bonus prompt: "Save your look for +25 SP"

### Search by Camera screen
- Badge: "Earn 40 SP per search (3 remaining today)"

### After any SP-earning action
- Toast / snackbar: "+10 SP" with current total

### End-of-session summary (optional)
- "You earned 84 SP today. 3 more days for your 7-day streak reward. 220 SP to Gold."

---

## Integration checklist

- [ ] Call `POST /points/checkin` once per day on app open
- [ ] Call `POST /points/action` for: share, follow brand, follow category, build collection, rate recommendation, camera search, try-on, retailer click, purchase
- [ ] Pass `metadata: { "result_generated": true }` for `camera_search_result` and `tryon_save`
- [ ] Display points summary from `GET /points/summary`
- [ ] Show missions from `GET /points/missions` with claim flow
- [ ] Show streak count and milestone celebrations from check-in response
- [ ] Show `tier_grace_until` warning if maintenance grace period is active
- [ ] Show `+N SP` feedback after actions that return `points_awarded > 0`
