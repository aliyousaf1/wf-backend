from pydantic import BaseModel
from typing import Optional

class ActionRequest(BaseModel):
    email: str
    action: str
    metadata: Optional[dict] = {}   # e.g. {"result_generated": True, "purchase_value": 1200}

class CheckinRequest(BaseModel):
    email: str

class MissionClaimRequest(BaseModel):
    email: str
    mission_id: str

class StreakFreezeRequest(BaseModel):
    email: str
