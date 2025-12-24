from typing import Optional
from pydantic import BaseModel

class AuthUserModel(BaseModel):
    email: str
    password: str
    fcm_token: Optional[str] = None