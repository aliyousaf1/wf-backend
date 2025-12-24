from typing import Optional
from pydantic import BaseModel

class NotificationSettingsModel(BaseModel):
    email: str
    appNotificationsMain: bool
    appSpecialOffers: bool
    appNewExclusive: bool
    appStockAlerts: bool
    emailNotificationsMain: bool
    emailSpecialOffers: bool
    emailNewExclusive: bool
    emailStockAlerts: bool