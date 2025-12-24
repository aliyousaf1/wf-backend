from pydantic import BaseModel, Field
from typing import List, Optional

class ProductModel(BaseModel):
    id: str = Field(..., alias="_id")
    img_url: str
    title: str
    price: str
    brand: str
    description: str
    highlights: List[str]

    class Config:
        populate_by_name = True  # allows using "id" instead of "_id"
        arbitrary_types_allowed = True
