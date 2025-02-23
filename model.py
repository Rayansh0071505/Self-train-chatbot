# models.py
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict
from datetime import datetime

class User(BaseModel):
    user_id: str
    company_name: str
    user_name: str
    email: EmailStr
    created_at: datetime = datetime.utcnow()

class OnboardingData(BaseModel):
    user_id: str
    industry: str
    goal: str
    support_link: str
    meeting_link: Optional[str]
    general_info: str
    shopify_store: str
    shopify_token: str
    created_at: datetime = datetime.utcnow()

class Product(BaseModel):
    user_id: str
    product_id: str
    title: str
    description: str
    price: float
    variants: List[Dict]
    images: List[Dict]
    embedding: List[float]
    created_at: datetime = datetime.utcnow()
    updated_at: datetime = datetime.utcnow()