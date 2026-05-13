from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any
from datetime import datetime


# ==================== MIDDLEWARE SCHEMAS ====================

class MiddlewareListItemSchema(BaseModel):
    """Schema for middleware list items"""
    middleware_id: int
    name: str
    description: Optional[str] = ""
    middleware_type: str
    config: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    is_frozen: bool = False

    model_config = ConfigDict(from_attributes=True)


class MiddlewareDetailSchema(BaseModel):
    """Schema for detailed middleware information"""
    middleware_id: int
    name: str
    description: Optional[str] = ""
    middleware_type: str
    config: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    is_frozen: bool = False

    model_config = ConfigDict(from_attributes=True)


class CreateUpdateMiddlewareSchema(BaseModel):
    """Schema for creating or updating a middleware"""
    name: str
    description: Optional[str] = ""
    middleware_type: str = "monitoring"
    config: Optional[Dict[str, Any]] = None
