"""Pydantic request/response schemas for the API's public contract."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ---- Authentication ---------------------------------------------------------

class RegisterRequest(BaseModel):
    """Payload for creating a new account."""
    email: EmailStr
    # bcrypt only uses the first 72 bytes, so we cap the length explicitly.
    password: str = Field(min_length=8, max_length=72)
    full_name: str | None = Field(default=None, max_length=120)


class LoginRequest(BaseModel):
    """Payload for exchanging credentials for an access token."""
    email: EmailStr
    password: str


class UserOut(BaseModel):
    """Public view of a user — never includes the password hash."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    full_name: str | None = None
    created_at: datetime


class TokenResponse(BaseModel):
    """Returned on successful login."""
    access_token: str
    token_type: str = "bearer"
    user: UserOut
