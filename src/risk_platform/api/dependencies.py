from __future__ import annotations

from fastapi import Header, HTTPException, status

from risk_platform.config import settings


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if settings.require_api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
