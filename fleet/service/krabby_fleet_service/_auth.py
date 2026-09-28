"""Cognito access-token verification and operator-group authorization.

`require_operator` is a FastAPI dependency: 401 for a missing, malformed,
expired, or wrong-audience token; 403 for a valid token whose user isn't in
the "operator" group -- being authenticated isn't enough on its own to open
a tunnel, group membership is a separate, required check.

WebSocket clients (browser ``WebSocket``) often cannot set Authorization
headers; ``require_operator_websocket`` also accepts ``?token=``.
"""
from __future__ import annotations

import logging
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Request, WebSocket
from jwt import PyJWKClient

from krabby_fleet_service._config import Settings, get_settings

logger = logging.getLogger(__name__)

_jwks_clients: dict[str, PyJWKClient] = {}


def _jwks_client(settings: Settings) -> PyJWKClient:
    # PyJWKClient caches keys internally too, but caching the client itself
    # avoids re-fetching the JWKS document on every single request.
    client = _jwks_clients.get(settings.cognito_jwks_url)
    if client is None:
        client = PyJWKClient(settings.cognito_jwks_url)
        _jwks_clients[settings.cognito_jwks_url] = client
    return client


def verify_operator_token(token: str, settings: Settings) -> dict[str, Any]:
    """Validate a Cognito access token and require ``operator`` group membership."""
    try:
        signing_key = _jwks_client(settings).get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=settings.cognito_issuer,
            # Cognito access tokens carry `client_id`, not the standard `aud`
            # claim -- verified manually below instead.
            options={"verify_aud": False},
        )
    except jwt.PyJWTError as exc:
        logger.warning("operator auth rejected: invalid token (%s)", exc.__class__.__name__)
        raise HTTPException(status_code=401, detail=f"invalid token: {exc}") from exc

    if claims.get("token_use") != "access":
        logger.warning("operator auth rejected: not an access token")
        raise HTTPException(status_code=401, detail="not an access token")
    if claims.get("client_id") != settings.cognito_app_client_id:
        logger.warning("operator auth rejected: wrong app client_id")
        raise HTTPException(status_code=401, detail="token was not issued for this client")

    if "operator" not in (claims.get("cognito:groups") or []):
        logger.warning("operator auth rejected: user not in operator group")
        raise HTTPException(status_code=403, detail="user is not in the operator group")

    return claims


def _bearer_token(request: Request) -> str:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="missing or malformed Authorization header")
    return token


def require_operator(
    request: Request, settings: Settings = Depends(get_settings)
) -> dict[str, Any]:
    return verify_operator_token(_bearer_token(request), settings)


def _websocket_token(websocket: WebSocket) -> str:
    scheme, _, token = websocket.headers.get("authorization", "").partition(" ")
    if scheme.lower() == "bearer" and token:
        return token
    q = websocket.query_params.get("token")
    if q:
        return q
    raise HTTPException(
        status_code=401,
        detail="missing token (Authorization: Bearer or ?token=)",
    )


def require_operator_websocket(
    websocket: WebSocket, settings: Settings = Depends(get_settings)
) -> dict[str, Any]:
    return verify_operator_token(_websocket_token(websocket), settings)
