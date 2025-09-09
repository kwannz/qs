import logging
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.security.http import HTTPBasic, HTTPBasicCredentials

from freqtrade.rpc.api_server.api_schemas import AccessAndRefreshToken, AccessToken
from freqtrade.rpc.api_server.deps import get_api_config


logger = logging.getLogger(__name__)

ALGORITHM = "HS256"

router_login = APIRouter()


def verify_auth(api_config, username: str, password: str):
    """Verify username/password"""
    return secrets.compare_digest(username, api_config.get("username")) and secrets.compare_digest(
        password, api_config.get("password")
    )


httpbasic = HTTPBasic(auto_error=False)
security = HTTPBasic()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


def get_user_from_token(token, secret_key: str, token_type: str = "access") -> dict:  # noqa: S107
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, secret_key, algorithms=[ALGORITHM])
        identity = payload.get("identity", {})
        username: str | None = identity.get("u")
        roles: list[str] = identity.get("roles", []) or []
        if username is None:
            raise credentials_exception
        if payload.get("type") != token_type:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    return {"username": username, "roles": roles, "type": token_type}


# This should be reimplemented to better realign with the existing tools provided
# by FastAPI regarding API Tokens
# https://github.com/tiangolo/fastapi/blob/master/fastapi/security/api_key.py
async def validate_ws_token(
    ws: WebSocket,
    ws_token: str | None = Query(default=None, alias="token"),
    api_config: dict[str, Any] = Depends(get_api_config),
):
    # Allow all when auth is disabled (personal/local use)
    if api_config.get("auth_disabled", False):
        return "anonymous"
    secret_ws_token = api_config.get("ws_token", None)
    secret_jwt_key = api_config.get("jwt_secret_key", "super-secret")

    # Check if ws_token is/in secret_ws_token
    if ws_token and secret_ws_token:
        is_valid_ws_token = False
        if isinstance(secret_ws_token, str):
            is_valid_ws_token = secrets.compare_digest(secret_ws_token, ws_token)
        elif isinstance(secret_ws_token, list):
            is_valid_ws_token = any(
                [secrets.compare_digest(potential, ws_token) for potential in secret_ws_token]
            )

        if is_valid_ws_token:
            return ws_token

    # Check if ws_token is a JWT
    try:
        user = get_user_from_token(ws_token, secret_jwt_key)
        return user.get("username")
    # If the token is a jwt, and it's valid return the user
    except HTTPException:
        pass

    # If it doesn't match, close the websocket connection
    await ws.close(code=status.WS_1008_POLICY_VIOLATION)


def create_token(data: dict, secret_key: str, token_type: str = "access") -> str:  # noqa: S107
    to_encode = data.copy()
    if token_type == "access":  # noqa: S105
        expire = datetime.now(UTC) + timedelta(minutes=15)
    elif token_type == "refresh":  # noqa: S105
        expire = datetime.now(UTC) + timedelta(days=30)
    else:
        raise ValueError()
    to_encode.update(
        {
            "exp": expire,
            "iat": datetime.now(UTC),
            "type": token_type,
        }
    )
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=ALGORITHM)
    return encoded_jwt


def http_basic_or_jwt_token(
    form_data: HTTPBasicCredentials = Depends(httpbasic),
    token: str = Depends(oauth2_scheme),
    api_config=Depends(get_api_config),
):
    # Bypass auth entirely when disabled
    if api_config.get("auth_disabled", False):
        return {"username": "anonymous", "roles": ["admin"], "type": "disabled"}
    if token:
        return get_user_from_token(token, api_config.get("jwt_secret_key", "super-secret"))
    elif form_data and verify_auth(api_config, form_data.username, form_data.password):
        # Basic auth path -> elevate to admin by default
        return {"username": form_data.username, "roles": ["admin"], "type": "basic"}

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized",
    )


@router_login.post("/token/login", response_model=AccessAndRefreshToken)
def token_login(
    form_data: HTTPBasicCredentials = Depends(security), api_config=Depends(get_api_config)
):
    # Warn on insecure jwt secret but allow login (tests/legacy configs)
    jwt_secret = api_config.get("jwt_secret_key", "super-secret")
    if jwt_secret in ("super-secret", "super-secret, somethingrandom"):
        logger.warning("Insecure 'jwt_secret_key' configured. Please set a strong random value.")
    if verify_auth(api_config, form_data.username, form_data.password):
        roles = api_config.get("roles", {}).get(form_data.username, ["admin"])  # default admin
        token_data = {"identity": {"u": form_data.username, "roles": roles}}
        access_token = create_token(token_data, jwt_secret, token_type="access")
        refresh_token = create_token(token_data, jwt_secret, token_type="refresh")
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )


@router_login.post("/token/refresh", response_model=AccessToken)
def token_refresh(token: str = Depends(oauth2_scheme), api_config=Depends(get_api_config)):
    # Refresh token
    user = get_user_from_token(
        token, api_config.get("jwt_secret_key", "super-secret"), "refresh"
    )
    token_data = {"identity": {"u": user.get("username"), "roles": user.get("roles", [])}}
    access_token = create_token(
        token_data,
        api_config.get("jwt_secret_key", "super-secret"),
        token_type="access",  # noqa: S106
    )
    return {"access_token": access_token}


def require_admin_for_write(
    request: Request,
    user: dict = Depends(http_basic_or_jwt_token),
):
    """
    RBAC guard: allow GET for any authenticated user; require 'admin' for mutating methods.
    """
    if request.method in {"GET", "OPTIONS", "HEAD"}:
        return True
    roles = set(user.get("roles", []))
    if "admin" in roles:
        return True
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
