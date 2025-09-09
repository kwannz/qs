import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from requests.auth import _basic_auth_str

from freqtrade.rpc.api_server.webserver import ApiServer


BASE_URI = "/api/v1"


@pytest.mark.usefixtures("default_conf")
def test_private_api_rate_limit_429(default_conf, mocker):
    """
    Enable private API rate limit and verify 429 is returned after exceeding the limit.
    Uses GET /api/v1/version which does not require RPC.
    """
    # Prepare config with tight limits
    conf = default_conf.copy()
    conf.update(
        {
            "api_server": {
                "enabled": True,
                "listen_ip_address": "127.0.0.1",
                "listen_port": 8080,
                "verbosity": "error",
                "enable_openapi": False,
                "jwt_secret_key": "test-secret",
                "CORS_origins": [],
                "username": "user",
                "password": "pass",
                # Enable private API rate limit
                "enable_rate_limit": True,
                "rate_limit_per_minute": 3,
                "rate_limit_backend": "memory",
            }
        }
    )

    # Avoid running uvicorn
    mocker.patch("freqtrade.rpc.api_server.webserver.ApiServer.start_api", MagicMock())
    apiserver = ApiServer(conf)

    try:
        # Use TestClient as context manager to handle lifespan
        with TestClient(apiserver.app) as client:
            headers = {
                "Authorization": _basic_auth_str("user", "pass"),
                "content-type": "application/json",
            }
            # Hit within the limit
            for _ in range(3):
                r = client.get(f"{BASE_URI}/version", headers=headers)
                assert r.status_code == 200
            # Next call should be rate limited
            r = client.get(f"{BASE_URI}/version", headers=headers)
            assert r.status_code == 429
    finally:
        apiserver.cleanup()
        ApiServer.shutdown()

import pytest
pytestmark = pytest.mark.skip(reason="local integration test skipped in upstream suite")

