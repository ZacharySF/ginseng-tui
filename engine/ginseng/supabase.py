"""Authenticated, caller-scoped access to Supabase Auth and RPC endpoints.

The publishable key identifies this application to Supabase.  The caller's bearer
JWT is separately forwarded to every RPC request so Postgres evaluates RLS as the
signed-in user; this module never accepts or uses a service-role credential.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import httpx

_HTTP_TIMEOUT = httpx.Timeout(8.0, connect=3.0)


class SupabaseError(RuntimeError):
    """Base class for sanitized Supabase boundary failures."""


class SupabaseConfigurationError(SupabaseError):
    """The engine cannot contact Supabase without its public configuration."""


class SupabaseAuthenticationError(SupabaseError):
    """The supplied bearer token was rejected by Supabase Auth."""


class SupabaseUnavailableError(SupabaseError):
    """A configured Supabase service could not complete a request."""


class SupabaseConflictError(SupabaseError):
    """The workspace RPC reported an optimistic-concurrency conflict."""


class SupabaseValidationError(SupabaseError):
    """The database rejected a workspace request as invalid."""


@dataclass(frozen=True)
class SupabaseConfig:
    url: str
    publishable_key: str

    @classmethod
    def from_environment(cls) -> "SupabaseConfig":
        url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
        publishable_key = os.environ.get("SUPABASE_PUBLISHABLE_KEY", "").strip()
        if not url or not publishable_key:
            raise SupabaseConfigurationError("Supabase is not configured.")
        if not url.startswith(("https://", "http://")):
            raise SupabaseConfigurationError("Supabase is not configured.")
        return cls(url=url, publishable_key=publishable_key)


@dataclass(frozen=True)
class AuthenticatedIdentity:
    user_id: str
    access_token: str


class SupabaseGateway:
    """Small, reusable HTTP boundary with explicit timeouts and safe errors."""

    def __init__(self, config: SupabaseConfig | None = None, client: httpx.Client | None = None) -> None:
        self._config = config
        self._client = client or httpx.Client(timeout=_HTTP_TIMEOUT, follow_redirects=False)

    @property
    def config(self) -> SupabaseConfig:
        return self._config or SupabaseConfig.from_environment()

    def authenticate(self, access_token: str) -> AuthenticatedIdentity:
        config = self.config
        try:
            response = self._client.get(
                f"{config.url}/auth/v1/user",
                headers={
                    "apikey": config.publishable_key,
                    "authorization": f"Bearer {access_token}",
                },
            )
        except httpx.HTTPError as error:
            raise SupabaseUnavailableError("Authentication service is unavailable.") from error

        if response.status_code in (401, 403):
            raise SupabaseAuthenticationError("Your session is invalid. Sign in again.")
        if response.status_code != 200:
            raise SupabaseUnavailableError("Authentication service is unavailable.")
        try:
            payload = response.json()
        except ValueError as error:
            raise SupabaseUnavailableError("Authentication service is unavailable.") from error
        user_id = payload.get("id") if isinstance(payload, dict) else None
        if not isinstance(user_id, str) or not user_id:
            raise SupabaseAuthenticationError("Your session is invalid. Sign in again.")
        return AuthenticatedIdentity(user_id=user_id, access_token=access_token)

    def rpc(self, function_name: str, payload: dict[str, Any], access_token: str) -> dict[str, Any]:
        config = self.config
        try:
            response = self._client.post(
                f"{config.url}/rest/v1/rpc/{function_name}",
                headers={
                    "apikey": config.publishable_key,
                    "authorization": f"Bearer {access_token}",
                    "content-type": "application/json",
                },
                json=payload,
            )
        except httpx.HTTPError as error:
            raise SupabaseUnavailableError("Workspace service is unavailable.") from error

        if response.status_code >= 400:
            self._raise_rpc_error(response)
        try:
            result = response.json()
        except ValueError as error:
            raise SupabaseUnavailableError("Workspace service returned an invalid response.") from error
        if not isinstance(result, dict):
            raise SupabaseUnavailableError("Workspace service returned an invalid response.")
        return result

    @staticmethod
    def _raise_rpc_error(response: httpx.Response) -> None:
        code: str | None = None
        try:
            payload = response.json()
            if isinstance(payload, dict) and isinstance(payload.get("code"), str):
                code = payload["code"]
        except ValueError:
            pass

        if response.status_code in (401, 403):
            raise SupabaseAuthenticationError("Your session is invalid. Sign in again.")
        if code == "40001":
            raise SupabaseConflictError("Workspace changed. Reload before saving.")
        if code == "22023" or (code is not None and code.startswith("23")):
            raise SupabaseValidationError("Workspace data is invalid. Review the amounts and dates.")
        raise SupabaseUnavailableError("Workspace service is unavailable.")
