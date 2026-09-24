"""Saved cash-workspace contracts and the owner-scoped RPC adapter."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ginseng.supabase import SupabaseGateway, SupabaseUnavailableError

MAX_WORKSPACE_REVISION = 9_007_199_254_740_991
MAX_EXPECTED_REVISION = MAX_WORKSPACE_REVISION - 1
MAX_ABS_BALANCE_CENTS = 100_000_000_000
MAX_BILL_CENTS = 100_000_000_000
MIN_WORKSPACE_DATE = date(1900, 1, 1)
MAX_WORKSPACE_DATE = date(2100, 12, 31)

StrictInt = Annotated[int, Field(strict=True)]
WorkspaceDate = Annotated[date, Field(ge=MIN_WORKSPACE_DATE, le=MAX_WORKSPACE_DATE)]


class CashAccount(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    name: Annotated[str, Field(min_length=1, max_length=100)]
    kind: Literal["checking", "savings"]
    balance_cents: Annotated[StrictInt, Field(ge=-MAX_ABS_BALANCE_CENTS, le=MAX_ABS_BALANCE_CENTS)]

    @field_validator("name")
    @classmethod
    def name_is_trimmed(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("Account names cannot start or end with whitespace.")
        return value


class CashBill(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    label: Annotated[str, Field(min_length=1, max_length=100)]
    amount_cents: Annotated[StrictInt, Field(ge=1, le=MAX_BILL_CENTS)]
    due_date: WorkspaceDate

    @field_validator("label")
    @classmethod
    def label_is_trimmed(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("Bill labels cannot start or end with whitespace.")
        return value


class CashWorkspace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: Annotated[StrictInt, Field(ge=0, le=MAX_WORKSPACE_REVISION)]
    as_of: WorkspaceDate | None
    currency: Literal["USD"]
    accounts: list[CashAccount] = Field(max_length=50)
    bills: list[CashBill] = Field(max_length=200)


class SaveWorkspaceRequest(BaseModel):
    """Full replacement snapshot written through the owner-scoped RPC."""

    model_config = ConfigDict(extra="forbid")

    expected_revision: Annotated[StrictInt, Field(ge=0, le=MAX_EXPECTED_REVISION)]
    as_of: WorkspaceDate
    accounts: list[CashAccount] = Field(max_length=50)
    bills: list[CashBill] = Field(max_length=200)



class CashWorkspaceRepository:
    """Repository that forwards the authenticated caller's JWT to Supabase RPCs."""

    def __init__(self, gateway: SupabaseGateway | None = None) -> None:
        self._gateway = gateway or SupabaseGateway()

    def get(self, access_token: str) -> CashWorkspace:
        payload = self._gateway.rpc("get_cash_workspace", {}, access_token)
        try:
            return CashWorkspace.model_validate(payload)
        except ValidationError as error:
            raise SupabaseUnavailableError("Workspace service returned an invalid response.") from error

    def save(self, request: SaveWorkspaceRequest, access_token: str) -> CashWorkspace:
        payload = self._gateway.rpc(
            "save_cash_workspace",
            {
                "p_expected_revision": request.expected_revision,
                "p_as_of": request.as_of.isoformat(),
                "p_accounts": [account.model_dump(mode="json") for account in request.accounts],
                "p_bills": [bill.model_dump(mode="json") for bill in request.bills],
            },
            access_token,
        )
        try:
            return CashWorkspace.model_validate(payload)
        except ValidationError as error:
            raise SupabaseUnavailableError("Workspace service returned an invalid response.") from error


