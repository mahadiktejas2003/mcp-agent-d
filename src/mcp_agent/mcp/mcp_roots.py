"""
Types and handlers for MCP root operations.
"""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, AnyUrl, Field

from mcp.types import Request, RequestParams, Result


class RootInfo(BaseModel):
    """Information about a root."""
    uri: AnyUrl
    """The URI of the root."""
    
    name: Optional[str] = None
    """Optional name of the root."""
    
    model_config = ConfigDict(extra="allow")


class ListRootsRequestParams(RequestParams):
    """Parameters for the roots/list request."""
    model_config = ConfigDict(extra="allow")


class ListRootsRequest(Request):
    """Request to list available roots."""
    method: str = "roots/list"
    params: ListRootsRequestParams


class ListRootsResult(Result):
    """Result containing the list of available roots."""
    roots: List[RootInfo]
    """List of roots that are currently available."""
    model_config = ConfigDict(extra="allow")


class RootsListNotificationParams(BaseModel):
    """Parameters for the roots/list notification."""
    roots: List[RootInfo]
    """List of roots that are currently available."""

    model_config = ConfigDict(extra="allow")
    
    meta: dict | None = Field(alias="_meta", default=None)
    """Optional metadata."""


class RootsListNotification(BaseModel):
    """Notification sent when the list of roots changes."""
    method: str = "notifications/roots/list"
    params: RootsListNotificationParams
    
    model_config = ConfigDict(extra="allow")