"""Pydantic schemas for the silo knowledge-graph API response."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class GraphNode(BaseModel):
    """A node in the knowledge graph."""

    id: str
    labels: List[str] = []
    properties: Dict[str, Any] = {}


class GraphEdge(BaseModel):
    """A directed relationship between two graph nodes."""

    id: str
    source: str
    target: str
    type: str
    properties: Dict[str, Any] = {}


class SiloGraphResponse(BaseModel):
    """Full graph payload returned by the GET /{silo_id}/graph endpoint."""

    nodes: List[GraphNode] = []
    edges: List[GraphEdge] = []
    node_count: int = 0
    edge_count: int = 0
    truncated: bool = False
    """True when the result was capped at max_nodes."""
