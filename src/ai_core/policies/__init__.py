"""Políticas comuns do ai-core."""

from __future__ import annotations

from ai_core.policies.approval import ApprovalMode, ApprovalPolicy
from ai_core.policies.privacy import PrivacyMode, PrivacyPolicy
from ai_core.policies.routing import (
    NO_DESTINATION_MESSAGE,
    RoutingPolicy,
    RoutingRule,
)

__all__ = [
    "NO_DESTINATION_MESSAGE",
    "ApprovalMode",
    "ApprovalPolicy",
    "PrivacyMode",
    "PrivacyPolicy",
    "RoutingPolicy",
    "RoutingRule",
]
