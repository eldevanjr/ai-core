from __future__ import annotations

import pytest

from ai_core.errors import PolicyViolationError
from ai_core.models.common import ProviderLocation
from ai_core.models.destination import DestinationRef
from ai_core.models.proposal import ApprovalRecord
from ai_core.models.source import Source, SourceKind
from ai_core.models.task import Task, TaskPriority, TaskType
from ai_core.policies.approval import ApprovalMode, ApprovalPolicy
from ai_core.policies.privacy import PrivacyMode, PrivacyPolicy
from ai_core.policies.routing import NO_DESTINATION_MESSAGE, RoutingPolicy, RoutingRule


def test_privacy_local_only() -> None:
    policy = PrivacyPolicy(mode=PrivacyMode.LOCAL_ONLY)
    assert policy.allows(ProviderLocation.LOCAL) is True
    assert policy.allows(ProviderLocation.CLOUD) is False
    assert policy.requires_confirmation(ProviderLocation.CLOUD) is False
    policy.assert_allowed(ProviderLocation.LOCAL)
    with pytest.raises(PolicyViolationError):
        policy.assert_allowed(ProviderLocation.CLOUD)


def test_privacy_local_first() -> None:
    policy = PrivacyPolicy(mode=PrivacyMode.LOCAL_FIRST)
    assert policy.allows(ProviderLocation.LOCAL) is True
    assert policy.allows(ProviderLocation.CLOUD) is True
    assert policy.requires_confirmation(ProviderLocation.CLOUD) is True
    assert policy.requires_confirmation(ProviderLocation.LOCAL) is False
    policy.assert_allowed(ProviderLocation.CLOUD)


def test_privacy_allow_cloud() -> None:
    policy = PrivacyPolicy(mode=PrivacyMode.ALLOW_CLOUD)
    assert policy.allows(ProviderLocation.CLOUD) is True
    assert policy.requires_confirmation(ProviderLocation.CLOUD) is False


@pytest.mark.parametrize(
    ("mode", "external", "expected"),
    [
        (ApprovalMode.ALWAYS_REQUIRED, True, True),
        (ApprovalMode.ALWAYS_REQUIRED, False, True),
        (ApprovalMode.REQUIRED_FOR_EXTERNAL_ACTION, True, True),
        (ApprovalMode.REQUIRED_FOR_EXTERNAL_ACTION, False, False),
        (ApprovalMode.AUTO_APPROVED, True, False),
        (ApprovalMode.AUTO_APPROVED, False, False),
        (ApprovalMode.MANUAL_ONLY, True, True),
        (ApprovalMode.MANUAL_ONLY, False, True),
    ],
)
def test_approval_requires_approval(mode: ApprovalMode, external: bool, expected: bool) -> None:
    assert ApprovalPolicy(mode=mode).requires_approval(external=external) is expected


def test_approval_permits_automatic_execution() -> None:
    assert ApprovalPolicy(mode=ApprovalMode.AUTO_APPROVED).permits_automatic_execution() is True
    assert ApprovalPolicy(mode=ApprovalMode.ALWAYS_REQUIRED).permits_automatic_execution() is False


def test_approval_default_mode() -> None:
    assert ApprovalPolicy().mode is ApprovalMode.ALWAYS_REQUIRED


def _task(**kwargs: object) -> Task:
    base: dict[str, object] = {
        "workspace_id": "ws",
        "type": TaskType.BUG,
        "title": "t",
        "priority": TaskPriority.HIGH,
        "labels": ["mobile"],
        "sources": [Source(type=SourceKind.LOG_LINE)],
        "metadata": {"env": "prod"},
    }
    base.update(kwargs)
    return Task(**base)


def test_rule_matches_all_conditions() -> None:
    rule = RoutingRule(
        task_type=TaskType.BUG,
        priority=TaskPriority.HIGH,
        labels=["mobile"],
        source_kind=SourceKind.LOG_LINE,
        metadata={"env": "prod"},
        destination=DestinationRef(provider="JIRA"),
    )
    assert rule.matches(_task()) is True
    assert rule.specificity() == 5


def test_rule_does_not_match_when_destination_disabled() -> None:
    rule = RoutingRule(destination=DestinationRef(provider="JIRA", enabled=False))
    assert rule.matches(_task()) is False


def test_rule_condition_mismatches() -> None:
    jira = DestinationRef(provider="J")
    assert RoutingRule(task_type=TaskType.FEATURE, destination=jira).matches(_task()) is False
    assert RoutingRule(priority=TaskPriority.LOW, destination=jira).matches(_task()) is False
    assert RoutingRule(labels=["backend"], destination=jira).matches(_task()) is False
    transcript = RoutingRule(source_kind=SourceKind.TRANSCRIPT, destination=jira)
    assert transcript.matches(_task()) is False
    assert RoutingRule(metadata={"env": "dev"}, destination=jira).matches(_task()) is False


def test_rule_matches_without_conditions() -> None:
    rule = RoutingRule(destination=DestinationRef(provider="J"))
    assert rule.matches(_task()) is True
    assert rule.specificity() == 0


def test_resolve_manual_destination() -> None:
    policy = RoutingPolicy()
    manual = DestinationRef(provider="GITHUB")
    assert policy.resolve(_task(), manual=manual) is manual


def test_resolve_ignores_disabled_manual_and_uses_default() -> None:
    default = DestinationRef(provider="JIRA")
    policy = RoutingPolicy(default_destination=default)
    disabled = DestinationRef(provider="GITHUB", enabled=False)
    assert policy.resolve(_task(), manual=disabled) is default


def test_resolve_prefers_most_specific_rule() -> None:
    type_only = RoutingRule(task_type=TaskType.BUG, destination=DestinationRef(provider="TYPE"))
    specific = RoutingRule(
        task_type=TaskType.BUG,
        priority=TaskPriority.HIGH,
        destination=DestinationRef(provider="SPECIFIC"),
    )
    policy = RoutingPolicy(rules=[type_only, specific])
    resolved = policy.resolve(_task())
    assert resolved is not None
    assert resolved.provider == "SPECIFIC"


def test_resolve_uses_default_when_no_rule_matches() -> None:
    default = DestinationRef(provider="JIRA")
    policy = RoutingPolicy(
        rules=[RoutingRule(task_type=TaskType.FEATURE, destination=DestinationRef(provider="G"))],
        default_destination=default,
    )
    assert policy.resolve(_task()) is default


def test_resolve_returns_none_without_destination() -> None:
    assert RoutingPolicy().resolve(_task()) is None
    disabled_default = RoutingPolicy(
        default_destination=DestinationRef(provider="JIRA", enabled=False)
    )
    assert disabled_default.resolve(_task()) is None
    assert NO_DESTINATION_MESSAGE


@pytest.mark.parametrize(
    ("mode", "approved", "approved_by", "expected"),
    [
        (ApprovalMode.ALWAYS_REQUIRED, True, None, True),
        (ApprovalMode.ALWAYS_REQUIRED, False, "u", False),
        (ApprovalMode.MANUAL_ONLY, True, "u", True),
        (ApprovalMode.MANUAL_ONLY, False, "u", False),
        (ApprovalMode.AUTO_APPROVED, True, None, True),
    ],
)
def test_approval_is_satisfied_by(
    mode: ApprovalMode, approved: bool, approved_by: str | None, expected: bool
) -> None:
    record = ApprovalRecord(
        mode=ApprovalMode.ALWAYS_REQUIRED, approved=approved, approved_by=approved_by
    )
    assert ApprovalPolicy(mode=mode).is_satisfied_by(record) is expected


def test_manual_only_policy_rejects_anonymous_record() -> None:
    record = ApprovalRecord(mode=ApprovalMode.ALWAYS_REQUIRED, approved=True)
    assert ApprovalPolicy(mode=ApprovalMode.MANUAL_ONLY).is_satisfied_by(record) is False


def test_privacy_accepts_locations_as_strings() -> None:
    policy = PrivacyPolicy(mode=PrivacyMode.LOCAL_ONLY)
    assert policy.allows("LOCAL") is True  # type: ignore[arg-type]
    assert policy.allows("CLOUD") is False  # type: ignore[arg-type]
    assert PrivacyPolicy().requires_confirmation("CLOUD") is True  # type: ignore[arg-type]


def test_resolve_tie_keeps_declaration_order() -> None:
    first = RoutingRule(task_type=TaskType.BUG, destination=DestinationRef(provider="FIRST"))
    second = RoutingRule(priority=TaskPriority.HIGH, destination=DestinationRef(provider="SECOND"))
    resolved = RoutingPolicy(rules=[first, second]).resolve(_task())
    assert resolved is not None
    assert resolved.provider == "FIRST"
