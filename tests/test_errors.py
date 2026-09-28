from __future__ import annotations

from ai_core.errors import (
    AiCoreError,
    ConfigurationError,
    NoProviderAvailableError,
    PolicyViolationError,
    WorkspaceMismatchError,
)


def test_error_hierarchy() -> None:
    for error_type in (
        NoProviderAvailableError,
        PolicyViolationError,
        WorkspaceMismatchError,
        ConfigurationError,
    ):
        error = error_type("mensagem")
        assert isinstance(error, AiCoreError)
        assert isinstance(error, Exception)
        assert str(error) == "mensagem"
