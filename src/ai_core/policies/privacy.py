"""Política de privacidade: controla o uso de providers cloud."""

from __future__ import annotations

from enum import StrEnum

from ai_core.errors import PolicyViolationError
from ai_core.models.base import CoreModel
from ai_core.models.common import ProviderLocation


class PrivacyMode(StrEnum):
    """Modos de privacidade suportados."""

    LOCAL_ONLY = "LOCAL_ONLY"
    LOCAL_FIRST = "LOCAL_FIRST"
    ALLOW_CLOUD = "ALLOW_CLOUD"


class PrivacyPolicy(CoreModel):
    """Regras aplicáveis ao :class:`ai_core.llm.router.ModelRouter` e afins.

    - ``LOCAL_ONLY``: nunca envia conteúdo para provider cloud.
    - ``LOCAL_FIRST``: prefere local; cloud exige configuração/confirmação explícita.
    - ``ALLOW_CLOUD``: cloud é permitido sem confirmação adicional.

    As comparações usam ``==`` (e não ``is``) para que localizações recebidas como
    string (ex.: vindas de configuração/JSON) sejam avaliadas corretamente.
    """

    mode: PrivacyMode = PrivacyMode.LOCAL_FIRST

    def allows(self, location: ProviderLocation) -> bool:
        """Indica se o provider na localização dada é permitido pela política."""
        if self.mode == PrivacyMode.LOCAL_ONLY:
            return location == ProviderLocation.LOCAL
        return True

    def requires_confirmation(self, location: ProviderLocation) -> bool:
        """Indica se o uso do provider exige confirmação explícita do usuário."""
        return self.mode == PrivacyMode.LOCAL_FIRST and location != ProviderLocation.LOCAL

    def assert_allowed(self, location: ProviderLocation) -> None:
        """Levanta :class:`PolicyViolationError` se a localização não for permitida."""
        if not self.allows(location):
            raise PolicyViolationError(
                f"Política de privacidade {self.mode} proíbe provider {location}."
            )
