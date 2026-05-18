from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DynamicsConfig:
    """Hyperparameters for the λ memory dynamics."""

    beta_by_type: dict[str, float] = field(
        default_factory=lambda: {
            "volatile": 0.20,
            "stable": 0.05,
            "identity": 0.001,
        }
    )
    default_type_class: str = "stable"
    mu_base: float = 0.1
    tau: float = 0.0
    intermediate_top_k: int = 3
    final_top_k: int = 5
    theta_a: float = 0.55
    theta_c: float = 0.78
    contradiction_top_k: int = 3
    cosine_floor: float = 0.0

    def beta_for(self, type_class: str | None) -> float:
        return float(
            self.beta_by_type.get(
                type_class or self.default_type_class,
                self.beta_by_type[self.default_type_class],
            )
        )


@dataclass(frozen=True)
class ModelRoutingConfig:
    """Centralized call-site model mapping.

    The default follows the user's requested backbone. Provider-specific API
    keys stay outside code and are read by LiteLLM from the environment.
    """

    main_llm: str = "deepseek-v4-pro"
    contradiction_micro: str = "deepseek-v4-pro"
    dreaming: str = "deepseek-v4-pro"


@dataclass(frozen=True)
class AgentHarnessConfig:
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    models: ModelRoutingConfig = field(default_factory=ModelRoutingConfig)
    adoption_method: str = "embedding"
    enable_contradiction_micro: bool = False
    enable_dreaming: bool = False
