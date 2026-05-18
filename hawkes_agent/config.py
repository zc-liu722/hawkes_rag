from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
# Downloaded once from Hugging Face (Qwen/Qwen3-Reranker-0.6B) into this folder.
LOCAL_QWEN_RERANKER_DIR = _REPO_ROOT / "models" / "Qwen3-Reranker-0.6B"
DEFAULT_QWEN_RERANKER_MODEL = str(LOCAL_QWEN_RERANKER_DIR)


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
    tau_h: float = 0.05
    tau_r: float = 0.10
    alpha: float = 0.5
    theta_flat: float = 0.85
    hot_margin_threshold: float = 0.05
    hot_entropy_threshold: float = 0.90
    intermediate_top_k: int = 20
    final_top_k: int = 10
    hot_top_k: int = 3
    cold_top_k: int = 3
    #: Optional cap on reranker batch size (hot pass only). ``<= 0`` reranks every
    #: coarse hot hit (up to ``intermediate_top_k``; same cap as cold-merge budget).
    rerank_top_k: int = 0
    min_hot_injected: int = 3
    hot_lambda_threshold: float = 0.10
    theta_a: float = 0.55
    theta_c: float = 0.65
    contradiction_top_k: int = 8
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
    reranker: str = DEFAULT_QWEN_RERANKER_MODEL


@dataclass(frozen=True)
class AgentHarnessConfig:
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    models: ModelRoutingConfig = field(default_factory=ModelRoutingConfig)
    adoption_method: str = "embedding"
    reranker_backend: str = "off"
    reranker_model: str | None = None
    enable_contradiction_micro: bool = False
    enable_dreaming: bool = False
