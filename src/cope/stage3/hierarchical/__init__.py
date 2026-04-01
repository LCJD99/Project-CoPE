from src.cope.stage3.hierarchical.advantage import (
    compute_micro_advantages,
    fuse_hierarchical_advantage,
)
from src.cope.stage3.hierarchical.cache import CounterfactualCache
from src.cope.stage3.hierarchical.counterfactual import (
    build_counterfactual_cache_key,
    generate_counterfactual_candidates,
)
from src.cope.stage3.hierarchical.decision_units import (
    DecisionExtractionResult,
    DecisionUnit,
    extract_decision_units,
)
from src.cope.stage3.hierarchical.prefix_signature import (
    build_prefix_signature,
    tool_category_from_token,
)

__all__ = [
    "DecisionExtractionResult",
    "DecisionUnit",
    "CounterfactualCache",
    "build_prefix_signature",
    "build_counterfactual_cache_key",
    "compute_micro_advantages",
    "extract_decision_units",
    "fuse_hierarchical_advantage",
    "generate_counterfactual_candidates",
    "tool_category_from_token",
]
