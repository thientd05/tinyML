"""Training drivers that span the whole model roster (as opposed to src/models/<fam>/,
which each train one family)."""
from src.training.multiseed import DEFAULT_FAMILIES, run_multiseed

__all__ = ["DEFAULT_FAMILIES", "run_multiseed"]
