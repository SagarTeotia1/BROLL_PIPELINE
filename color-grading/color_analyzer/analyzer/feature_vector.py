"""Global feature-vector assembly.

Flattens the results of every analyzer into a single, *stably ordered* feature
vector.  Consistency of names/order across runs is guaranteed by:

1. A fixed analyzer order (the ``sections`` list passed in).
2. Deterministic recursion in :meth:`FeatureResult.flatten`.

The result is exposed simultaneously as a NumPy array, an ordered ``dict`` and
JSON — the three shapes an AI video editor typically needs (model input, keyed
lookup, and serialised storage).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Tuple

import numpy as np

from .utils import FeatureResult


@dataclass
class GlobalFeatureVector:
    """A named, ordered numeric feature vector with multiple export shapes."""

    names: List[str] = field(default_factory=list)
    values: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    @classmethod
    def from_sections(cls, sections: Mapping[str, FeatureResult]) -> "GlobalFeatureVector":
        """Build the vector from ``{section_name: FeatureResult}`` in order.

        Each section is flattened with its name as prefix so feature keys look
        like ``lab.warmth_score`` or ``cinematic.teal_orange_score``.
        """
        names: List[str] = []
        values: List[float] = []
        for section, result in sections.items():
            flat = result.flatten(prefix=f"{section}.")
            for key in sorted(flat.keys()):  # deterministic within a section
                names.append(key)
                values.append(flat[key])
        return cls(names=names, values=np.asarray(values, dtype=np.float64))

    # -- export shapes ------------------------------------------------------
    def to_array(self) -> np.ndarray:
        """Return the raw ``float64`` NumPy feature vector."""
        return self.values

    def to_dict(self) -> Dict[str, float]:
        """Return an ordered ``{name: value}`` mapping."""
        return {n: float(v) for n, v in zip(self.names, self.values)}

    def to_json(self, indent: int | None = 2) -> str:
        """Return a JSON object string of the feature mapping."""
        return json.dumps(self.to_dict(), indent=indent)

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def items(self) -> List[Tuple[str, float]]:
        return list(zip(self.names, [float(v) for v in self.values]))
