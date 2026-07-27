"""The output contract, and the guard that keeps technical data out of it.

:data:`SECTIONS` is what a valid document contains.  :func:`validate` checks
both directions: every required section is present, and nothing forbidden has
crept in.

The forbidden list is enforced, not documented
----------------------------------------------
The whole point of this engine is that it emits *editable colour state* and not
statistics.  A rule like that decays — someone adds a variance "just for
debugging", a caller starts depending on it, and two releases later the output
is a feature dump again.  So the ban is a test that runs on every document:
:data:`FORBIDDEN_KEYS` is matched against every key at every depth, and any
array longer than :data:`MAX_ARRAY_LENGTH` is rejected on the grounds that
nothing an editor adjusts is a thousand numbers long.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Tuple

#: Top-level sections of a valid document.
SECTIONS: Tuple[str, ...] = (
    "meta", "look", "white_balance", "tone", "color",
    "wheels", "split_toning", "palette", "skin_tone", "hsl",
)

#: The seven HSL bands, in panel order.
HSL_BANDS: Tuple[str, ...] = (
    "red", "orange", "yellow", "green", "cyan", "blue", "purple",
)

#: The three colour wheels.
WHEELS: Tuple[str, ...] = ("lift", "gamma", "gain")

#: Distinctive words banned anywhere inside a key, matched case-insensitively.
FORBIDDEN_SUBSTRINGS: Tuple[str, ...] = (
    "histogram", "entropy", "variance", "kurtosis", "skew", "moment",
    "feature_vector", "feature_names", "percentile", "stddev",
)

#: Short, ambiguous names banned only as a whole ``_``-separated token. Matching
#: these as substrings would reject innocent keys — "lab" is inside "label",
#: "std" is inside "standard" — so the check is on token boundaries.
FORBIDDEN_TOKENS: Tuple[str, ...] = (
    "lab", "xyz", "ycrcb", "cdf", "std", "var", "peak", "peaks",
    "valley", "valleys", "bins", "hist",
)

#: Longest list allowed anywhere. The palette is a handful of swatches; anything
#: longer is a distribution that does not belong in an editor-facing document.
MAX_ARRAY_LENGTH = 16


class SchemaError(ValueError):
    """Raised when a document violates the contract."""


def validate(document: Mapping[str, Any]) -> None:
    """Check ``document`` against the contract.

    Raises
    ------
    SchemaError
        On a missing section, an unexpected section, a forbidden key anywhere in
        the tree, or an over-long array.
    """
    missing = set(SECTIONS) - set(document)
    if missing:
        raise SchemaError(f"missing sections: {sorted(missing)}")
    extra = set(document) - set(SECTIONS)
    if extra:
        raise SchemaError(f"unexpected sections: {sorted(extra)}")

    if tuple(document["hsl"]) != HSL_BANDS:
        raise SchemaError(f"hsl must contain exactly {HSL_BANDS}, got {tuple(document['hsl'])}")
    if tuple(document["wheels"]) != WHEELS:
        raise SchemaError(f"wheels must contain exactly {WHEELS}, got {tuple(document['wheels'])}")

    _walk(document, path="")


def _walk(node: Any, path: str) -> None:
    """Recursively enforce the key ban and the array-length limit."""
    if isinstance(node, Mapping):
        for key, value in node.items():
            _check_key(str(key), path)
            _walk(value, f"{path}.{key}" if path else str(key))
        return

    if isinstance(node, (list, tuple)):
        if len(node) > MAX_ARRAY_LENGTH:
            raise SchemaError(
                f"{path or '<root>'} has {len(node)} entries; "
                f"arrays longer than {MAX_ARRAY_LENGTH} are statistical data, not editable state"
            )
        for index, value in enumerate(node):
            _walk(value, f"{path}[{index}]")


def _check_key(key: str, path: str) -> None:
    """Reject a key whose name names a statistical quantity."""
    lowered = key.lower()
    location = f"{path}.{key}" if path else key

    for banned in FORBIDDEN_SUBSTRINGS:
        if banned in lowered:
            raise SchemaError(
                f"forbidden key {location!r}: this engine emits editable colour "
                f"state, not '{banned}' data"
            )

    tokens = set(re.split(r"[^a-z0-9]+", lowered))
    for banned in FORBIDDEN_TOKENS:
        if banned in tokens:
            raise SchemaError(
                f"forbidden key {location!r}: this engine emits editable colour "
                f"state, not '{banned}' data"
            )
