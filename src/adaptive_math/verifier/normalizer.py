"""Surface normalization for extracted answers.

Representation-preserving cleanup only: NFKC, trim, unicode minus to ASCII,
one outer math delimiter, frac spelling, and two spellings whose alternate form
the parser cannot read at all (``\\degree`` and ``n!^k``). It must never simplify
algebra. Thousands separators are deliberately NOT handled here: distinguishing
"1,234" from set elements like "{99,100}" requires answer-type context, so
they are resolved in the typed numeric parser (verifier.numeric).
"""

import re

from adaptive_math.core.hashing import canonical_text

_FRAC_SPELLING = re.compile(r"\\[dt]frac\b")
_DELIMITER_PAIRS = (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$"))
# latex2sympy rejects \degree outright ("I don't understand this") while it reads
# ^\circ -- and then drops the unit, so 60\degree and 60^\circ both mean 60.
_DEGREE_SPELLING = re.compile(r"\\degree\b")
# 2!^4 means (2!)^4 by convention, but the parser refuses the bare form, which
# made any answer using it unreadable rather than wrong.
_FACTORIAL_POWER = re.compile(r"(\d+)!\^\{?(\d+)\}?")


def normalize_surface(value: str) -> str:
    text = canonical_text(value)
    text = text.replace("\u2212", "-")
    text = _FRAC_SPELLING.sub(r"\\frac", text)
    text = _DEGREE_SPELLING.sub(r"^\\circ ", text)
    text = _FACTORIAL_POWER.sub(r"(\1!)^{\2}", text)
    return _strip_outer_delimiters(text)


def _strip_outer_delimiters(text: str) -> str:
    # fixed-point stripping keeps normalization idempotent for inputs like
    # "$ $x$ $" where one pass exposes another outer delimiter pair.
    while True:
        stripped = _strip_one(text)
        if stripped == text:
            return text
        text = stripped


def _strip_one(text: str) -> str:
    for opener, closer in _DELIMITER_PAIRS:
        if text.startswith(opener) and text.endswith(closer):
            inner = text[len(opener) : len(text) - len(closer)].strip()
            if inner:
                return inner
    return text
