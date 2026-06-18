from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


class ValType:
    """Minimal gym3.types-compatible base for VPT action-head metadata."""


@dataclass(frozen=True)
class Discrete(ValType):
    n: int


@dataclass(frozen=True)
class Real(ValType):
    pass


@dataclass(frozen=True)
class TensorType(ValType):
    shape: tuple[int, ...]
    eltype: ValType

    @property
    def size(self) -> int:
        return math.prod(self.shape)


class DictType(dict, ValType):
    def __init__(self, **spaces: Any):
        super().__init__(spaces)
        self.spaces = self

