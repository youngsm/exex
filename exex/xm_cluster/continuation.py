"""Cooperative continuation policy; checkpoint contents remain application-owned."""

from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True, kw_only=True)
class Continuation:
    """Resume a named output as the same input, within a total attempt budget.

    pause_before is seconds reserved for reaching a safe point, saving, and
    retaining the checkpoint. max_attempts includes the first attempt.
    """

    checkpoint: str
    max_attempts: int
    pause_before: int = 120

    def __post_init__(self):
        if (
            PurePosixPath(self.checkpoint).parts != (self.checkpoint,)
            or self.checkpoint == ".."
        ):
            raise ValueError("checkpoint must be a single artifact name")
        if self.max_attempts < 1 or not 0 < self.pause_before <= 65535:
            raise ValueError(
                "max_attempts must be positive; pause_before must be 1..65535 seconds"
            )
