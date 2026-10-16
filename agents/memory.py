from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class MemoryEntry:
    step: int
    observation: tuple        # compact summary (position, facing, held key)
    action: str               # macro-action label
    outcome: str              # environment feedback
    hypothesis: tuple         # hypothesis id, e.g. ("KD","red","blue")
    prediction: bool          # predicted effect
    error: bool               # prediction != observed effect


class EpisodicMemory:
    """Structured episodic memory. capacity=None: unbounded; 0: none; n: keep last n entries."""

    def __init__(self, capacity: Optional[int] = None):
        self.capacity = capacity
        self.entries = []

    def add(self, e: MemoryEntry):
        if self.capacity == 0:
            return
        self.entries.append(e)
        if self.capacity is not None and len(self.entries) > self.capacity:
            self.entries.pop(0)

    def __len__(self):
        return len(self.entries)
