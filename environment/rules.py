"""Hidden rule definitions. Agents never receive these objects."""
from dataclasses import dataclass, field
from typing import Dict, Tuple

KEY_DOOR = "KEY_DOOR"
SWITCH_BRIDGE = "SWITCH_BRIDGE"
PUSH_BLOCK = "PUSH_BLOCK"
SWITCH_DOOR = "SWITCH_DOOR"   # a door opened by a switch (not a key): tests rule-TYPE discovery
TRAP = "TRAP"
GATE_RULES = (KEY_DOOR, SWITCH_BRIDGE, PUSH_BLOCK, SWITCH_DOOR)
ALL_RULES = GATE_RULES + (TRAP,)
COLORS = ("red", "blue", "green", "yellow", "purple", "orange")
ALL_COLORS = COLORS + ("cyan", "pink")   # extended universe used by regime D and RL encodings


@dataclass(frozen=True)
class HiddenRules:
    rule_types: Tuple[str, ...]
    key_door: Dict[str, str]        # key color -> door color it opens
    switch_bridge: Dict[str, str]   # switch color -> chasm color it bridges
    switch_door: Dict[str, str] = field(default_factory=dict)  # switch color -> door color it opens
    key_door_extra: Dict[str, tuple] = field(default_factory=dict)  # a key that ALSO opens a second door (violates exclusivity)
