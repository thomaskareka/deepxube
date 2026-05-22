from typing import Dict, List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray

from dataclasses import dataclass
from pathlib import Path

WALKABLE_TILES = {"."}

MOVE_DELTAS: Dict[int, Tuple[int, int]] = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
    4: (0, 0),
}

#utils for map environment
#bucket map_file    width   height  start_x start_y goal_x  goal_y  optimal_len
@dataclass(frozen=True)
class ScenarioEntry:
    bucket: int
    map_name: str
    width: int
    height: int
    start_x: int
    start_y: int
    goal_x: int
    goal_y: int
    optimal_length: float

def read_scenario_file(path: str | Path) -> List[ScenarioEntry]:
    path = Path(path)
    lines = path.read_text().splitlines()

    entries: List[ScenarioEntry] = []

    for row in lines:
        line = row.strip()

        if not line: 
            continue
        if line.lower().startswith("version"):
            continue

        parts = line.split()
        if len(parts) != 9:
            raise ValueError(f"invalid row in {path}, {row}")
        
        entry: ScenarioEntry = ScenarioEntry(
            bucket=int(parts[0]),
            map_name=parts[1],
            width=int(parts[2]),
            height=int(parts[3]),
            start_x=int(parts[4]),
            start_y=int(parts[5]),
            goal_x=int(parts[6]),
            goal_y=int(parts[7]),
            optimal_length=float(parts[8]),
        )
        entries.append(entry)
    
    return entries
        
def read_map_file(path: str | Path) -> Tuple[NDArray[np.bool], int, int]:
    path = Path(path)
    lines = path.read_text().splitlines()

    width: Optional[int] = 0
    height: Optional[int] = 0
    map_start: Optional[int] = 0

    for i, line in enumerate(lines):
        line = line.strip()
        lower = line.lower()
        if lower.startswith("width"):
            width = int(line.split()[1])
        elif lower.startswith("height"):
            height = int(line.split()[1])
        elif lower == "map":
            map_start = i + 1
            break
    
    if width is None or height is None or map_start is None:
        raise ValueError(f"{path} is invalid")
    
    map_end = map_start + height
    rows = [line.rstrip("\n") for line in lines[map_start:map_end]]

    if len(rows) != height:
        raise ValueError(f"expected {height} rows, got {len(rows)} in {path}")
    
    obstacles = np.zeros((height,width), dtype=bool)

    for y, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(f"expected width {width} got {len(row)} in {path}")
        for x, character in enumerate(row):
            obstacles[y, x] = character not in WALKABLE_TILES
    
    return obstacles, width, height

# prevent vertex and edge conflicts, iterates through until no swaps needed
def resolve_conflicts(current: List[Tuple[int, int]],
    proposed: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    n = len(current)
    proposed = list(proposed)

    while True:
        changed = False
        blocked = set()
        proposed_agents: Dict[Tuple[int, int], int] = {}
        #vertex, ensure no two agents are on same cell
        for i, target in enumerate(proposed):
            a = proposed_agents.get(target)

            if a is None:
                proposed_agents[target] = i
            else:
                blocked.add(i)
                blocked.add(a)
        
        #edge conflicts, ensure no swapping
        agent_moves: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int,] = {}

        for i in range(n):
            start = current[i]
            end = proposed[i]
            if start != end:
                agent_moves[(start, end)] = i
        
        for i in range(n):
            start = current[i]
            end = proposed[i]

            if start == end: continue

            reverse_agent = agent_moves.get((end, start))

            if reverse_agent is not None and reverse_agent != i:
                blocked.add(i)
                blocked.add(reverse_agent)
        
        if not blocked:
            return proposed
        
        for i in blocked:
            if proposed[i] != current[i]:
                proposed[i] = current[i]
                changed = True
        
        if not changed: return proposed