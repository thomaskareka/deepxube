from typing import List, Tuple, Dict, Any, Optional, Type
import numpy as np
from matplotlib.figure import Figure
from torch import nn, Tensor

from deepxube.base.factory import Parser
from deepxube.base.domain import State, Action, Goal, ActsEnumFixed, StartGoalWalkable, StateGoalVizable, StringToAct
from deepxube.base.nnet_input import StateGoalIn, HasFlatSGActsEnumFixedIn, HasFlatSGAIn
from deepxube.base.heuristic import HeurNNet
from deepxube.nnet.pytorch_models import Conv2dModel, FullyConnectedModel
from deepxube.factories.heuristic_factory import heuristic_factory

from deepxube.factories.domain_factory import domain_factory
from deepxube.factories.nnet_input_factory import register_nnet_input

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from numpy.typing import NDArray

import re

from dataclasses import dataclass
from pathlib import Path

WALKABLE_TILES = {"."}

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




# Define states, goals, and actions
class MapGridState(State):
    def __init__(self, robot_x: int, robot_y: int):
        self.robot_x: int = robot_x
        self.robot_y: int = robot_y

    def __hash__(self) -> int:
        return hash(self.robot_x + self.robot_y)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MapGridState):
            return (self.robot_x == other.robot_x) and (self.robot_y == other.robot_y)
        return NotImplemented


class MapGridGoal(Goal):
    def __init__(self, robot_x: int, robot_y: int):
        self.robot_x: int = robot_x
        self.robot_y: int = robot_y


class MapGridAction(Action):
    def __init__(self, action: int):
        self.action = action

    def __hash__(self) -> int:
        return self.action

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MapGridAction):
            return self.action == other.action
        return NotImplemented

    def __repr__(self) -> str:
        return f"{self.action}"


@domain_factory.register_class("mapgrid")
class MapGrid(ActsEnumFixed[MapGridState, MapGridAction, MapGridGoal], StartGoalWalkable[MapGridState, MapGridAction, MapGridGoal],
           StateGoalVizable[MapGridState, MapGridAction, MapGridGoal], StringToAct[MapGridState, MapGridAction, MapGridGoal],
           HasFlatSGActsEnumFixedIn[MapGridState, MapGridAction, MapGridGoal], HasFlatSGAIn[MapGridState, MapGridAction, MapGridGoal]):
    def __init__(self, dim: int = 7, map_path: Optional[str] = None, scenario_path: Optional[str] = None, scenario_index: int = 0, randomize_scenarios: bool = False):
        super().__init__()

        self.map_path: Optional[str] = None if map_path is None else str(map_path)
        self.scenario_path: Optional[str] = None if scenario_path is None else str(scenario_path)
        self.scenario_index: int = int(scenario_index)
        self.randomize_scenarios: bool = bool(randomize_scenarios)

        if map_path is None:
            self.dim: int = dim
            self.height: int = dim
            self.width: int = dim
            self.obstacles: NDArray[np.bool] = np.zeros((self.height, self.width), dtype=bool)
        else:
            self.obstacles, self.width, self.height = read_map_file(map_path)
            self.dim = max(self.height, self.width)
        
        self.scenario_entries: Optional[List[ScenarioEntry]] = None

        if scenario_path is not None:
            self.scenario_entries = read_scenario_file(scenario_path)
            self._validate_scenarios()
        
        self.free_cells = self._compute_free_cells()

        if len(self.free_cells) == 0:
            raise ValueError("no free cells in map")
        
        self.actions_fixed: List[MapGridAction] = [MapGridAction(x) for x in [0, 1, 2, 3]]

    def _compute_free_cells(self) -> NDArray[np.int64]:
        free_rows, free_columns = np.where(~self.obstacles)
        return np.stack([free_rows, free_columns], axis=1)
    
    def _in_bounds(self, robot_x: int, robot_y: int) -> bool:
        return (0 <= robot_x < self.height) and (0 <= robot_y < self.width)
    
    def _is_free(self, robot_x: int, robot_y: int) -> bool:
        return self._in_bounds(robot_x, robot_y) and not bool(self.obstacles[robot_x, robot_y])

    def _validate_scenarios(self) -> None:
        if self.scenario_entries is None:
            return
        
        for entry in self.scenario_entries:
            if entry.width != self.width or entry.height != self.height:
                raise ValueError(f"scenario dims {entry.width}x{entry.height} don't match map {self.width}x{self.height}")
            
            #column/row order has to be swapped
            start_robot_x = entry.start_y
            start_robot_y = entry.start_x
            goal_robot_x = entry.goal_y
            goal_robot_y = entry.goal_x

            if not self._is_free(start_robot_x, start_robot_y) or not self._is_free(goal_robot_x, goal_robot_y):
                raise ValueError(f"invalid start {entry}")
        
    def _scenario_state_goal(self) -> Tuple[MapGridState, MapGridGoal]:
        if self.scenario_entries is None:
            raise ValueError("no scenario file")
        if self.randomize_scenarios:
            i = np.random.randint(len(self.scenario_entries))
        else:
            i = min(max(self.scenario_index, 0), len(self.scenario_entries)-1)
        
        entry = self.scenario_entries[i]

        state = MapGridState(robot_x=entry.start_y, robot_y=entry.start_x)
        goal = MapGridGoal(robot_x=entry.goal_y, robot_y=entry.goal_x)

        return state, goal
        


    def is_solved(self, states: List[MapGridState], goals: List[MapGridGoal]) -> List[bool]:
        return [(state.robot_x == goal.robot_x) and (state.robot_y == goal.robot_y) for state, goal in zip(states, goals)]

    def sample_start_states(self, num_states: int) -> List[MapGridState]:
        # return [MapGridState(np.random.randint(self.dim), np.random.randint(self.dim)) for _ in range(num_states)]
        if self.scenario_entries is not None:
            start_state, _ = self._scenario_state_goal()
            return [start_state for _ in range(num_states)]
        
        states: List[MapGridState] = []
        for _ in range(num_states):
            i = int(np.random.randint(len(self.free_cells)))
            robot_x = int(self.free_cells[i, 0])
            robot_y = int(self.free_cells[i, 1])
            states.append(MapGridState(robot_x, robot_y))

        return states

    def next_state(self, states: List[MapGridState], actions: List[MapGridAction]) -> Tuple[List[MapGridState], List[float]]:
        states_next: List[MapGridState] = []
        for state, action in zip(states, actions):
            next_robot_x = state.robot_x
            next_robot_y = state.robot_y
            if action.action == 0:  # up
                next_robot_x = state.robot_x - 1
            elif action.action == 1:  # down
                next_robot_x = state.robot_x + 1
            elif action.action == 2:  # left
                next_robot_y = state.robot_y - 1
            elif action.action == 3:  # right
                next_robot_y = state.robot_y + 1
            
            if self._is_free(next_robot_x, next_robot_y):
                states_next.append(MapGridState(next_robot_x, next_robot_y))
            else:
                states_next.append(MapGridState(state.robot_x, state.robot_y))

        return states_next, [1.0] * len(states_next)

    def sample_goal_from_state(self, states_start: Optional[List[MapGridState]], states_goal: List[MapGridState]) -> List[MapGridGoal]:
        # return [MapGridGoal(state_goal.robot_x, state_goal.robot_y) for state_goal in states_goal]
        if self.scenario_entries is not None:
            _, goal = self._scenario_state_goal()
            return [goal for _ in states_goal]
        
        return [MapGridGoal(state_goal.robot_x, state_goal.robot_y) for state_goal in states_goal]

    def get_input_info_flat_sg(self) -> Tuple[List[int], List[int]]:
        return [4], [self.dim]

    def get_input_info_flat_sga(self) -> Tuple[List[int], List[int]]:
        return [4, 1], [self.dim, self.get_num_acts()]

    def to_np_flat_sg(self, states: List[MapGridState], goals: List[MapGridGoal]) -> List[NDArray]:
        return [np.stack([np.stack([state.robot_x for state in states]), np.stack([state.robot_y for state in states]),
                          np.stack([goal.robot_x for goal in goals]), np.stack([goal.robot_y for goal in goals])], axis=1)]

    def to_np_flat_sga(self, states: List[MapGridState], goals: List[MapGridGoal], actions: List[MapGridAction]) -> List[NDArray]:
        return self.to_np_flat_sg(states, goals) + [np.expand_dims(np.array(self.actions_to_indices(actions)), 1)]

    def actions_to_indices(self, actions: List[MapGridAction]) -> List[int]:
        return [action_i.action for action_i in actions]

    def visualize_state_goal(self, state: MapGridState, goal: MapGridGoal, fig: Figure) -> None:
        ax = plt.axes()

        #0 free, 1 obstacle, 2 robot, 3 goal, 4 robot+goal
        grid: NDArray = np.zeros_like(self.obstacles, dtype=np.int32)
        grid[self.obstacles] = 1
        grid[state.robot_x, state.robot_y] = 2
        grid[goal.robot_x, goal.robot_y] = 3

        if state.robot_x == goal.robot_x and state.robot_y == goal.robot_y:
            grid[state.robot_x, state.robot_y] = 4

        ax.imshow(grid, cmap=ListedColormap(["white", "black", "blue", "red", "green"]), origin="upper")
        fig.add_axes(ax)

    def string_to_action(self, act_str: str) -> Optional[MapGridAction]:
        if act_str in {"0", "1", "2", "3"}:
            return MapGridAction(int(act_str))
        else:
            return None

    def string_to_action_help(self) -> str:
        return "0, 1, 2, or 3 for down, up, right, and left, respectively."

    def get_actions_fixed(self) -> List[MapGridAction]:
        return self.actions_fixed.copy()

    def __repr__(self) -> str:
        if self.map_path is None:
            return f"MapGrid(dim={self.dim})"
        
        return (
            f"MapGrid(map_path={self.map_path!r}, "
            f"scenario_path={self.scenario_path!r}, "
            f"height={self.height}, width={self.width}, "
            f"scenario_index={self.scenario_index})"
        )


@domain_factory.register_parser("mapgrid")
class MapGridParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        args_str = args_str.strip()

        if args_str.isdigit():
            return {"dim": int(args_str)}
        
        kwargs: Dict[str, Any] = {}

        for item in args_str.split(","):
            item = item.strip()

            if not item: continue
            if '=' not in item: continue

            key, value = item.split('=', 1)
            key = key.strip().lower()
            value = value.strip()

            if key == "map":
                kwargs["map_path"] = value
            elif key == "scenario":
                kwargs["scenario_path"] = value
            elif key == "index":
                kwargs["scenario_index"] = int(value)
            elif key == "random":
                kwargs["randomize_scenarios"] = True
        return kwargs

    def help(self) -> str:
        return (
            "an integer dimension, e.g. 'mapgrid.7', or key-value args:\n"
            "mapgrid.map=/maps/Berlin_1_256.map,"
            "scen=/scens/Berlin_1_256.map.scen,index=0,random=false"
        )


@register_nnet_input("mapgrid", "mapgrid_nnet_input")
class MapGridNNetInput(StateGoalIn[MapGrid, MapGridState, MapGridGoal]):
    def get_input_info(self) -> int:
        return self.domain.dim

    def to_np(self, states: List[MapGridState], goals: List[MapGridGoal]) -> List[NDArray]:
        np_rep: NDArray = np.zeros((len(states), 3, self.domain.dim, self.domain.dim))

        np_rep[:, 0, :, :] = self.domain.obstacles

        for idx, (state, goal) in enumerate(zip(states, goals)):
            np_rep[idx, 1, state.robot_x, state.robot_y] = 1
            np_rep[idx, 2, goal.robot_x, goal.robot_y] = 1

        return [np_rep]


@heuristic_factory.register_class("mapgridnet")
class MapGridNet(HeurNNet[MapGridNNetInput]):
    @staticmethod
    def nnet_input_type() -> Type[MapGridNNetInput]:
        return MapGridNNetInput

    def __init__(self, nnet_input: MapGridNNetInput, out_dim: int, q_fix: bool, chan_size: int = 8, fc_size: int = 100):
        super().__init__(nnet_input, out_dim, q_fix)
        # one hots
        self.one_hots: nn.ModuleList = nn.ModuleList()
        
        height: int = self.nnet_input.domain.height
        width: int = self.nnet_input.domain.width

        self.heur: nn.Module = nn.Sequential(
            Conv2dModel(3, [chan_size, chan_size], [3, 3], [1, 1], ["RELU", "RELU"], batch_norms=[True, True]),
            nn.Flatten(),
            FullyConnectedModel(width * height * chan_size, [fc_size], ["RELU"], batch_norms=[True]),
            nn.Linear(fc_size, self.out_dim)
        )

    def _forward(self, inputs: List[Tensor]) -> Tensor:
        x: Tensor = self.heur(inputs[0])
        return x


@heuristic_factory.register_parser("mapgridnet")
class MapGridNetParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        args_str_l: List[str] = args_str.split("_")
        kwargs: Dict[str, Any] = dict()
        for args_str_i in args_str_l:
            channel_re = re.search(r"^(\S+)CH$", args_str_i)
            fc_re = re.search(r"^(\S+)FC$", args_str_i)
            if channel_re is not None:
                kwargs["chan_size"] = int(channel_re.group(1))
            elif fc_re is not None:
                kwargs["fc_size"] = int(fc_re.group(1))
            else:
                raise ValueError(f"Unexpected argument {args_str_i!r}")
        return kwargs

    def help(self) -> str:
        return ("Arguments are delimited by '_' and can be in any order.\n<num>C (number of channels), "
                "<num>FC (width of fully-connected layer), bn (batch_norm), wn (weight_norm).\n"
                "E.g. gridnet.10CH_200FC")
