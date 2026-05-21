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

from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
from numpy.typing import NDArray

import re

from .mapfutils import read_map_file, read_scenario_file, ScenarioEntry

# Define states, goals, and actions
class MultiMapGridState(State):
    def __init__(self, robot_xs: Tuple[int, ...], robot_ys: Tuple[int, ...], turn_index: int = 0, offset: Optional[int] = None):
        self.robot_xs = robot_xs
        self.robot_ys = robot_ys
        self.turn_index = turn_index
        self.offset = offset

        self.occupied = frozenset(zip(self.robot_xs, self.robot_ys))

    def __hash__(self) -> int:
        return hash((self.robot_xs, self.robot_ys, self.turn_index, self.offset))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MultiMapGridState):
            return (self.robot_xs == other.robot_xs
                    and self.robot_ys == other.robot_ys
                    and self.turn_index == other.turn_index
                    and self.offset == other.offset)
        return NotImplemented


class MultiMapGridGoal(Goal):
    def __init__(self, goal_xs: Tuple[int, ...], goal_ys: Tuple[int, ...]):
        self.goal_xs = tuple(int(x) for x in goal_xs)
        self.goal_ys = tuple(int(y) for y in goal_ys)


class MultiMapGridAction(Action):
    def __init__(self, action: int):
        self.action = action

    def __hash__(self) -> int:
        return self.action

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MultiMapGridAction):
            return self.action == other.action
        return NotImplemented

    def __repr__(self) -> str:
        return f"{self.action}"


@domain_factory.register_class("multimapgrid")
class MultiMapGrid(ActsEnumFixed[MultiMapGridState, MultiMapGridAction, MultiMapGridGoal], StartGoalWalkable[MultiMapGridState, MultiMapGridAction, MultiMapGridGoal],
           StateGoalVizable[MultiMapGridState, MultiMapGridAction, MultiMapGridGoal], StringToAct[MultiMapGridState, MultiMapGridAction, MultiMapGridGoal],
           HasFlatSGActsEnumFixedIn[MultiMapGridState, MultiMapGridAction, MultiMapGridGoal], HasFlatSGAIn[MultiMapGridState, MultiMapGridAction, MultiMapGridGoal]):
    def __init__(self, dim: int = 7, num_agents: int = 2, map_path: Optional[str] = None, scenario_path: Optional[str] = None, scenario_index: int = 0, randomize_scenarios: bool = False):
        super().__init__()

        self.num_agents: int = int(num_agents)
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
        
        if self.num_agents > len(self.free_cells):
            raise ValueError(f"can't place {self.num_agents}, only {len(self.free_cells)} free")
        
        self.actions_fixed: List[MultiMapGridAction] = [MultiMapGridAction(x) for x in [0, 1, 2, 3, 4]]

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
    
    def _scenario_offset(self) -> int:
        if self.scenario_entries is None:
            raise ValueError("no scenario file")
        
        max_offset = len(self.scenario_entries) - self.num_agents
        if max_offset < 0:
            raise ValueError(f"{len(self.scenario_entries)} entries but {self.num_agents} agents")

        if self.randomize_scenarios:
            return int(np.random.randint(max_offset + 1))

        return min(max(self.scenario_index, 0), max_offset)

    def _scenario_state_goal(self, offset: int = 0) -> Tuple[MultiMapGridState, MultiMapGridGoal]:
        if self.scenario_entries is None:
            raise ValueError("no scenario file")
        
        entries = self.scenario_entries[offset:offset+self.num_agents]
        
        robot_xs: List[int] = []
        robot_ys: List[int] = []
        goal_xs: List[int] = []
        goal_ys: List[int] = []

        for entry in entries:
            robot_xs.append(entry.start_y)
            robot_ys.append(entry.start_x)
            goal_xs.append(entry.goal_y)
            goal_ys.append(entry.goal_x)
        
        #collision detection
        starts = list(zip(robot_xs, robot_ys))
        goals = list(zip(goal_xs, goal_ys))

        if (len(set(starts)) != len(starts)) or (len(set(goals)) != len(goals)):
            raise ValueError(f"duplicate in scenario with offset {offset}")

        state = MultiMapGridState(
            robot_xs=tuple(robot_xs), robot_ys= tuple(robot_ys),
            turn_index=0, offset=offset
        )

        goal = MultiMapGridGoal(goal_xs=tuple(goal_xs), goal_ys=tuple(goal_ys))

        return state, goal


    def is_solved(self, states: List[MultiMapGridState], goals: List[MultiMapGridGoal]) -> List[bool]:
        solved: List[bool] = []
        for state, goal in zip(states, goals):
            solved.append(state.robot_xs == goal.goal_xs and state.robot_ys == goal.goal_ys)
        
        return solved

    def sample_start_states(self, num_states: int) -> List[MultiMapGridState]:
        # return [MultiMapGridState(np.random.randint(self.dim), np.random.randint(self.dim)) for _ in range(num_states)]
        if self.scenario_entries is not None:
            states: List[MultiMapGridState] = []
            for _ in range(num_states):
                offset = self._scenario_offset()
                state, _ = self._scenario_state_goal(offset)
                states.append(state)
            return states
        
        states = []

        for _ in range(num_states):
            chosen_indexes = np.random.choice(len(self.free_cells), size=self.num_agents, replace=False)

            robot_xs: List[int] = []
            robot_ys: List[int] = []
        
            for i in chosen_indexes:
                robot_xs.append(int(self.free_cells[i, 0]))
                robot_ys.append(int(self.free_cells[i, 1]))
            
            states.append(
                MultiMapGridState(
                    robot_xs=tuple(robot_xs), robot_ys=tuple(robot_ys),
                    turn_index=0, offset=None
                )
            )

        return states

    def next_state(self, states: List[MultiMapGridState], actions: List[MultiMapGridAction]) -> Tuple[List[MultiMapGridState], List[float]]:
        next_states: List[MultiMapGridState] = []
        for state, action in zip(states, actions):
            agent_index = state.turn_index

            robot_xs = list(state.robot_xs)
            robot_ys = list(state.robot_ys)

            next_robot_x = robot_xs[agent_index]
            next_robot_y = robot_ys[agent_index]
            if action.action == 0:  # up
                next_robot_x -= 1
            elif action.action == 1:  # down
                next_robot_x += 1
            elif action.action == 2:  # left
                next_robot_y -= 1
            elif action.action == 3:  # right
                next_robot_y += 1
            else:
                pass

            current_pos = (robot_xs[agent_index], robot_ys[agent_index])
            next_pos = (next_robot_x, next_robot_y)

            valid_move = True

            if not self._is_free(next_robot_x, next_robot_y):
                valid_move = False
            
            if next_pos != current_pos:
                occupied = set(zip(robot_xs, robot_ys))
                occupied.remove(current_pos)

                if next_pos in occupied:
                    valid_move = False
            
            if valid_move:
                robot_xs[agent_index] = next_robot_x
                robot_ys[agent_index] = next_robot_y
            
            next_index = (state.turn_index + 1) % self.num_agents

            next_states.append(
                MultiMapGridState(
                    robot_xs=tuple(robot_xs), robot_ys=tuple(robot_ys),
                    turn_index=next_index, offset=state.offset
                )
            )
                

        return next_states, [1.0] * len(next_states)

    def sample_goal_from_state(self, states_start: Optional[List[MultiMapGridState]], states_goal: List[MultiMapGridState]) -> List[MultiMapGridGoal]:
        if self.scenario_entries is not None:
            goals: List[MultiMapGridGoal] = []

            for state_goal in states_goal:
                if state_goal.offset is None:
                    offset = self._scenario_offset()
                else:
                    offset = state_goal.offset
                
                _, goal = self._scenario_state_goal(offset)
                goals.append(goal)
            return goals
        
        return [
            MultiMapGridGoal(goal_xs=state_goal.robot_xs, goal_ys=state_goal.robot_ys) for state_goal in states_goal
        ]

    def get_input_info_flat_sg(self) -> Tuple[List[int], List[int]]:
        return [4 * self.num_agents + 1], [self.dim]

    def get_input_info_flat_sga(self) -> Tuple[List[int], List[int]]:
        return [4 * self.num_agents + 1, 1], [self.dim, self.get_num_acts()]

    def to_np_flat_sg(self, states: List[MultiMapGridState], goals: List[MultiMapGridGoal]) -> List[NDArray]:
        arr = np.zeros((len(states), 4 * self.num_agents + 1), dtype=np.int64)

        for row, (state,goal) in enumerate(zip(states,goals)):
            arr[row, 0:self.num_agents] = np.array(state.robot_xs)
            arr[row, self.num_agents:2 * self.num_agents] = np.array(state.robot_ys)
            arr[row, 2 * self.num_agents:3 * self.num_agents] = np.array(goal.goal_xs)
            arr[row, 3 * self.num_agents:4 * self.num_agents] = np.array(goal.goal_ys)
            arr[row, 4 * self.num_agents] = state.turn_index
        
        return [arr]


    def to_np_flat_sga(self, states: List[MultiMapGridState], goals: List[MultiMapGridGoal], actions: List[MultiMapGridAction]) -> List[NDArray]:
        return self.to_np_flat_sg(states, goals) + [np.expand_dims(np.array(self.actions_to_indices(actions)), 1)]

    def actions_to_indices(self, actions: List[MultiMapGridAction]) -> List[int]:
        return [action_i.action for action_i in actions]

    def visualize_state_goal(self, state: MultiMapGridState, goal: MultiMapGridGoal, fig: Figure) -> None:
        ax = plt.axes()

        #0 free, 1 obstacle, 2 inactive agent, 3 goal, 4 current agent, 5 current goal, 6 agent on any goal, 7 agent on right goal
        grid: NDArray = np.zeros_like(self.obstacles, dtype=np.int32)
        grid[self.obstacles] = 1

        for x,y in zip(goal.goal_xs, goal.goal_ys):
            if grid[x, y] == 0:
                grid[x,y] = 3
        
        current_index = state.turn_index
        current_goal = (goal.goal_xs[current_index], goal.goal_ys[current_index])

        if grid[current_goal[0], current_goal[1]] != 1:
            grid[current_goal[0], current_goal[1]] = 5
        
        goal_positions = set(zip(goal.goal_xs, goal.goal_ys))

        for i, (x,y) in enumerate(zip(state.robot_xs, state.robot_ys)):
            if i == current_index and (x,y) == current_goal:
                grid[x,y] = 7
            elif i == current_index:
                grid[x,y] = 4
            elif (x,y) in goal_positions:
                grid[x,y] = 6
            else:
                grid[x,y] = 2
        
        cmap = ListedColormap(["white", "black", "gray", "yellow", "purple", "pink", "orange", "green"])
        norm = BoundaryNorm(np.arange(-0.5, 8.5, 1), cmap.N)
        ax.imshow(grid, cmap=cmap, norm=norm, origin="upper")
        fig.add_axes(ax)

    def string_to_action(self, act_str: str) -> Optional[MultiMapGridAction]:
        if act_str in {"0", "1", "2", "3", "4"}:
            return MultiMapGridAction(int(act_str))
        else:
            return None

    def string_to_action_help(self) -> str:
        return "0, 1, 2, 3,, or 4 for down, up, right, left, and no-op respectively."

    def get_actions_fixed(self) -> List[MultiMapGridAction]:
        return self.actions_fixed.copy()

    def __repr__(self) -> str:
        if self.map_path is None:
            return f"MultiMapGrid(dim={self.dim})"
        
        return (
            f"MultiMapGrid(map_path={self.map_path!r}, "
            f"scenario_path={self.scenario_path!r}, "
            f"height={self.height}, width={self.width}, "
            f"scenario_index={self.scenario_index})"
        )


@domain_factory.register_parser("multimapgrid")
class MultiMapGridParser(Parser):
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
            elif key == "num_agents":
                kwargs["num_agents"] = int(value)
            elif key == "index":
                kwargs["scenario_index"] = int(value)
            elif key == "random":
                kwargs["randomize_scenarios"] = True
        return kwargs

    def help(self) -> str:
        return (
            "an integer dimension, e.g. 'mapgrid.7', or key-value args:\n"
            "mapgrid.map=/maps/Berlin_1_256.map,"
            "scenario=/scens/Berlin_1_256.map.scen,num_agents=10,index=0,random=false"
        )


@register_nnet_input("multimapgrid", "multimapgrid_nnet_input")
class MultiMapGridNNetInput(StateGoalIn[MultiMapGrid, MultiMapGridState, MultiMapGridGoal]):
    def get_input_info(self) -> int:
        return self.domain.dim

    #0 obstacles, 1 agents, 2 goals, 3 current agent, 4 current goal, 5 agents that have moved
    def to_np(self, states: List[MultiMapGridState], goals: List[MultiMapGridGoal]) -> List[NDArray]:
        np_rep: NDArray = np.zeros((len(states), 6, self.domain.height, self.domain.width))

        np_rep[:, 0, :, :] = self.domain.obstacles.astype(np.float32)

        for batch_idx, (state, goal) in enumerate(zip(states, goals)):
            current_idx = state.turn_index

            for agent_idx, (rx, ry) in enumerate(zip(state.robot_xs, state.robot_ys)):
                np_rep[batch_idx, 1, rx, ry] = 1.0

                if agent_idx < current_idx:
                    np_rep[batch_idx, 5, rx, ry] = 1.0

            for gx, gy in zip(goal.goal_xs, goal.goal_ys):
                np_rep[batch_idx, 2, gx, gy] = 1.0

            current_rx = state.robot_xs[current_idx]
            current_ry = state.robot_ys[current_idx]
            current_gx = goal.goal_xs[current_idx]
            current_gy = goal.goal_ys[current_idx]

            np_rep[batch_idx, 3, current_rx, current_ry] = 1.0
            np_rep[batch_idx, 4, current_gx, current_gy] = 1.0

        return [np_rep]


@heuristic_factory.register_class("multimapgridnet")
class MultiMapGridNet(HeurNNet[MultiMapGridNNetInput]):
    @staticmethod
    def nnet_input_type() -> Type[MultiMapGridNNetInput]:
        return MultiMapGridNNetInput

    def __init__(self, nnet_input: MultiMapGridNNetInput, out_dim: int, q_fix: bool, chan_size: int = 8, fc_size: int = 100):
        super().__init__(nnet_input, out_dim, q_fix)
        # one hots
        self.one_hots: nn.ModuleList = nn.ModuleList()
        
        height: int = self.nnet_input.domain.height
        width: int = self.nnet_input.domain.width

        self.heur: nn.Module = nn.Sequential(
            Conv2dModel(6, [chan_size, chan_size], [3, 3], [1, 1], ["RELU", "RELU"], batch_norms=[True, True]),
            nn.Flatten(),
            FullyConnectedModel(width * height * chan_size, [fc_size], ["RELU"], batch_norms=[True]),
            nn.Linear(fc_size, self.out_dim)
        )

    def _forward(self, inputs: List[Tensor]) -> Tensor:
        x: Tensor = self.heur(inputs[0])
        return x


@heuristic_factory.register_parser("multimapgridnet")
class MultiMapGridNetParser(Parser):
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
