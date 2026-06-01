from typing import List, Tuple, Dict, Any, Optional, Type
import numpy as np
from matplotlib.figure import Figure
from torch import nn, Tensor

from deepxube.base.factory import Parser
from deepxube.base.domain import State, Action, Goal, ActsFixed, StartGoalWalkable, StateGoalVizable, StringToAct
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

from .mapfutils import MOVE_DELTAS, read_map_file, read_scenario_file, resolve_conflicts, ScenarioEntry

class SyncMAPFGridState(State):
    def __init__(self, robot_xs: Tuple[int, ...], robot_ys: Tuple[int, ...], offset: Optional[int] = None):
        self.robot_xs = robot_xs
        self.robot_ys = robot_ys
        self.offset = offset
        self.occupied = frozenset(zip(self.robot_xs, self.robot_ys))

    def __hash__(self) -> int:
        return hash((self.robot_xs, self.robot_ys, self.offset))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SyncMAPFGridState):
            return (self.robot_xs == other.robot_xs
                    and self.robot_ys == other.robot_ys
                    and self.offset == other.offset)
        return NotImplemented
        
class SyncMAPFGridGoal(Goal):
    def __init__(self, goal_xs: Tuple[int, ...], goal_ys: Tuple[int, ...]):
        self.goal_xs = goal_xs
        self.goal_ys = goal_ys

class SyncMAPFGridAction(Action):
    def __init__(self, agent_actions: Tuple[int, ...]):
        self.agent_actions = agent_actions
    
    def __hash__(self):
        return hash(self.agent_actions)

    def __eq__(self, other):
        if isinstance(other, SyncMAPFGridAction):
            return self.agent_actions == other.agent_actions
        return NotImplemented

#here to sample_goal_from_state are the same as the other map grid implementations
#using ActsFixed instead of ActsEnumFixed
@domain_factory.register_class("syncmapfgrid")
class SyncMAPFGrid(
    ActsFixed[SyncMAPFGridState, SyncMAPFGridAction, SyncMAPFGridGoal],
    StartGoalWalkable[SyncMAPFGridState, SyncMAPFGridAction, SyncMAPFGridGoal],
    StateGoalVizable[SyncMAPFGridState, SyncMAPFGridAction, SyncMAPFGridGoal],
    StringToAct[SyncMAPFGridState, SyncMAPFGridAction, SyncMAPFGridGoal]
):
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
            self.obstacles: NDArray[np.bool_] = np.zeros((self.height, self.width), dtype=bool)
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

    def _scenario_state_goal(self, offset: int = 0) -> Tuple[SyncMAPFGridState, SyncMAPFGridGoal]:
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

        state = SyncMAPFGridState(
            robot_xs=tuple(robot_xs), robot_ys= tuple(robot_ys), offset=offset
        )

        goal = SyncMAPFGridGoal(goal_xs=tuple(goal_xs), goal_ys=tuple(goal_ys))

        return state, goal
    
    def sample_start_states(self, num_states: int) -> List[SyncMAPFGridState]:
        if self.scenario_entries is not None:
            states: List[SyncMAPFGridState] = []
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
                SyncMAPFGridState(
                    robot_xs=tuple(robot_xs), robot_ys=tuple(robot_ys), offset=None
                )
            )

        return states
    
    def sample_goal_from_state(self, states_start: Optional[List[SyncMAPFGridState]], states_goal: List[SyncMAPFGridState]) -> List[SyncMAPFGridGoal]:
        if self.scenario_entries is not None:
            goals: List[SyncMAPFGridGoal] = []

            for state_goal in states_goal:
                if state_goal.offset is None:
                    offset = self._scenario_offset()
                else:
                    offset = state_goal.offset
                
                _, goal = self._scenario_state_goal(offset)
                goals.append(goal)
            return goals
        return [
            SyncMAPFGridGoal(goal_xs=state_goal.robot_xs, goal_ys=state_goal.robot_ys) for state_goal in states_goal
        ]

    def is_solved(self, states: List[SyncMAPFGridState], goals: List[SyncMAPFGridGoal]) -> List[bool]:
        solved: List[bool] = []
        for state, goal in zip(states, goals):
            solved.append(state.robot_xs == goal.goal_xs and state.robot_ys == goal.goal_ys)
        
        return solved
    # for ActsFixed
    def sample_action(self, num: int) -> List[SyncMAPFGridAction]:
        actions = np.random.randint(low=0, high=len(MOVE_DELTAS), size=(num, self.num_agents))

        return [
            SyncMAPFGridAction(tuple(int(action) for action in actions[row]))
            for row in range(num)
        ]
    
    def _get_wait_action_id(self) -> int:
        for action_id, delta in MOVE_DELTAS.items():
            dx, dy = delta
            if int(dx) == 0 and int(dy) == 0:
                return int(action_id)
        return 0

    def _get_wait_actions(self) -> SyncMAPFGridAction:
        wait_id = self._get_wait_action_id()
        return SyncMAPFGridAction(tuple(wait_id for _ in range(self.num_agents)))
    
    def samp_edges(self, steps_gen):
        if self.scenario_entries is not None:
            raise ValueError("cannot use fixed scenarios for training")
        
        start_states = self.sample_start_states(len(steps_gen))

        next_states, first_actions, _ = self.sample_next_state(start_states)

        steps = np.asarray(steps_gen, dtype=np.int64)

        wait_id = self._get_wait_action_id()
        for i in np.where(steps == 0)[0]:
            index = int(i)
            next_states[index] = start_states[index]
            first_actions[index] = wait_id
        
        reduced_steps = np.maximum(steps-1, 0).tolist()

        states_goal, _, _ = self.random_walk(next_states, reduced_steps)
        goals = self.sample_goal_from_state(start_states, states_goal)

        return start_states, goals, first_actions
    
    def _initial_action_proposals(
            self, state: SyncMAPFGridState, action: SyncMAPFGridAction
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        if len(action.agent_actions) != self.num_agents:
            raise ValueError(f"need {self.num_agents} agent actions and got {len(action.agent_actions)}")
        
        current_positions = list(zip(state.robot_xs, state.robot_ys))
        proposed_positions: List[Tuple[int, int]] = []

        for i, action in enumerate(action.agent_actions):
            dx, dy = MOVE_DELTAS.get(int(action), (0,0))
            cx, cy = current_positions[i]
            nx, ny = cx + dx, cy + dy

            if not self._is_free(nx, ny):
                proposed_positions.append((cx, cy))
            else: 
                proposed_positions.append((nx, ny))
        
        return current_positions, proposed_positions
    
    def next_state(
        self, states: List[SyncMAPFGridState], actions: List[SyncMAPFGridAction],
    ) -> Tuple[List[SyncMAPFGridState], List[float]]:
        next_states: List[SyncMAPFGridState] = []

        for state, action in zip(states, actions):
            current, proposed = self._initial_action_proposals(state, action)
            resolved = resolve_conflicts(current, proposed)

            robot_xs = tuple(int(pos[0]) for pos in resolved)
            robot_ys = tuple(int(pos[1]) for pos in resolved)

            next_states.append(
                SyncMAPFGridState(robot_xs=robot_xs, robot_ys=robot_ys, offset=state.offset)
            )
        return next_states, [1.0] * len(next_states)
    
    def is_solved(self, states: List[SyncMAPFGridState], goals: List[SyncMAPFGridGoal]) -> List[bool]:
        solved: List[bool] = []
        for state, goal in zip(states, goals):
            solved.append(state.robot_xs == goal.goal_xs and state.robot_ys == goal.goal_ys)
        
        return solved
    
    def visualize_state_goal(self, state: SyncMAPFGridState, goal: SyncMAPFGridGoal, fig: Figure) -> None:
        ax = plt.axes()

        #0 free, 1 obstacle, 2 goal, 3 agent, 4 agent on own goal, 5 agent on wrong goal
        grid: NDArray = np.zeros_like(self.obstacles, dtype=np.int32)
        grid[self.obstacles] = 1

        for x,y in zip(goal.goal_xs, goal.goal_ys):
            if grid[x, y] == 0:
                grid[x,y] = 2
        
        goals = set(zip(goal.goal_xs, goal.goal_ys))

        for i, (x,y) in enumerate(zip(state.robot_xs, state.robot_ys)):
            target_goal = (goal.goal_xs[i], goal.goal_ys[i])

            if (x, y) == target_goal:
                grid[x,y] = 4
            elif (x,y) in goals:
                grid[x,y] = 5
            else:
                grid[x,y] = 3
        
        cmap = ListedColormap(["white", "black", "gray", "purple", "green", "red",])
        norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)
        ax.imshow(grid, cmap=cmap, norm=norm, origin="upper")
        fig.add_axes(ax)
    
    def string_to_action(self, act_str) -> Optional[SyncMAPFGridAction]:
        #1 digit, all move, sequence moves individual agents
        s = act_str.strip()

        if s in {"0", "1", "2", "3", "4"}:
            return SyncMAPFGridAction(tuple(int(s) for _ in range(self.num_agents)))
        tokens = re.findall(r"[0-4]", s)
        if len(tokens) != self.num_agents:
            return None

        return SyncMAPFGridAction(tuple(int(t) for t in tokens))
    
    def string_to_action_help(self) -> str:
        return f"0-4 to move each agent, can provide string of {self.num_agents} digits to move individual agents"
    
    def __repr__(self) -> str:
        if self.map_path is None:
            return f"SyncMAPFGrid(dim={self.dim}, num_agents={self.num_agents})"

        return (
            f"SyncMAPFGrid(map_path={self.map_path!r}, "
            f"scenario_path={self.scenario_path!r}, "
            f"height={self.height}, width={self.width}, "
            f"num_agents={self.num_agents}, "
            f"scenario_index={self.scenario_index})"
        )

@domain_factory.register_parser("syncmapfgrid")
class SyncMAPFGridParser(Parser):
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
            elif key == "dim":
                kwargs["dim"] = int(value)
        return kwargs
    def help(self) -> str:
        return (
            "Examples:\n"
            "syncmapfgrid.32\n"
            "syncmapfgrid.map=maps/maze-32-32-2.map,num_agents=50\n"
            "syncmapfgrid.map=maps/maze-32-32-2.map,"
            "syncmapfgrid.map=maps/maze-32-32-2.map,cenario=scens/maze-32-32-2.map.scen,num_agents=50,index=0"
        )