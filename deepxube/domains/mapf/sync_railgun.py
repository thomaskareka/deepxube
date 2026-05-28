from typing import List, Tuple, Dict, Any, Optional, Type
import numpy as np
import torch.nn.functional as F
import torch
from torch import nn, Tensor

from deepxube.base.factory import Parser
from deepxube.base.heuristic import PolicyNNet

from deepxube.base.nnet_input import PolicyNNetIn

from deepxube.factories.heuristic_factory import policy_factory
from deepxube.factories.nnet_input_factory import register_nnet_input

from numpy.typing import NDArray

import re

from collections import deque

from .mapfutils import MOVE_DELTAS, resolve_conflicts
from .syncmapfgrid import (SyncMAPFGrid, SyncMAPFGridAction, SyncMAPFGridGoal, SyncMAPFGridState)

@register_nnet_input("syncmapfgrid", "syncmapf_railgun_policy_input")
class SyncMAPFRailgunPolicyInput(PolicyNNetIn[SyncMAPFGrid, SyncMAPFGridState, SyncMAPFGridGoal, SyncMAPFGridAction]):
    # eval: [features, agent_coords]
    # training: [features, agent_coords, action_targets, action_mask]
    # channels:
    # 0: obstacles, 1: current agent locations, normalized id
    # 2: goal locations, normalized id
    # 3: individual shortest-path distance at occupied cells
    # 4: row-direction hint, 5: column-direction hint
    # robot_x: row, robot_y: column
    FEATURE_DIM: int = 6
    INF_DISTANCE: float = 2048.0
    def __init__(self, domain: SyncMAPFGrid):
        super().__init__(domain)
        self._distance_cache: Dict[Tuple[int, int], NDArray[np.float32]] = {}
    
    def get_input_info(self) -> Dict[str, int]:
        return {
            "channels": self.FEATURE_DIM,
            "height": self.domain.height,
            "width": self.domain.width,
            "num_agents": self.domain.num_agents
        }

    def states_goals_actions_split_idx(self) -> int:
        return 2
    
    # builds cache for a given goal, or retrieves an array of valid positions
    def _dist_to_goal(self, goal_x: int, goal_y: int) -> NDArray[np.float32]:
        key = (goal_x, goal_y)

        if key in self._distance_cache:
            return self._distance_cache[key]
        
        dist = np.full(
            (self.domain.height, self.domain.width),
            self.INF_DISTANCE,
            dtype=np.float32
        )

        q = deque()
        q.append((goal_x, goal_y))
        dist[goal_x, goal_y] = 0.0

        while q:
            x,y = q.popleft()

            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if not self.domain._is_free(nx, ny):
                    continue

                nd = dist[x,y] + 1.0
                if nd < dist[nx, ny]:
                    dist[nx, ny] = nd
                    q.append((nx, ny))

        self._distance_cache[key] = dist
        return dist

    def _state_goal_to_features(self, state: SyncMAPFGridState, goal: SyncMAPFGridGoal) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
        h, w = self.domain.height, self.domain.width
        n = self.domain.num_agents

        features = np.zeros((self.FEATURE_DIM, h, w), dtype=np.float32)
        coords = np.zeros((n, 2), dtype=np.int64)

        features[0] = self.domain.obstacles.astype(np.float32)

        for i, (rx, ry, gx, gy) in enumerate(
            zip(state.robot_xs, state.robot_ys, goal.goal_xs, goal.goal_ys)
        ):
            rx, ry, gx, gy = int(rx), int(ry), int(gx), int(gy)

            coords[i] = [rx, ry]

            agent_val = float(i + 1) / float(n) #normalize
            features[1, rx, ry] = agent_val
            features[2, gx, gy] = agent_val

            dist = self._dist_to_goal(gx, gy)
            d0 = float(dist[rx, ry])

            features[3, rx, ry] = min(d0, self.INF_DISTANCE) / self.INF_DISTANCE

            best_dx = 0
            best_dy = 0
            best_d = d0

            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = rx + dx, ry + dy

                if not self.domain._is_free(nx, ny):
                    continue

                if dist[nx, ny] < best_d:
                    best_d = float(dist[nx, ny])
                    best_dx = dx
                    best_dy = dy
            
            features[4, rx, ry] = float(best_dx)
            features[5, rx, ry] = float(best_dy)
        
        return features, coords
    
    def to_np_fn(self, states, goals):
        b = len(states)
        h, w = self.domain.height, self.domain.width
        n = self.domain.num_agents

        features = np.zeros((b, self.FEATURE_DIM, h, w), dtype=np.float32)
        coords = np.zeros((b, n, 2), dtype=np.int64)

        for i, (state, goal) in enumerate(zip(states, goals)):
            features[i], coords[i] = self._state_goal_to_features(state, goal)
        
        return [features, coords]

    # def to_np(self, states, goals, actions):
    #     features, coords = self.to_np_fn(states, goals)

    #     b = len(states)
    #     h, w = self.domain.height, self.domain.width

    #     action_targets = np.zeros((b, h, w), dtype=np.int64)
    #     action_mask = np.zeros((b, h, w), dtype=np.float32)

    #     for bi, (state, action) in enumerate(zip(states, actions)):
    #         for ai, (rx, ry) in enumerate(zip(state.robot_xs, state.robot_ys)):
    #             action_targets[bi, rx, ry] = int(action.agent_actions[ai])
    #             action_mask[bi, rx, ry] = 1.0
    #     return [features, coords, action_targets, action_mask]

    # TODO: proper experts for training, like RAILGUN
    # this is just best distance

    def _distance_expert_action(self, state: SyncMAPFGridState, goal: SyncMAPFGridGoal) -> SyncMAPFGridAction:
        agent_actions: List[int] = []
        
        for rx, ry, gx, gy in zip(state.robot_xs, state.robot_ys, goal.goal_xs, goal.goal_ys):
            rx, ry, gx, gy = int(rx), int(ry), int(gx), int(gy)
            dist = self._dist_to_goal(gx, gy)

            best_action = 4
            best_distance = dist[rx, ry]

            for a, (dx, dy) in MOVE_DELTAS.items():
                nx, ny = rx + int(dx), ry + int(dy)

                if not self.domain._is_free(nx, ny):
                    continue
                next_dist = dist[nx, ny]
                if next_dist < best_distance:
                    best_distance = next_dist
                    best_action = a
            agent_actions.append(int(best_action))
        return SyncMAPFGridAction(tuple(agent_actions))
    
    def _delta_to_action_id(self) -> Dict[Tuple[int, int], int]:
        return {
            (int(dx), int(dy)): int(action_id)
            for action_id, (dx, dy) in MOVE_DELTAS.items()
        }

    def to_np(self, states, goals, actions):
        features, coords = self.to_np_fn(states, goals)

        b = len(states)
        h, w = self.domain.height, self.domain.width

        action_targets = np.zeros((b, h, w), dtype=np.int64)
        action_mask = np.zeros((b, h, w), dtype=np.float32)

        for bi, (state, goal) in enumerate(zip(states, goals)):
            expert_action = self._distance_expert_action(state, goal)

            for ai, (rx, ry) in enumerate(zip(state.robot_xs, state.robot_ys)):
                action_targets[bi, rx, ry] = int(expert_action.agent_actions[ai])
                action_mask[bi, rx, ry] = 1.0
                
        return [features, coords, action_targets, action_mask]
    
    def nnet_out_to_actions(self, nnet_out):
        actions_np = np.asarray(nnet_out[0]).astype(np.int64)
        return [
            SyncMAPFGridAction(tuple(int(a) for a in row))
            for row in actions_np
        ]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else: 
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.conv_block(x)
        out = out + self.shortcut(x)
        return self.relu(out)

class RailgunCNN(nn.Module):
    def __init__(self, n_channels: int = 6, n_classes: int = 5, base_channels: int = 64, max_channels: int = 512):
        super().__init__()

        c1 = base_channels
        c2 = min(base_channels * 2, max_channels)
        c3 = min(base_channels * 4, max_channels)
        c4 = min(base_channels * 8, max_channels)
        c5 = min(base_channels * 16, max_channels)

        self.initial_conv = nn.Sequential(
            nn.Conv2d(n_channels, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(c1, c2),
            ResidualBlock(c2, c3),
            ResidualBlock(c3, c4),
            ResidualBlock(c4, c5),
            ResidualBlock(c5, c4),
            ResidualBlock(c4, c3),
            ResidualBlock(c3, c2),
            ResidualBlock(c2, c1),
        )

        self.final_conv = nn.Conv2d(c1, n_classes, kernel_size=1)
    def forward(self, x: Tensor) -> Tensor:
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        logits = self.final_conv(x)
        return logits


@policy_factory.register_class("syncmapf_railgun")
class SyncMAPFRailgunPolicy(PolicyNNet[SyncMAPFRailgunPolicyInput]):
    @staticmethod
    def nnet_input_type() -> Type[SyncMAPFRailgunPolicyInput]:
        return SyncMAPFRailgunPolicyInput
    
    def __init__(self, nnet_input: SyncMAPFRailgunPolicyInput, num_samp, channels: int = 64, max_channels: int = 512, lr: float = 0.001, lr_d: float = 0.9999993):
        super().__init__(nnet_input, num_samp)

        self.net = RailgunCNN(
            n_channels=SyncMAPFRailgunPolicyInput.FEATURE_DIM,
            n_classes=len(MOVE_DELTAS),
            base_channels=channels,
            max_channels=max_channels
        )

        self.channels = int(channels)
        self.max_channels = int(max_channels)

        self.lr = float(lr)
        self.lr_d = float(lr_d)
    
    def get_loss_and_info(self, fwd_tr_tensors, get_info):
        loss_per_state = fwd_tr_tensors[0]
        loss = loss_per_state.mean()

        info: Optional[str] = None
        if get_info:
            info = f"loss: {loss.item():.4f}"

        return loss, info
    
    def _forward_train(self, inputs):
        features = inputs[0].float()
        targets = inputs[2].long()
        mask = inputs[3].float()

        logits = self.net(features)

        loss_per_cell = F.cross_entropy(
            logits, targets, reduction="none"
        )

        loss_per_state = (loss_per_cell * mask).sum(dim=(1, 2)) / mask.sum(
            dim=(1, 2)
        ).clamp_min(1.0)
        
        return [loss_per_state]
    
    def _forward_eval(self, states_goals):
        features = states_goals[0].float()
        coords = states_goals[1].long()

        logits = self.net(features)

        #batch, 5, height, width -> batch, height, width, 5
        logits_hw = logits.permute(0, 2, 3, 1).contiguous()

        batch_size = logits.shape[0]
        num_agents = coords.shape[1]
        height = logits.shape[2]
        width = logits.shape[3]

        rows = coords[:, :, 0].clamp(0, height-1)
        cols = coords[:, :, 1].clamp(0, width-1)

        batch_index = torch.arange(batch_size, device=logits.device).view(batch_size, 1).expand(batch_size, num_agents)

        agent_logits = logits_hw[batch_index, rows, cols, :]
        dist = torch.distributions.Categorical(logits=agent_logits)

        sampled_actions = dist.sample((self.num_samp,))

        sampled_log_probs = dist.log_prob(sampled_actions)
        

        actions = sampled_actions.permute(1, 0, 2).contiguous()

        probs = torch.exp(
            sampled_log_probs.sum(dim=-1).transpose(0,1)
        ).clamp_min(1e-30)

        return [actions.float(), probs]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"channels={self.channels}, "
            f"max_channels={self.max_channels}, "
            f"num_samp={self.num_samp}"
            f"lr={self.lr}, "
            f"lr_d={self.lr_d})"
        )
    
@policy_factory.register_parser("syncmapf_railgun")
class SyncMAPFRailgunPolicyParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        args_str_l: List[str] = args_str.split("_")
        kwargs: Dict[str, Any] = dict()

        for args_str_i in args_str_l:
            channel_re = re.search(r"^(\d+)CH$", args_str_i)
            max_channel_re = re.search(r"^(\d+)MAX$", args_str_i)
            lr_re = re.search(r"^LR([0-9.]+)$", args_str_i)
            lr_d_re = re.search(r"^LRD([0-9.]+)$", args_str_i)

            if channel_re is not None:
                kwargs["channels"] = int(channel_re.group(1))
            elif max_channel_re is not None:
                kwargs["max_channels"] = int(max_channel_re.group(1))
            elif lr_re is not None:
                kwargs["lr"] = float(lr_re.group(1))
            elif lr_d_re is not None:
                kwargs["lr_d"] = float(lr_d_re.group(1))
            else:
                raise ValueError(f"Unexpected argument {args_str_i!r}")

        return kwargs

    def help(self) -> str:
        return (
            "Arguments are delimited by '_' and can be in any order.\n"
            "<num>CH sets base CNN channels, <num>MAX sets max CNN channels.\n"
            "LR<num> sets learning rate, LRD<num> sets learning-rate decay.\n"
            "E.g. syncmapf_railgun.32CH_256MAX_LR0.001_LRD0.9999993"
        )
#TODO: u-net style network like RAILGUN