# -*- coding: utf-8 -*-
import math
import random
import numpy as np
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces

@dataclass
class RewardCfg:
    correct: float = 1.0
    incorrect: float = -0.5
    invalid_action: float = -0.05   # small penalty instead of nuking learning
    step_cost: float = -0.001       # discourages dithering
    mastery_bonus: float = 0.25     # when learner crosses mastery threshold

@dataclass
class CurriculumCfg:
    # Prefer items near the forgetting boundary (medium Leitner levels)
    use: bool = True
    p_medium: float = 0.55
    p_low: float = 0.25
    p_high: float = 0.20
    # Mastery threshold per learner (rolling accuracy)
    mastery_window: int = 30
    mastery_threshold: float = 0.85

class MultiLearnerTutorEnv(gym.Env):
    """
    Single-agent PPO controlling a *population* of learners.
    Discrete action = (learner_id * n_items) + item_id
    This patch:
      - Guards invalid actions robustly (no silent no-ops).
      - Emits action_mask in info for masked policies (optional).
      - Adds light curriculum sampling and shaped rewards.
      - Normalizes obs to [-1, 1] ranges.
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 n_learners: int,
                 n_items: int,
                 horizon: int = 600,
                 seed: int = 0,
                 reward_cfg: RewardCfg = RewardCfg(),
                 curriculum_cfg: CurriculumCfg = CurriculumCfg()):
        super().__init__()
        self.last_action_mask = None
        self.rng = np.random.default_rng(seed)
        self.n_learners = n_learners
        self.n_items = n_items
        self.horizon = horizon
        self.reward_cfg = reward_cfg
        self.curr_cfg = curriculum_cfg

        # Example state per learner: [leitner_level, recent_acc, time_since_seen]
        self.obs_dim_per_learner = 3
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_learners * self.obs_dim_per_learner,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_learners * self.n_items)

        # Internal state
        self._t = 0
        self.leitner = np.ones((n_learners, n_items), dtype=np.int32)  # 1..5
        self.seen = np.zeros((n_learners, n_items), dtype=np.int32)
        self.recent = [[] for _ in range(n_learners)]  # rolling correctness
        self.mastered = np.zeros((n_learners, n_items), dtype=np.bool_)

        # Cache of valid actions (all valid by default)
        self._valid_mask = np.ones(self.action_space.n, dtype=np.bool_)

    # ---------- helpers ----------
    def _rolling_acc(self, learner_id: int) -> float:
        buf = self.recent[learner_id]
        if not buf:
            return 0.0
        return float(np.mean(buf[-self.curr_cfg.mastery_window:]))

    def _curriculum_pick(self, learner_id: int) -> int:
        """Pick an item index with preference to ‘medium’ Leitner levels."""
        if not self.curr_cfg.use:
            return int(self.rng.integers(0, self.n_items))
        levels = self.leitner[learner_id]
        low = np.where(levels <= 2)[0]
        med = np.where((levels >= 3) & (levels <= 4))[0]
        high = np.where(levels >= 5)[0]
        r = self.rng.random()
        if med.size and r < self.curr_cfg.p_medium:
            return int(self.rng.choice(med))
        if low.size and r < self.curr_cfg.p_medium + self.curr_cfg.p_low:
            return int(self.rng.choice(low))
        if high.size:
            return int(self.rng.choice(high))
        return int(self.rng.integers(0, self.n_items))

    def _decode_action(self, a: int):
        learner_id = a // self.n_items
        item_id = a % self.n_items
        return learner_id, item_id

    def _encode_action(self, learner_id: int, item_id: int):
        return learner_id * self.n_items + item_id

    def _update_valid_mask(self):
        # Example: if an item is already mastered for a learner, mark invalid
        self.last_action_mask = self._valid_mask.copy()
        mask = np.ones(self.action_space.n, dtype=np.bool_)
        for l in range(self.n_learners):
            for it in range(self.n_items):
                if self.mastered[l, it]:
                    mask[self._encode_action(l, it)] = False
        self._valid_mask = mask

    def _obs(self):
        # Build per-learner features and squish to [-1, 1]
        feats = []
        for l in range(self.n_learners):
            # Normalize leitner to [-1,1] assuming 1..7
            lv = np.clip(self.leitner[l].mean(), 1, 7)
            leitner_norm = (lv - 4) / 3.0
            acc_norm = (self._rolling_acc(l) - 0.5) * 2.0  # [0,1] -> [-1,1]
            # Normalize recency: cap at horizon
            rec = np.clip(self.seen[l].mean(), 0, self.horizon)
            rec_norm = (rec / self.horizon) * 2.0 - 1.0
            feats.extend([leitner_norm, acc_norm, rec_norm])
        return np.asarray(feats, dtype=np.float32)

    # ---------- gym API ----------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._t = 0
        self.leitner[:] = 1
        self.seen[:] = 0
        self.mastered[:] = False
        for l in range(self.n_learners):
            self.recent[l] = []
        self._update_valid_mask()
        obs = self._obs()
        info = {"action_mask": self._valid_mask.copy()}
        self.last_action_mask = self._valid_mask.copy()
        return obs, info
    
    def get_action_mask(self):
         
         """Return a copy of the current valid action mask."""
    
         return self._valid_mask.copy()
    
    def step(self, action: int):
        self._t += 1
        done = (self._t >= self.horizon)

        # Guard invalid index
        if action < 0 or action >= self.action_space.n:
            # treat as invalid action
            reward = self.reward_cfg.invalid_action + self.reward_cfg.step_cost
            obs = self._obs()
            info = {"action_mask": self._valid_mask.copy(), "valid_action": False}
            return obs, reward, done, False, info

        learner_id, item_id = self._decode_action(action)

        # If this action targets a mastered item, soft-penalize & continue
        if self.mastered[learner_id, item_id]:
            reward = self.reward_cfg.invalid_action + self.reward_cfg.step_cost
            obs = self._obs()
            info = {"action_mask": self._valid_mask.copy(), "valid_action": False}
            return obs, reward, done, False, info

        # Simulate correctness probability from Leitner level (toy model)
        level = int(self.leitner[learner_id, item_id])
        # Higher level → higher chance of correct
        p_correct = min(0.2 + 0.15 * level, 0.95)
        correct = self.rng.random() < p_correct

        # Update stats
        self.seen[learner_id, item_id] += 1
        if correct:
            self.leitner[learner_id, item_id] = min(level + 1, 7)
        else:
            self.leitner[learner_id, item_id] = max(level - 1, 1)

        # Track rolling accuracy
        self.recent[learner_id].append(1.0 if correct else 0.0)
        if len(self.recent[learner_id]) > self.curr_cfg.mastery_window:
            self.recent[learner_id] = self.recent[learner_id][-self.curr_cfg.mastery_window:]

        # Mark mastery if rolling acc high and seen enough
        if (self._rolling_acc(learner_id) >= self.curr_cfg.mastery_threshold
            and self.seen[learner_id, item_id] >= 3):
            self.mastered[learner_id, item_id] = True

        # Reward
        reward = (self.reward_cfg.correct if correct else self.reward_cfg.incorrect)
        # Small step cost (stability) + bonus when newly mastered
        if self.mastered[learner_id, item_id] and self.seen[learner_id, item_id] == 3:
            reward += self.reward_cfg.mastery_bonus
        reward += self.reward_cfg.step_cost

        # Refresh valid mask for next step
        self._update_valid_mask()

        obs = self._obs()
        info = {
            "action_mask": self._valid_mask.copy(),
            "valid_action": True,
            "correct": correct,
            "p_correct": p_correct,
            "learner_id": learner_id,
            "item_id": item_id
        }
        return obs, reward, done, False, info

    # Optional convenience if you also have a “suggested” action sampler
    def sample_valid_action(self):
        # Policy could call this in eval mode to avoid invalids
        idxs = np.where(self._valid_mask)[0]
        if idxs.size == 0:
            # pick a curriculum-based suggestion
            l = int(self.rng.integers(0, self.n_learners))
            it = self._curriculum_pick(l)
            return self._encode_action(l, it)
        return int(self.rng.choice(idxs))
