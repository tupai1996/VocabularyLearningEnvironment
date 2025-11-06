# -*- coding: utf-8 -*-
import math
import numpy as np
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces

@dataclass
class RewardCfg:
    correct: float = 1.0
    incorrect: float = -0.5
    invalid_action: float = -0.05
    step_cost: float = -0.001
    mastery_bonus: float = 0.25

@dataclass
class CurriculumCfg:
    use: bool = True
    p_medium: float = 0.55
    p_low: float = 0.25
    p_high: float = 0.20
    mastery_window: int = 30
    mastery_threshold: float = 0.85

class MultiLearnerTutorEnv(gym.Env):
    """
    Single-agent PPO controlling a population of learners.
    Discrete action = (learner_id * n_items) + item_id

    This version adds per-learner memory parameters:
      alpha[l]: forgetting rate (higher => faster forgetting)
      beta[l]:  learning gain per correct review (higher => faster strengthening)

    Memory model per (l, i):
      gap g = t - last_seen[l, i]
      p_correct = exp( - alpha[l] * g / (1 + strength[l, i]) )
      if correct:   strength += beta[l]
      else:         strength *= 0.9
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_learners: int,
        n_items: int,
        horizon: int = 600,
        seed: int = 0,
        reward_cfg: RewardCfg = RewardCfg(),
        curriculum_cfg: CurriculumCfg = CurriculumCfg(),
        # ranges for sampling per-learner alpha/beta
        alpha_range=(0.05, 0.20),
        beta_range=(0.40, 0.80),
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.n_learners = n_learners
        self.n_items = n_items
        self.horizon = horizon
        self.reward_cfg = reward_cfg
        self.curr_cfg = curriculum_cfg
        self.alpha_range = alpha_range
        self.beta_range = beta_range

        # Observation: per-learner summary features (3 each) -> same as your original
        self.obs_dim_per_learner = 3
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_learners * self.obs_dim_per_learner,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_learners * self.n_items)

        # Time
        self._t = 0

        # --- Per-learner parameters (sampled at reset) ---
        self.alpha = np.zeros(self.n_learners, dtype=np.float32)  # forgetting
        self.beta  = np.zeros(self.n_learners, dtype=np.float32)  # learning gain

        # --- Per (learner, item) state ---
        # memory strength (>=0), last_seen (step index), seen count
        self.strength  = np.zeros((n_learners, n_items), dtype=np.float32)
        self.last_seen = np.full((n_learners, n_items), -10_000, dtype=np.int32)
        self.seen      = np.zeros((n_learners, n_items), dtype=np.int32)

        # rolling correctness per learner; mastery per item
        self.recent   = [[] for _ in range(n_learners)]
        self.mastered = np.zeros((n_learners, n_items), dtype=np.bool_)

        # Action mask cache
        self._valid_mask = np.ones(self.action_space.n, dtype=np.bool_)
        self.last_action_mask = None

        # For optional "Leitner-like" curriculum bucketing (by *implied* level)
        # We'll define a pseudo level based on strength: level = 1 + floor(strength)
        # purely for curriculum sampling convenience.
    
    # ---------- helpers ----------
    def _rolling_acc(self, learner_id: int) -> float:
        buf = self.recent[learner_id]
        if not buf:
            return 0.0
        return float(np.mean(buf[-self.curr_cfg.mastery_window:]))

    def _decode_action(self, a: int):
        learner_id = a // self.n_items
        item_id = a % self.n_items
        return learner_id, item_id

    def _encode_action(self, learner_id: int, item_id: int):
        return learner_id * self.n_items + item_id

    def _update_valid_mask(self):
        mask = np.ones(self.action_space.n, dtype=np.bool_)
        for l in range(self.n_learners):
            for it in range(self.n_items):
                if self.mastered[l, it]:
                    mask[self._encode_action(l, it)] = False
        self.last_action_mask = mask.copy()
        self._valid_mask = mask

    def _pseudo_level(self, l: int, it: int) -> int:
        # map strength to an integer "level" 1..7 to reuse your curriculum idea
        return int(np.clip(1 + int(self.strength[l, it]), 1, 7))

    def _curriculum_pick(self, learner_id: int) -> int:
        """Prefer medium 'levels' (derived from strength) if enabled."""
        if not self.curr_cfg.use:
            return int(self.rng.integers(0, self.n_items))
        levels = np.array([self._pseudo_level(learner_id, it) for it in range(self.n_items)])
        low  = np.where(levels <= 2)[0]
        med  = np.where((levels >= 3) & (levels <= 4))[0]
        high = np.where(levels >= 5)[0]
        r = float(self.rng.random())
        if med.size and r < self.curr_cfg.p_medium:
            return int(self.rng.choice(med))
        if low.size and r < self.curr_cfg.p_medium + self.curr_cfg.p_low:
            return int(self.rng.choice(low))
        if high.size:
            return int(self.rng.choice(high))
        return int(self.rng.integers(0, self.n_items))

    def _obs(self):
        # Build per-learner features (normalized to [-1, 1])
        feats = []
        for l in range(self.n_learners):
            # 1) "level" proxy from strength, normalized roughly like your code (1..7 -> [-1,1])
            lv = np.clip(1 + self.strength[l].mean(), 1, 7)
            leitner_norm = (lv - 4.0) / 3.0
            # 2) rolling accuracy -> [-1,1]
            acc_norm = (self._rolling_acc(l) - 0.5) * 2.0
            # 3) recency: average gap normalized by horizon -> [-1,1]
            gaps = np.maximum(0, self._t - self.last_seen[l])
            rec = float(np.clip(gaps.mean(), 0, self.horizon))
            rec_norm = (rec / self.horizon) * 2.0 - 1.0
            feats.extend([leitner_norm, acc_norm, rec_norm])
        return np.asarray(feats, dtype=np.float32)

    # ---------- gym API ----------
    def reset(self, *, seed=None, options=None):
        self._new_mastered_this_ep = 0
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._t = 0

        # sample per-learner alpha/beta
        a0, a1 = self.alpha_range
        b0, b1 = self.beta_range
        self.alpha[:] = self.rng.uniform(a0, a1, size=self.n_learners).astype(np.float32)
        self.beta[:]  = self.rng.uniform(b0, b1, size=self.n_learners).astype(np.float32)

        # reset per (l, i) state
        self.strength[:]  = 0.0
        self.last_seen[:] = -10_000
        self.seen[:]      = 0
        self.mastered[:]  = False
        for l in range(self.n_learners):
            self.recent[l] = []

        self._update_valid_mask()
        obs = self._obs()
        info = {"action_mask": self._valid_mask.copy(),
                "alpha": self.alpha.copy(), "beta": self.beta.copy()}
        return obs, info

    def get_action_mask(self):
        return self._valid_mask.copy()

    def step(self, action: int):
        self._t += 1
        done = (self._t >= self.horizon)

        # invalid index
        if action < 0 or action >= self.action_space.n:
            reward = self.reward_cfg.invalid_action + self.reward_cfg.step_cost
            obs = self._obs()
            info = {"action_mask": self._valid_mask.copy(), "valid_action": False}
            return obs, reward, done, False, info

        l, it = self._decode_action(action)

        # mastered item -> soft penalty, no update
        if self.mastered[l, it]:
            reward = self.reward_cfg.invalid_action + self.reward_cfg.step_cost
            obs = self._obs()
            info = {"action_mask": self._valid_mask.copy(), "valid_action": False}
            return obs, reward, done, False, info

        # -------- memory model with alpha/beta --------
        gap = self._t - self.last_seen[l, it] if self.last_seen[l, it] >= 0 else self.horizon
        # exponential forgetting with strength
        p_correct = float(np.exp(- self.alpha[l] * gap/ (1.0 + self.strength[l, it])))
        p_correct = float(np.clip(p_correct, 0.01, 0.99))
        correct = (self.rng.random() < p_correct)

        # update per (l, it)
        self.seen[l, it] += 1 
        self.last_seen[l, it] = self._t
        if correct:
            self.strength[l, it] += self.beta[l]        # learn more if beta high
        else:
            self.strength[l, it] *= 0.90                 # slight weakening on failure
            self.strength[l, it] = float(max(0.0, self.strength[l, it]))

        # rolling acc buffer
        self.recent[l].append(1.0 if correct else 0.0)
        if len(self.recent[l]) > self.curr_cfg.mastery_window:
            self.recent[l] = self.recent[l][-self.curr_cfg.mastery_window:]

        # mastery when rolling acc high + seen at least 3 times
        if (self._rolling_acc(l) >= self.curr_cfg.mastery_threshold
            and self.seen[l, it] >= 3):
            self.mastered[l, it] = True

        # reward
        reward = (self.reward_cfg.correct if correct else self.reward_cfg.incorrect)
        if self.mastered[l, it] and self.seen[l, it] == 3:
            reward += self.reward_cfg.mastery_bonus
        reward += self.reward_cfg.step_cost

        # refresh mask
        self._update_valid_mask()

        obs = self._obs()
        info = {
            "action_mask": self._valid_mask.copy(),
            "valid_action": True,
            "correct": correct,
            "p_correct": p_correct,
            "learner_id": l,
            "item_id": it,
            "alpha": self.alpha[l],
            "beta": self.beta[l],
            "strength": float(self.strength[l, it]),
            "gap": int(gap),
        }
        return obs, reward, done, False, info

    # Optional helper
    def sample_valid_action(self):
        idxs = np.where(self._valid_mask)[0]
        if idxs.size == 0:
            l = int(self.rng.integers(0, self.n_learners))
            it = self._curriculum_pick(l)
            return self._encode_action(l, it)
        return int(self.rng.choice(idxs))
