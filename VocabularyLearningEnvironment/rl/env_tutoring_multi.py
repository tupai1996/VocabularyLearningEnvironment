# -*- coding: utf-8 -*-
import math
import numpy as np
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces

# ---------- configs ----------
@dataclass
class RewardCfg:
    correct: float = 1.0
    incorrect: float = -1          # softer penalty (helps early PPO)
    invalid_action: float = -0.10
    step_cost: float = -0.001
    mastery_bonus: float = 2.0        # stronger signal when mastering
    shaping_lambda: float = 0.25      # gentle difficulty-aware shaping

@dataclass
class CurriculumCfg:
    use: bool = True
    p_medium: float = 0.80            # nudge toward boundary items
    p_low: float = 0.20
    p_high: float = 0.10
    mastery_window: int = 30
    mastery_threshold: float = 0.85

@dataclass
class MemoryCfg:
    alpha: float
    beta: float

# ---------- environment ----------
class MultiLearnerTutorEnv(gym.Env):
    """
    Multi-learner tutoring environment with:
      - Fixed alpha/beta per learner (exponential forgetting).
      - Tuned reward scheme + optional difficulty-aware shaping.
      - Curriculum sampling & valid-action masking.
      - Per-learner and global metrics in info for TensorBoard.
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
        self.rng = np.random.default_rng(seed)
        self.n_learners = n_learners
        self.n_items = n_items
        self.horizon = horizon
        self.reward_cfg = reward_cfg
        self.curr_cfg = curriculum_cfg

        # obs/action space: [avg_leitner, rolling_acc, avg_time_since_seen_norm] per learner
        self.obs_dim_per_learner = 3
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_learners * self.obs_dim_per_learner,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_learners * self.n_items)

        # state
        self._t = 0
        self.leitner = np.ones((n_learners, n_items), dtype=np.int32)
        self.seen = np.zeros((n_learners, n_items), dtype=np.int32)
        self.recent = [[] for _ in range(n_learners)]
        self.mastered = np.zeros((n_learners, n_items), dtype=np.bool_)

        # fixed memory parameters (α,β) per learner
        self.memory_params = []
        for l in range(n_learners):
            alpha = 4.0e-6 + l * 5e-7  # mild variety across learners
            beta = 0.5
            self.memory_params.append(MemoryCfg(alpha, beta))

        # timing & repetition tracking
        self.last_seen = np.zeros((n_learners, n_items), dtype=np.int32)  # 0 means never
        self.n_reviews = np.zeros((n_learners, n_items), dtype=np.int32)

        # action mask
        self._valid_mask = np.ones(self.action_space.n, dtype=np.bool_)
        self.last_action_mask = self._valid_mask.copy()

    # ---------- helpers ----------
    def _rolling_acc(self, learner_id: int) -> float:
        buf = self.recent[learner_id]
        if not buf:
            return 0.0
        return float(np.mean(buf[-self.curr_cfg.mastery_window:]))

    def _curriculum_pick(self, learner_id: int) -> int:
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
        return a // self.n_items, a % self.n_items

    def _encode_action(self, learner_id: int, item_id: int):
        return learner_id * self.n_items + item_id

    def _update_valid_mask(self):
        mask = np.ones(self.action_space.n, dtype=np.bool_)
        for l in range(self.n_learners):
            for it in range(self.n_items):
                if self.mastered[l, it]:
                    mask[self._encode_action(l, it)] = False
        self._valid_mask = mask

    def _obs(self):
        feats = []
        # avg "time since seen" per learner (normalize by horizon)
        for l in range(self.n_learners):
            lv = np.clip(self.leitner[l].mean(), 1, 7)
            leitner_norm = (lv - 4) / 3.0

            acc_norm = (self._rolling_acc(l) - 0.5) * 2.0

            # compute time since seen for each item (0 if never seen -> treat as horizon)
            tss = self._t - self.last_seen[l]               # vector
            tss = np.where(self.last_seen[l] == 0, self.horizon, tss)
            avg_tss = float(np.mean(np.clip(tss, 0, self.horizon)))
            rec_norm = (avg_tss / self.horizon) * 2.0 - 1.0

            feats.extend([leitner_norm, acc_norm, rec_norm])
        return np.asarray(feats, dtype=np.float32)

    # ---------- gym API ----------
    def reset(self, *, seed=None, options=None):
        self._new_mastered_this_ep = 0
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._t = 0
        self.leitner[:] = 1
        self.seen[:] = 0
        self.mastered[:] = False
        self.last_seen[:] = 0
        self.n_reviews[:] = 0
        for l in range(self.n_learners):
            self.recent[l] = []
        self._update_valid_mask()
        obs = self._obs()
        info = {"action_mask": self._valid_mask.copy()}
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

        learner_id, item_id = self._decode_action(action)

        # if mastered, penalize
        if self.mastered[learner_id, item_id]:
            reward = self.reward_cfg.invalid_action + self.reward_cfg.step_cost
            obs = self._obs()
            info = {"action_mask": self._valid_mask.copy(), "valid_action": False}
            return obs, reward, done, False, info

        # exponential forgetting model
        last = self.last_seen[learner_id, item_id]
        delta_t = (self._t - last) if last > 0 else max(1, self._t)  # if never seen, use current t
        mp = self.memory_params[learner_id]
        n = self.n_reviews[learner_id, item_id] + 1
        eff_alpha = mp.alpha * ((1.0 - mp.beta) ** (n - 1))
        p_correct = math.exp(-eff_alpha * max(1, delta_t))
        correct = self.rng.random() < p_correct

        # update counts
        self.last_seen[learner_id, item_id] = self._t
        self.n_reviews[learner_id, item_id] = n
        self.seen[learner_id, item_id] += 1

        # update Leitner level
        level = int(self.leitner[learner_id, item_id])
        if correct:
            self.leitner[learner_id, item_id] = min(level + 1, 7)
        else:
            self.leitner[learner_id, item_id] = max(level - 1, 1)

        # rolling acc
        self.recent[learner_id].append(1.0 if correct else 0.0)
        if len(self.recent[learner_id]) > self.curr_cfg.mastery_window:
            self.recent[learner_id] = self.recent[learner_id][-self.curr_cfg.mastery_window:]

        # -------- mark mastery --------
        was_mastered = self.mastered[learner_id, item_id]
        if (self._rolling_acc(learner_id) >= self.curr_cfg.mastery_threshold
            and self.seen[learner_id, item_id] >= 3):
            self.mastered[learner_id, item_id] = True

        # ---------- rewards ----------
        reward = (self.reward_cfg.correct if correct else self.reward_cfg.incorrect)

        # difficulty-aware shaping (encourage boundary)
        lam = self.reward_cfg.shaping_lambda
        if lam > 0.0:
            reward += (lam * (1.0 - p_correct)) if correct else (-lam * p_correct)

        # mastery bonus only when transitioning from False → True
        if (not was_mastered) and self.mastered[learner_id, item_id]:
            reward += self.reward_cfg.mastery_bonus
            reward += 0.01   # tiny diversity nudge only when *new mastery happens*
            self._new_mastered_this_ep += 1

        # step cost
        reward += self.reward_cfg.step_cost

        # refresh mask
        self._update_valid_mask()

        # -------- info for TensorBoard (per-learner + global) --------
        learner_roll_acc = self._rolling_acc(learner_id)
        mastered_count_l = int(np.sum(self.mastered[learner_id]))
        total_mastered = int(np.sum(self.mastered))

        obs = self._obs()
        info = {
            "action_mask": self._valid_mask.copy(),
            "valid_action": True,
            "correct": correct,
            "p_correct": p_correct,
            "learner_id": learner_id,
            "item_id": item_id,

            # Per-learner series
            f"per_learner/{learner_id}/reward": float(reward),
            f"per_learner/{learner_id}/correct": float(correct),
            f"per_learner/{learner_id}/rolling_acc": float(learner_roll_acc),
            f"per_learner/{learner_id}/p_correct": float(p_correct),
            f"per_learner/{learner_id}/mastered_count": mastered_count_l,

            # Global counters
            "global/total_mastered": total_mastered,
        }
        return obs, reward, done, False, info

    def sample_valid_action(self):
        idxs = np.where(self._valid_mask)[0]
        if idxs.size == 0:
            l = int(self.rng.integers(0, self.n_learners))
            it = self._curriculum_pick(l)
            return self._encode_action(l, it)
        return int(self.rng.choice(idxs))
