# -*- coding: utf-8 -*-
"""
hier_env.py
Standalone hierarchical multi-learner tutoring env.
Drop into VocabularyLearningEnvironment/rl/ and run with the separate trainer.
This file is self-contained and does NOT modify your original env.
"""

import math
import numpy as np
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces

# ----------------- Configs -----------------
@dataclass
class RewardCfg:
    correct: float = 1.0
    incorrect: float = -1.0
    invalid_action: float = -0.10
    step_cost: float = -0.001
    mastery_bonus: float = 2.0
    shaping_lambda: float = 0.25

@dataclass
class CurriculumCfg:
    use: bool = True
    p_medium: float = 0.55
    p_low: float = 0.25
    p_high: float = 0.20
    mastery_window: int = 30
    mastery_threshold: float = 0.85

@dataclass
class HierCfg:
    n_groups: int = 2
    beta_fixed: float = 0.5
    alpha_grid_min: float = 2.0e-6
    alpha_grid_max: float = 6.0e-6
    per_learner_noise_scale: float = 0.12

# ----------------- small hierarchical memory model (fixed params) -----------------
class SimpleHierModel:
    def __init__(self, n_learners, n_items, cfg: HierCfg, seed=0):
        self.rng = np.random.default_rng(seed)
        self.n_learners = int(n_learners)
        self.n_items = int(n_items)
        self.n_groups = max(1, min(cfg.n_groups, self.n_learners))
        self.group_of_learner = np.array([i % self.n_groups for i in range(self.n_learners)], dtype=int)
        # grid alphas
        if self.n_groups == 1:
            self.group_alphas = np.array([(cfg.alpha_grid_min + cfg.alpha_grid_max) / 2.0], dtype=float)
        else:
            self.group_alphas = np.linspace(cfg.alpha_grid_min, cfg.alpha_grid_max, self.n_groups, dtype=float)
        self.group_betas = np.full(self.n_groups, cfg.beta_fixed, dtype=float)
        self.per_noise = float(cfg.per_learner_noise_scale)

        # per-learner arrays
        self.alpha = np.zeros(self.n_learners, dtype=float)
        self.beta = np.full(self.n_learners, cfg.beta_fixed, dtype=float)
        self.init_per_learner(resample_noise=True)

    def init_per_learner(self, resample_noise=True):
        for l in range(self.n_learners):
            g = int(self.group_of_learner[l])
            ga = float(self.group_alphas[g])
            gb = float(self.group_betas[g])
            if resample_noise:
                m = 1.0 + self.rng.normal(0.0, self.per_noise)
                m = float(np.clip(m, 0.7, 1.3))
            else:
                m = 1.0
            self.alpha[l] = ga * m
            self.beta[l] = float(np.clip(gb * m, 0.0, 0.9999))

    def get_params(self):
        return self.alpha.copy(), self.beta.copy()

    def summary(self):
        return {
            "n_learners": self.n_learners,
            "n_groups": self.n_groups,
            "group_alphas": self.group_alphas.tolist(),
            "alpha_per_learner": self.alpha.tolist(),
            "beta_per_learner": self.beta.tolist()
        }

# ----------------- Hierarchical environment -----------------
class HierarchicalMultiLearnerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 n_learners: int,
                 n_items: int,
                 horizon: int = 600,
                 seed: int = 0,
                 reward_cfg: RewardCfg = RewardCfg(),
                 curriculum_cfg: CurriculumCfg = CurriculumCfg(),
                 hier_cfg: HierCfg = HierCfg()):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.n_learners = int(n_learners)
        self.n_items = int(n_items)
        self.horizon = int(horizon)
        self.reward_cfg = reward_cfg
        self.curr_cfg = curriculum_cfg
        # hierarchical model
        self.hier = SimpleHierModel(self.n_learners, self.n_items, hier_cfg, seed=seed)
        self.alpha, self.beta = self.hier.get_params()

        # observation / action
        self.obs_dim_per_learner = 3
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(self.n_learners * self.obs_dim_per_learner,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_learners * self.n_items)

        # state
        self._t = 0
        self.leitner = np.ones((self.n_learners, self.n_items), dtype=np.int32)
        self.seen = np.zeros((self.n_learners, self.n_items), dtype=np.int32)
        self.last_seen = np.full((self.n_learners, self.n_items), -1, dtype=np.int32)
        self.n_reviews = np.zeros((self.n_learners, self.n_items), dtype=np.int32)
        self.recent = [[] for _ in range(self.n_learners)]
        self.mastered = np.zeros((self.n_learners, self.n_items), dtype=bool)

        # mask
        self._valid_mask = np.ones(self.action_space.n, dtype=bool)
        self.last_action_mask = self._valid_mask.copy()

    # ---------- helpers ----------
    def _rolling_acc(self, l):
        buf = self.recent[l]
        if not buf:
            return 0.0
        return float(np.mean(buf[-self.curr_cfg.mastery_window:]))

    def _curriculum_pick(self, l):
        if not self.curr_cfg.use:
            return int(self.rng.integers(0, self.n_items))
        levels = self.leitner[l]
        low = np.where(levels <= 2)[0]
        med = np.where((levels >= 3) & (levels <= 4))[0]
        high = np.where(levels >= 5)[0]
        r = self.rng.random()
        if med.size and r < self.curr_cfg.p_medium:
            return int(self.rng.choice(med))
        if low.size and r < (self.curr_cfg.p_medium + self.curr_cfg.p_low):
            return int(self.rng.choice(low))
        if high.size:
            return int(self.rng.choice(high))
        return int(self.rng.integers(0, self.n_items))

    def _encode_action(self, l, it):
        return l * self.n_items + it

    def _decode_action(self, a):
        return a // self.n_items, a % self.n_items

    def _update_valid_mask(self):
        mask = np.ones(self.action_space.n, dtype=bool)
        for l in range(self.n_learners):
            for it in range(self.n_items):
                if self.mastered[l, it]:
                    mask[self._encode_action(l, it)] = False
        self._valid_mask = mask
        self.last_action_mask = mask.copy()

    def _obs(self):
        feats = []
        for l in range(self.n_learners):
            lv = np.clip(self.leitner[l].mean(), 1, 7)
            leitner_norm = (lv - 4) / 3.0
            acc_norm = (self._rolling_acc(l) - 0.5) * 2.0
            rec = np.clip(self.seen[l].mean(), 0, self.horizon)
            rec_norm = (rec / self.horizon) * 2.0 - 1.0
            feats.extend([leitner_norm, acc_norm, rec_norm])
        return np.asarray(feats, dtype=np.float32)

    # ---------- gym API ----------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            # resample hier params deterministically
            self.hier = SimpleHierModel(self.n_learners, self.n_items, HierCfg(), seed=seed)
            self.alpha, self.beta = self.hier.get_params()
        self._t = 0
        self.leitner[:] = 1
        self.seen[:] = 0
        self.last_seen[:] = -1
        self.n_reviews[:] = 0
        self.mastered[:] = False
        for l in range(self.n_learners):
            self.recent[l] = []
        self._update_valid_mask()
        obs = self._obs()
        info = {"action_mask": self._valid_mask.copy(), "hier_summary": self.hier.summary()}
        return obs, info

    def get_action_mask(self):
        return self._valid_mask.copy()

    def step(self, action: int):
        self._t += 1
        done = (self._t >= self.horizon)

        # invalid action guard
        if action < 0 or action >= self.action_space.n:
            reward = self.reward_cfg.invalid_action + self.reward_cfg.step_cost
            obs = self._obs()
            return obs, reward, done, False, {"action_mask": self._valid_mask.copy(), "valid_action": False}

        l, it = self._decode_action(action)

        if self.mastered[l, it]:
            reward = self.reward_cfg.invalid_action + self.reward_cfg.step_cost
            obs = self._obs()
            return obs, reward, done, False, {"action_mask": self._valid_mask.copy(), "valid_action": False}

        # memory model using hierarchical alpha/beta (fixed)
        last = self.last_seen[l, it]
        delta_t = (self._t - last) if last >= 0 else max(1, self._t)
        a = float(self.alpha[l])
        b = float(self.beta[l])
        n = self.n_reviews[l, it] + 1
        eff_alpha = a * ((1.0 - b) ** (n - 1))
        p_correct = math.exp(-eff_alpha * max(1, delta_t))
        correct = self.rng.random() < p_correct

        # update tracking
        self.last_seen[l, it] = self._t
        self.n_reviews[l, it] = n
        self.seen[l, it] += 1

        # update Leitner
        level = int(self.leitner[l, it])
        if correct:
            self.leitner[l, it] = min(level + 1, 7)
        else:
            self.leitner[l, it] = max(level - 1, 1)

        # rolling acc buffer
        self.recent[l].append(1.0 if correct else 0.0)
        if len(self.recent[l]) > self.curr_cfg.mastery_window:
            self.recent[l] = self.recent[l][-self.curr_cfg.mastery_window:]

        # mastery transition
        was_mastered = self.mastered[l, it]
        if (self._rolling_acc(l) >= self.curr_cfg.mastery_threshold and self.seen[l, it] >= 3):
            self.mastered[l, it] = True

        # reward shaping
        reward = (self.reward_cfg.correct if correct else self.reward_cfg.incorrect)
        lam = self.reward_cfg.shaping_lambda
        if lam > 0:
            reward += (lam * (1.0 - p_correct)) if correct else (-lam * p_correct)

        # transition bonus
        if (not was_mastered) and self.mastered[l, it]:
            reward += self.reward_cfg.mastery_bonus
            reward += 0.01

        reward += self.reward_cfg.step_cost

        # update mask and info
        self._update_valid_mask()
        learner_roll_acc = self._rolling_acc(l)
        mastered_count_l = int(np.sum(self.mastered[l]))
        total_mastered = int(np.sum(self.mastered))

        obs = self._obs()
        info = {
            "action_mask": self._valid_mask.copy(),
            "valid_action": True,
            "correct": correct,
            "p_correct": p_correct,
            "learner_id": l,
            "item_id": it,
            f"per_learner/{l}/reward": float(reward),
            f"per_learner/{l}/correct": float(correct),
            f"per_learner/{l}/rolling_acc": float(learner_roll_acc),
            f"per_learner/{l}/p_correct": float(p_correct),
            f"per_learner/{l}/mastered_count": mastered_count_l,
            "global/total_mastered": total_mastered
        }
        # per-group stats
        for g in range(self.hier.n_groups):
            members = np.where(self.hier.group_of_learner == g)[0]
            if members.size:
                info[f"group/{g}/mastered_count"] = float(np.sum(self.mastered[members]))
                info[f"group/{g}/avg_rolling_acc"] = float(np.mean([self._rolling_acc(m) for m in members]))
            else:
                info[f"group/{g}/mastered_count"] = 0.0
                info[f"group/{g}/avg_rolling_acc"] = 0.0

        return obs, reward, done, False, info

    def sample_valid_action(self):
        idxs = np.where(self._valid_mask)[0]
        if idxs.size == 0:
            l = int(self.rng.integers(0, self.n_learners))
            it = self._curriculum_pick(l)
            return self._encode_action(l, it)
        return int(self.rng.choice(idxs))
