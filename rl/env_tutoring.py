# rl/env_tutoring.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Optional

from ..learners.exp_memory import ExpMemoryLearner, MemoryState
from ..teacher.items import WordItem

class TutoringEnv(gym.Env):
    """
    PPO env: action = pick 1 item from K candidates.
    Uses your ExpMemoryLearner (reply/learn) and WordItem.
    Observation = K*4 candidate feats + 4 global feats.
    Reward = step shaping + terminal mastery bonus.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_items: int = 500,
        horizon: int = 600,
        candidate_k: int = 32,
        recall_threshold: float = 0.9,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_items = n_items
        self.horizon = horizon
        self.K = candidate_k
        self.rho = recall_threshold
        self.seed_val = seed

        # synthetic items compatible with your API
        self.items: List[WordItem] = [WordItem(f"w{i}", f"t{i}") for i in range(n_items)]
        self.questions = [it.get_question() for it in self.items]

        # your learner
        self.learner = ExpMemoryLearner(alpha=.004, beta=.2)

        # bookkeeping
        self.t = 0
        self.step_idx = 0
        self.seen = np.zeros(n_items, dtype=bool)
        self.mastered = np.zeros(n_items, dtype=bool)
        self.current_candidates: List[int] = []

        # obs/action spaces
        cand_feat_dim = 4
        self.per_cand_dim = cand_feat_dim
        self.global_dim = 4
        obs_dim = self.K * cand_feat_dim + self.global_dim
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.K)

        # rewards
        self.r_correct = 0.2
        self.r_wrong = -0.05
        self.r_prob_gain = 0.1
        self.r_cross_threshold = 1.0
        self.terminal_mastery_bonus = 1.0

        self.rng = np.random.default_rng(seed)

    # ---------- helper accessors into your learner ----------
    def _state_for(self, idx: int) -> Optional[MemoryState]:
        q = self.questions[idx]
        return self.learner.memory.get(q)

    def _p_recall(self, idx: int) -> float:
        st = self._state_for(idx)
        if st is None:
            return 0.0
        p = float(st.get_probability(self.t))
        return float(np.clip(p, 0.0, 1.0))

    def _introduced_mask(self) -> np.ndarray:
        return np.array([q in self.learner.memory for q in self.questions], dtype=bool)

    def _time_since_last(self, idx: int) -> int:
        st = self._state_for(idx)
        return 0 if st is None else max(0, int(self.t - st.last_occurrence))

    def _reps(self, idx: int) -> int:
        st = self._state_for(idx)
        return 0 if st is None else int(st.n_occurrences)

    # ---------- candidate set ----------
    def _build_candidates(self) -> List[int]:
        probs = np.array([self._p_recall(i) for i in range(self.n_items)])
        introduced_ids = np.where(self._introduced_mask())[0]
        unseen_ids = np.where(~self._introduced_mask())[0]

        cand: List[int] = []
        if introduced_ids.size > 0:
            introduced_sorted = introduced_ids[np.argsort(probs[introduced_ids])]  # low->high (at risk first)
            cand.extend(introduced_sorted.tolist())

        unseen_list = unseen_ids.tolist()
        self.rng.shuffle(unseen_list)
        cand.extend(unseen_list)

        if len(cand) >= self.K:
            cand = cand[:self.K]
        else:
            pool = list(set(range(self.n_items)) - set(cand))
            self.rng.shuffle(pool)
            cand.extend(pool[: max(0, self.K - len(cand))])

        return cand

    def _obs(self) -> np.ndarray:
        self.current_candidates = self._build_candidates()
        feats = []
        for idx in self.current_candidates:
            p = self._p_recall(idx)
            dt = self._time_since_last(idx)
            reps = self._reps(idx)
            introduced = 1.0 if self._state_for(idx) is not None else 0.0
            feats.extend([
                p,
                float(np.clip(dt / max(1, self.horizon), 0.0, 1.0)),
                float(np.clip(reps / 10.0, 0.0, 1.0)),
                introduced
            ])

        frac_seen = float(self.seen.mean())
        frac_mastered = float(self.mastered.mean())
        steps_left_norm = float((self.horizon - self.step_idx) / self.horizon)
        unseen_count_norm = float((~self.seen).sum() / max(1, self.n_items))
        feats.extend([frac_seen, frac_mastered, steps_left_norm, unseen_count_norm])

        return np.asarray(feats, dtype=np.float32)

    def _terminal_bonus(self) -> float:
        probs = np.array([self._p_recall(i) for i in range(self.n_items)])
        n_mastered = int((probs >= self.rho).sum())
        return self.terminal_mastery_bonus * (n_mastered / self.n_items)

    # ---------- Gym API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.learner = ExpMemoryLearner(alpha=.004, beta=.2)
        self.seen[:] = False
        self.mastered[:] = False
        self.t = 0
        self.step_idx = 0
        return self._obs(), {}

    def step(self, action: int):
        assert 0 <= action < self.K, "invalid action"
        item_id = self.current_candidates[action]
        item = self.items[item_id]

        # before
        p_before = self._p_recall(item_id)

        # interact with your learner
        answer = self.learner.reply(item.get_question(), self.t)
        correct = (answer == item.get_answer())

        # learning update
        self.learner.learn(item, self.t)
        self.seen[item_id] = True

        # after
        p_after = self._p_recall(item_id)

        # reward shaping
        reward = (self.r_correct if correct else self.r_wrong)
        reward += self.r_prob_gain * max(0.0, p_after - p_before)
        if (p_after >= self.rho) and (not self.mastered[item_id]):
            self.mastered[item_id] = True
            reward += self.r_cross_threshold

        # advance time/episode
        self.step_idx += 1
        self.t += 1
        terminated = (self.step_idx >= self.horizon)
        if terminated:
            reward += self._terminal_bonus()

        info = {"item_id": item_id, "correct": bool(correct), "p_before": float(p_before), "p_after": float(p_after)}
        return self._obs(), float(reward), bool(terminated), False, info

    # For sb3-contrib MaskablePPO; all K actions valid here
    def action_masks(self) -> np.ndarray:
        return np.ones(self.K, dtype=bool)
