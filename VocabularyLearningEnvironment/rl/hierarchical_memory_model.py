# -*- coding: utf-8 -*-
"""
HierarchicalMemoryModel
- Creates a hierarchical, fixed-parameter memory model for a population of learners.
- Beta is fixed (0.5) as requested.
- Alpha is fixed per-group from a small grid; per-learner alpha = group_alpha * (1 + small_noise).
- Provides helpers: get_per_learner_params(), simulate_responses(), log_likelihood().
- Intentionally does NOT implement inference (placeholders for later).
"""

from dataclasses import dataclass
import numpy as np
import math
from typing import Optional, Tuple, List, Dict

@dataclass
class HierarchicalMemoryConfig:
    n_learners: int
    n_items: int
    n_groups: int = 2
    beta_fixed: float = 0.5
    alpha_grid_min: float = 2.0e-6
    alpha_grid_max: float = 8.0e-5
    per_learner_noise_scale: float = 0.08   # multiplicative noise applied to group alpha
    seed: int = 0


class HierarchicalMemoryModel:
    """
    Hierarchical memory model with fixed parameters for now.
    - group_alphas: vector of length n_groups (set from a grid between alpha_grid_min and alpha_grid_max)
    - group_beta: fixed (beta_fixed) or optional group values (but set to fixed now)
    - per-learner alpha & beta derived from groups + small multiplicative noise
    """

    def __init__(self, cfg: HierarchicalMemoryConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # validate groups
        self.n_groups = max(1, min(cfg.n_groups, cfg.n_learners))
        self.group_of_learner = np.array([i % self.n_groups for i in range(cfg.n_learners)], dtype=int)

        # create group-level alpha grid
        self.group_alphas = self._build_alpha_grid(cfg.alpha_grid_min, cfg.alpha_grid_max, self.n_groups)
        # betas: fix to cfg.beta_fixed for all groups
        self.group_betas = np.full(self.n_groups, cfg.beta_fixed, dtype=float)

        # per-learner params (filled by init_per_learner_params)
        self.alpha_per_learner = np.zeros(cfg.n_learners, dtype=float)
        self.beta_per_learner = np.full(cfg.n_learners, cfg.beta_fixed, dtype=float)

        # create initial per-learner params
        self.init_per_learner_params()

    # ---------- parameter setup ----------
    def _build_alpha_grid(self, amin: float, amax: float, ng: int) -> np.ndarray:
        if ng == 1:
            return np.array([ (amin + amax) / 2.0 ], dtype=float)
        return np.linspace(amin, amax, ng, dtype=float)

    def init_per_learner_params(self, resample_noise: bool = True) -> None:
        """
        Initialize per-learner alpha and beta from group hyperparams.
        If resample_noise is True, we apply multiplicative gaussian noise (clamped).
        """
        for l in range(self.cfg.n_learners):
            g = int(self.group_of_learner[l])
            ga = float(self.group_alphas[g])
            gb = float(self.group_betas[g])

            if resample_noise:
                m = 1.0 + self.rng.normal(loc=0.0, scale=self.cfg.per_learner_noise_scale)
                m = float(np.clip(m, 0.7, 1.3))
            else:
                m = 1.0

            self.alpha_per_learner[l] = ga * m
            self.beta_per_learner[l] = float(np.clip(gb * m, 0.0, 0.9999))

    def set_group_alphas(self, alphas: List[float]):
        """Manually set group alphas (length must equal n_groups). Does not resample per-learner noise."""
        arr = np.asarray(alphas, dtype=float)
        if arr.shape[0] != self.n_groups:
            raise ValueError("alphas length != n_groups")
        self.group_alphas = arr.copy()
        # update learners deterministically (keep existing noise multipliers by resampling)
        self.init_per_learner_params(resample_noise=True)

    # ---------- accessors ----------
    def get_per_learner_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (alpha_per_learner, beta_per_learner) arrays."""
        return self.alpha_per_learner.copy(), self.beta_per_learner.copy()

    def get_group_of_learner(self) -> np.ndarray:
        return self.group_of_learner.copy()

    # ---------- simulation helpers ----------
    def recall_probability(self, learner_id: int, item_last_seen_gap: int, n_reviews: int) -> float:
        """
        Exponential forgetting model as used in your env:
          eff_alpha = alpha * ((1 - beta) ** (n - 1))
          p_correct = exp(- eff_alpha * delta_t)
        Note: delta_t is gap in steps since last seen.
        """
        a = float(self.alpha_per_learner[learner_id])
        b = float(self.beta_per_learner[learner_id])
        n = max(1, int(n_reviews))
        eff_alpha = a * ((1.0 - b) ** (n - 1))
        p = math.exp(- eff_alpha * max(1, item_last_seen_gap))
        # clamp
        return float(np.clip(p, 1e-6, 1.0 - 1e-6))

    def simulate_responses(self,
                           learner_id: int,
                           item_last_seen_gaps: np.ndarray,
                           item_n_reviews: np.ndarray) -> np.ndarray:
        """
        Vectorized simulation of Bernoulli responses for a single learner:
        - item_last_seen_gaps: array shape (n_items,) of delta_t for each item
        - item_n_reviews: array shape (n_items,) of n_reviews
        Returns binary array of same shape with simulated correct/incorrect.
        """
        gaps = np.asarray(item_last_seen_gaps, dtype=float)
        nrev = np.asarray(item_n_reviews, dtype=float)
        a = float(self.alpha_per_learner[learner_id])
        b = float(self.beta_per_learner[learner_id])
        eff_alpha = a * np.power((1.0 - b), np.maximum(0, nrev - 1.0))
        p = np.exp(- eff_alpha * np.maximum(1.0, gaps))
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        draws = self.rng.random(size=p.shape) < p
        return draws.astype(np.int32)

    def log_likelihood_learner(self,
                               learner_id: int,
                               item_last_seen_gaps: np.ndarray,
                               item_n_reviews: np.ndarray,
                               item_observed_correct: np.ndarray) -> float:
        """
        Compute log-likelihood of observed Bernoulli responses under current per-learner params.
        Returns scalar log-likelihood (sum over items).
        """
        gaps = np.asarray(item_last_seen_gaps, dtype=float)
        nrev = np.asarray(item_n_reviews, dtype=float)
        obs = np.asarray(item_observed_correct, dtype=float)
        a = float(self.alpha_per_learner[learner_id])
        b = float(self.beta_per_learner[learner_id])
        eff_alpha = a * np.power((1.0 - b), np.maximum(0, nrev - 1.0))
        p = np.exp(- eff_alpha * np.maximum(1.0, gaps))
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        ll = np.sum(obs * np.log(p) + (1.0 - obs) * np.log(1.0 - p))
        return float(ll)

    # ---------- placeholder: inference stubs ----------
    def fit_em(self, *args, **kwargs):
        """
        Placeholder: Expect to implement hierarchical EM later.
        For now, this raises NotImplementedError to signal 'inference later'.
        """
        raise NotImplementedError("EM fitting is not implemented in this stage. To be added later.")

    def fit_variational(self, *args, **kwargs):
        """Placeholder for variational inference fitting routine."""
        raise NotImplementedError("Variational inference is not implemented in this stage.")

    # ---------- utilities ----------
    def summary(self) -> Dict[str, object]:
        return {
            "n_learners": int(self.cfg.n_learners),
            "n_items": int(self.cfg.n_items),
            "n_groups": int(self.n_groups),
            "group_alphas": self.group_alphas.tolist(),
            "group_betas": self.group_betas.tolist(),
            "alpha_per_learner": self.alpha_per_learner.tolist(),
            "beta_per_learner": self.beta_per_learner.tolist(),
        }


# -----------------------------
# Quick demo / smoke test (callable)
# -----------------------------
def demo_print(cfg: Optional[HierarchicalMemoryConfig] = None, show_first=8):
    if cfg is None:
        cfg = HierarchicalMemoryConfig(n_learners=8, n_items=200, n_groups=2, seed=1)
    hm = HierarchicalMemoryModel(cfg)
    print("Group alphas:", hm.group_alphas)
    print("Group betas:", hm.group_betas)
    a, b = hm.get_per_learner_params()
    for i in range(min(show_first, cfg.n_learners)):
        print(f"L{i}: alpha={a[i]:.3e}, beta={b[i]:.3f}, group={hm.group_of_learner[i]}")
    # simulate a single learner quick test
    gaps = np.full(cfg.n_items, 10, dtype=int)
    nrev = np.zeros(cfg.n_items, dtype=int)
    draws = hm.simulate_responses(0, gaps, nrev)
    print("simulated correct fraction (learner 0):", float(np.mean(draws)))
    return hm

if __name__ == "__main__":
    demo_print()
