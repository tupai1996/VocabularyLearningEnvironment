# rl/masked_multihead_policy.py
from typing import Tuple
import torch as th
from torch import nn
from stable_baselines3.ppo.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class IdentityExtractor(BaseFeaturesExtractor):
    """Pass-through: obs['dense'] is already flat."""
    def __init__(self, observation_space, features_dim: int):
        super().__init__(observation_space, features_dim)
        self._features_dim = features_dim
    def forward(self, obs):
        return obs["dense"]

class MaskedMultiHeadPolicy(ActorCriticPolicy):
    """
    PPO policy for MultiDiscrete with (L, Kc) action mask.
    We rely on the base class to build action_net/value_net and the distribution.
    We only inject the mask into the logits before sampling.
    """
    def __init__(self, *args, L: int, Kc: int, **kwargs):
        self.L = L
        self.Kc = Kc
        super().__init__(*args, **kwargs)  # DO NOT call self._build() here

    def forward(self, obs, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # Feature extraction + torso from base class
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Base class heads
        logits = self.action_net(latent_pi)                # shape: (B, sum(nvec)) == (B, L*Kc)
        values = self.value_net(latent_vf).squeeze(-1)

        # Apply action mask
        mask = obs["action_mask"].reshape(logits.shape[0], self.L * self.Kc)
        big_neg = th.finfo(logits.dtype).min / 4
        masked_logits = logits + (1.0 - mask) * big_neg

        # Build distribution from masked logits and sample
        dist = self._get_action_dist_from_latent(masked_logits)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        return actions, values, log_prob

    def _predict(self, obs, deterministic: bool = False) -> th.Tensor:
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.policy_net(features)
        logits = self.action_net(latent_pi)
        mask = obs["action_mask"].reshape(logits.shape[0], self.L * self.Kc)
        big_neg = th.finfo(logits.dtype).min / 4
        masked_logits = logits + (1.0 - mask) * big_neg
        dist = self._get_action_dist_from_latent(masked_logits)
        return dist.get_actions(deterministic=deterministic)
