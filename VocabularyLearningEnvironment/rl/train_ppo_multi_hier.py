# -*- coding: utf-8 -*-
"""
train_ppo_multi_hier.py
A copy of your PPO training loop adapted to the hierarchical env (hier_env.HierarchicalMultiLearnerEnv).
Place in VocabularyLearningEnvironment/rl/ and run as a module (python -m VocabularyLearningEnvironment.rl.train_ppo_multi_hier)
"""

import os
import argparse
from functools import partial

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed

from VocabularyLearningEnvironment.rl.hier_env import HierarchicalMultiLearnerEnv, RewardCfg, CurriculumCfg, HierCfg

# Optional tqdm
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self._last = 0
        self._tqdm = None

    def _on_training_start(self):
        if _HAS_TQDM:
            self._tqdm = tqdm(total=self.total_timesteps, desc="Training PPO (hier)", unit="steps")
        else:
            print("Training PPO (hier): tqdm not installed.")
        return None

    def _on_step(self) -> bool:
        cur = int(self.model.num_timesteps)
        delta = cur - self._last
        self._last = cur
        if _HAS_TQDM and self._tqdm is not None:
            if delta > 0:
                self._tqdm.update(delta)
        return True

    def _on_training_end(self):
        if _HAS_TQDM and self._tqdm is not None:
            self._tqdm.close()
        return None

class PerLearnerCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            for k, v in info.items():
                if isinstance(k, str) and (k.startswith("per_learner/") or k.startswith("group/") or k.startswith("global/")):
                    try:
                        self.logger.record(k, float(v))
                    except Exception:
                        pass
        return True

class EvalTB(EvalCallback):
    def _on_step(self) -> bool:
        result = super()._on_step()
        if (self.n_calls % max(1, self.eval_freq) == 0) and (self.last_mean_reward is not None):
            try:
                self.logger.record("eval/mean_reward", float(self.last_mean_reward))
            except Exception:
                pass
        return result

def make_env(rank, seed, n_learners, n_items, horizon, n_groups):
    def _init():
        env = HierarchicalMultiLearnerEnv(
            n_learners=n_learners,
            n_items=n_items,
            horizon=horizon,
            seed=seed + rank,
            reward_cfg=RewardCfg(),
            curriculum_cfg=CurriculumCfg(),
            hier_cfg=HierCfg(n_groups=n_groups)
        )
        return env
    return _init

def linear_schedule(initial_value: float):
    def func(progress_remaining: float):
        return progress_remaining * initial_value
    return func

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="runs/ppo_hier")
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--total_timesteps", type=int, default=800_000)
    parser.add_argument("--n_learners", type=int, default=8)
    parser.add_argument("--n_items", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=800)
    parser.add_argument("--n_groups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_random_seed(args.seed)

    env_fns = [make_env(i, args.seed, args.n_learners, args.n_items, args.horizon, args.n_groups) for i in range(args.n_envs)]
    vec = SubprocVecEnv(env_fns)
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=5.0, clip_reward=5.0, gamma=0.99)
    vec = VecMonitor(vec)

    eval_env = SubprocVecEnv([make_env(100+i, args.seed, args.n_learners, args.n_items, args.horizon, args.n_groups) for i in range(2)])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=5.0)
    eval_env = VecMonitor(eval_env)

    os.makedirs(args.logdir, exist_ok=True)

    eval_cb = EvalTB(
        eval_env,
        best_model_save_path=os.path.join(args.logdir, "best"),
        log_path=os.path.join(args.logdir, "eval"),
        eval_freq=max(1, 25_000 // max(1, args.n_envs)),
        deterministic=True,
        render=False,
        verbose=1
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, 100_000 // max(1, args.n_envs)),
        save_path=os.path.join(args.logdir, "ckpts"),
        name_prefix="ppo_hier"
    )
    pbar_cb = TqdmCallback(total_timesteps=args.total_timesteps)
    per_learner_cb = PerLearnerCallback()

    policy_kwargs = dict(net_arch=[dict(pi=[256,128], vf=[256,128])])

    model = PPO(
        "MlpPolicy",
        vec,
        learning_rate=linear_schedule(3e-4),
        n_steps=4096,
        batch_size=2048,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=args.logdir,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        verbose=1
    )

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=[eval_cb, ckpt_cb, pbar_cb, per_learner_cb], progress_bar=True)
    except TypeError:
        model.learn(total_timesteps=args.total_timesteps, callback=[eval_cb, ckpt_cb, pbar_cb, per_learner_cb])

    model.save(os.path.join(args.logdir, "final_model"))
    vec.save(os.path.join(args.logdir, "vecnorm.pkl"))

if __name__ == "__main__":
    main()
