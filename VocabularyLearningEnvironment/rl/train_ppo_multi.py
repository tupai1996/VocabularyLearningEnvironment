# -*- coding: utf-8 -*-
import os
import argparse
from functools import partial

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed

from VocabularyLearningEnvironment.rl.env_tutoring_multi import MultiLearnerTutorEnv

# -------- Optional tqdm-based progress bar for older SB3 versions --------
# If SB3>=2.0 you can also rely on model.learn(progress_bar=True).
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


class TqdmCallback(BaseCallback):
    """
    A progress bar that works with any SB3 version.
    - Updates every ._on_step() using model.num_timesteps
    - Closes cleanly at training end
    """
    def __init__(self, total_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self._last = 0
        self._tqdm = None
        self._fallback_printed = False

    def _on_training_start(self) -> None:
        if _HAS_TQDM:
            self._tqdm = tqdm(total=self.total_timesteps, desc="Training PPO (multi-learner)", unit="steps")
        else:
            # Very lightweight fallback
            print("Training PPO (multi-learner): tqdm not installed. Showing coarse progress...")
        return None

    def _on_step(self) -> bool:
        cur = int(self.model.num_timesteps)
        delta = cur - self._last
        self._last = cur
        if _HAS_TQDM and self._tqdm is not None:
            if delta > 0:
                self._tqdm.update(delta)
        else:
            # Fallback: print every 10%
            pct = (cur / max(1, self.total_timesteps)) * 100.0
            # Print at ~10% steps
            bucket = int(pct // 10) * 10
            # store last bucket on self.locals to avoid spam
            lb = getattr(self, "_last_bucket", -1)
            if bucket != lb and bucket <= 100:
                print(f"Progress: {bucket}% ({cur}/{self.total_timesteps} steps)")
                setattr(self, "_last_bucket", bucket)
        return True

    def _on_training_end(self) -> None:
        if _HAS_TQDM and self._tqdm is not None:
            self._tqdm.update(self.total_timesteps - self._tqdm.n)
            self._tqdm.close()
        else:
            print("Training complete.")
        return None


def make_env(rank, seed, n_learners, n_items, horizon):
    def _init():
        env = MultiLearnerTutorEnv(
            n_learners=n_learners,
            n_items=n_items,
            horizon=horizon,
            seed=seed + rank
        )
        return env
    return _init


def linear_schedule(initial_value: float):
    def func(progress_remaining: float):
        return progress_remaining * initial_value
    return func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="runs/ppo_tutor_multi_stable")
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--total_timesteps", type=int, default=1_500_000)
    parser.add_argument("--n_learners", type=int, default=8)
    parser.add_argument("--n_items", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_random_seed(args.seed)

    # ----- Vectorized envs -----
    env_fns = [make_env(i, args.seed, args.n_learners, args.n_items, args.horizon)
               for i in range(args.n_envs)]
    vec = SubprocVecEnv(env_fns)

    # ----- Obs/Reward normalization -----
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=5.0, clip_reward=5.0, gamma=0.99)
    vec = VecMonitor(vec)
    # ----- Eval env (no reward norm update) -----
    eval_env = SubprocVecEnv([make_env(10+i, args.seed, args.n_learners, args.n_items, args.horizon) for i in range(2)])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=5.0)
    eval_env = VecMonitor(eval_env)
    os.makedirs(args.logdir, exist_ok=True)

    # ----- Callbacks -----
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.logdir, "best"),
        log_path=os.path.join(args.logdir, "eval"),
        eval_freq=max(1, 25_000 // max(1, args.n_envs)),
        deterministic=True,
        render=False
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, 100_000 // max(1, args.n_envs)),
        save_path=os.path.join(args.logdir, "ckpts"),
        name_prefix="ppo_multi"
    )

    # Progress bar callback (works for all SB3 versions)
    pbar_cb = TqdmCallback(total_timesteps=args.total_timesteps)

    # ----- PPO config -----
    policy_kwargs = dict(net_arch=[dict(pi=[256, 128], vf=[256, 128])])

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

    # Prefer SB3 native progress bar if available (>=2.0); otherwise TqdmCallback covers it.
    # Using try/except to not rely on version checks.
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[eval_cb, ckpt_cb, pbar_cb],
            progress_bar=True  # native SB3 progress bar (ignored by older SB3)
        )
    except TypeError:
        # Older SB3: no progress_bar kw; still show TqdmCallback
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[eval_cb, ckpt_cb, pbar_cb]
        )

    # Save final + VecNormalize stats
    model.save(os.path.join(args.logdir, "final_model"))
    vec.save(os.path.join(args.logdir, "vecnorm.pkl"))


if __name__ == "__main__":
    main()
