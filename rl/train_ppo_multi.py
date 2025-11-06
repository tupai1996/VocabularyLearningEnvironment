# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import (
    SubprocVecEnv, VecNormalize, VecMonitor, VecEnvWrapper
)

from VocabularyLearningEnvironment.rl.env_tutoring_multi import MultiLearnerTutorEnv

# -------- Optional tqdm progress for any SB3 version --------
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

    def _on_training_start(self) -> None:
        if _HAS_TQDM:
            self._tqdm = tqdm(total=self.total_timesteps, desc="Training PPO (multi-learner)", unit="steps")
        else:
            print("Training PPO (multi-learner): tqdm not installed. Showing coarse progress...")

    def _on_step(self) -> bool:
        cur = int(self.model.num_timesteps)
        delta = cur - self._last
        self._last = cur
        if _HAS_TQDM and self._tqdm is not None and delta > 0:
            self._tqdm.update(delta)
        return True

    def _on_training_end(self) -> None:
        if _HAS_TQDM and self._tqdm is not None:
            self._tqdm.update(self.total_timesteps - self._tqdm.n)
            self._tqdm.close()


# -------- Per-learner logging to SB3's TB writer --------
class PerLearnerTensorboardCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            for k, v in info.items():
                if isinstance(k, str) and (k.startswith("per_learner/") or k.startswith("global/")):
                    try:
                        self.logger.record(k, float(v))
                    except Exception:
                        pass
        return True


# -------- Eval logging to TB --------
class EvalTensorboard(EvalCallback):
    def _on_step(self) -> bool:
        result = super()._on_step()
        # Log immediately after an eval cycle
        if (self.n_calls % max(1, self.eval_freq) == 0) and (self.last_mean_reward is not None):
            self.logger.record("eval/mean_reward", float(self.last_mean_reward))
            try:
                import numpy as _np
                self.logger.record("eval/ep_len_mean", float(_np.mean(self._episode_lengths)))
            except Exception:
                pass
        return result


# -------- Action-masking wrapper for training --------
class MaskInvalidActionsVec(VecEnvWrapper):
    """
    VecEnv wrapper that replaces invalid actions (per sub-env get_action_mask())
    with a uniformly-sampled valid action before stepping.

    Enable by wrapping the *outermost* vec env: PPO -> MaskInvalidActionsVec(VecMonitor(VecNormalize(Subproc...)))
    """
    def __init__(self, venv, enable_masking: bool = True):
        super().__init__(venv)
        self.enable_masking = enable_masking

    def reset(self):
        """
        Forward reset() to underlying vec env and return obs.
        """
        return self.venv.reset()

    def step_async(self, actions):
        if not self.enable_masking:
            return self.venv.step_async(actions)

        fixed = np.array(actions).copy()
        try:
            # returns list of masks, one per sub-env
            masks = self.venv.env_method("get_action_mask")
            for i, m in enumerate(masks):
                if m is None:
                    continue
                if not bool(m[fixed[i]]):  # invalid action → resample
                    valid = np.flatnonzero(m)
                    if valid.size > 0:
                        fixed[i] = int(np.random.choice(valid))
        except Exception:
            fixed = actions  # fallback: no masking
        return self.venv.step_async(fixed)

    def step_wait(self):
        return self.venv.step_wait()



# -------- Env factory --------
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


# -------- Linear LR schedule --------
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
    parser.add_argument("--mask_training", action="store_true",
                        help="Enable action masking during training (resample invalid actions).")
    args = parser.parse_args()

    set_random_seed(args.seed)

    # ----- Vectorized envs -----
    env_fns = [make_env(i, args.seed, args.n_learners, args.n_items, args.horizon)
               for i in range(args.n_envs)]
    vec = SubprocVecEnv(env_fns)

    # Normalize then monitor
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=5.0, clip_reward=5.0, gamma=0.99)
    vec = VecMonitor(vec)

    # Wrap with training-time masking if requested (must be outermost)
    if args.mask_training:
        vec = MaskInvalidActionsVec(vec, enable_masking=True)

 # ----- Eval env -----
    eval_env = SubprocVecEnv([make_env(10+i, args.seed, args.n_learners, args.n_items, args.horizon) for i in range(2)])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=5.0)
    eval_env = VecMonitor(eval_env)

    # Match training env type if masking enabled
    if args.mask_training:
        eval_env = MaskInvalidActionsVec(eval_env, enable_masking=True)


    os.makedirs(args.logdir, exist_ok=True)

    # ----- Callbacks -----
    eval_cb = EvalTensorboard(
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
        name_prefix="ppo_multi"
    )
    pbar_cb = TqdmCallback(total_timesteps=args.total_timesteps)
    per_learner_cb = PerLearnerTensorboardCallback()

    # ----- PPO config -----
    policy_kwargs = dict(net_arch=dict(pi=[256, 128], vf=[256, 128]))


    model = PPO(
        "MlpPolicy",
        vec,
        learning_rate=linear_schedule(3e-4),
        n_steps=4096,
        batch_size=2048,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=args.logdir,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        verbose=1
    )

    # Learn (support SB3>=2.0 and older)
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[eval_cb, ckpt_cb, pbar_cb, per_learner_cb],
            progress_bar=True
        )
    except TypeError:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[eval_cb, ckpt_cb, pbar_cb, per_learner_cb]
        )

    # Save final + VecNormalize stats
    model.save(os.path.join(args.logdir, "final_model"))
    # If you wrapped with MaskInvalidActionsVec, unwrap before saving VecNormalize stats
    base_vec = vec.venv if isinstance(vec, MaskInvalidActionsVec) else vec
    if isinstance(base_vec, VecMonitor):
        base_vec = base_vec.venv
    if isinstance(base_vec, VecNormalize):
        base_vec.save(os.path.join(args.logdir, "vecnorm.pkl"))


if __name__ == "__main__":
    main()
