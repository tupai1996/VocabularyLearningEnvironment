# rl/train_ppo.py
import os
import argparse
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from .env_tutoring import TutoringEnv

from sb3_contrib.common.wrappers import ActionMasker

def mask_fn(env):
    # robustly reach the base env even if wrapped by Monitor/VecNormalize/etc.
    try:
        return env.unwrapped.action_masks()
    except AttributeError:
        # fallback: walk down .env chain
        while hasattr(env, "env"):
            env = env.env
        return env.action_masks()


def make_env(n_items=500, horizon=600, k=32, seed=42):
    env = TutoringEnv(n_items=n_items, horizon=horizon, candidate_k=k, seed=seed)
    env = Monitor(env)
    env = ActionMasker(env, mask_fn)
    return env

def main(args):
    os.makedirs(args.logdir, exist_ok=True)
    env = make_env(args.n_items, args.horizon, args.k, args.seed)
    eval_env = make_env(args.n_items, args.horizon, args.k, args.seed + 1)

    policy_kwargs = dict(net_arch=[256, 256])
    Model = MaskablePPO  # swap to PPO if you don't want masking

    model = Model(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        tensorboard_log=args.logdir,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.logdir, "best"),
        log_path=os.path.join(args.logdir, "eval"),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True
    )
    ckpt_cb = CheckpointCallback(save_freq=50_000, save_path=args.logdir, name_prefix="ppo_tutor")

    model.learn(total_timesteps=args.total_steps, callback=[eval_cb, ckpt_cb])
    model.save(os.path.join(args.logdir, "final_model"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=str, default="runs/ppo_tutor")
    p.add_argument("--n_items", type=int, default=500)
    p.add_argument("--horizon", type=int, default=600)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--total_steps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
