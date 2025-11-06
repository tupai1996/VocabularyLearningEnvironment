# eval_random.py
import argparse
import numpy as np
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import is_masking_supported, get_action_masks
from .env_tutoring import TutoringEnv

def mask_fn(env):
    return env.unwrapped.action_masks()

def make_env(n_items, horizon, seed):
    base = TutoringEnv(n_items=n_items, horizon=horizon, seed=seed)
    base = ActionMasker(base, mask_fn)
    base = Monitor(base)
    return base

def run_random(env, n_episodes):
    rews = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, trunc = False, False
        ep_r = 0.0
        while not (done or trunc):
            if is_masking_supported(env):
                mask = get_action_masks(env)[0]  # usually (n_actions,)
                mask = np.atleast_1d(mask)       # <-- ensure it's always 1D
                if mask.ndim == 0 or mask.shape == ():  # scalar True/False
                    valid_actions = [env.action_space.sample()]
                else:
                    valid_actions = np.where(mask)[0]
                a = np.random.choice(valid_actions)
            else:
                a = env.action_space.sample()
            obs, r, done, trunc, info = env.step(a)
            ep_r += r
        rews.append(ep_r)
    return float(np.mean(rews)), float(np.std(rews))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=20)
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = make_env(args.n_items, args.horizon, args.seed)
    mean_r, std_r = run_random(env, args.n_episodes)
    print(f"\n=== Random Baseline ===")
    print(f"Episodes: {args.n_episodes}")
    print(f"Mean reward: {mean_r:.3f} ± {std_r:.3f}")

if __name__ == "__main__":
    main()
