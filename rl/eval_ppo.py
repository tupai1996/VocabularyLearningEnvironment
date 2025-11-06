# eval_ppo.py
import argparse
import numpy as np

from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy

from .env_tutoring import TutoringEnv  # same env you trained on


def mask_fn(env):
    # Works even if env is wrapped (Monitor, Vec, etc.)
    return env.unwrapped.action_masks()


def make_env(n_items, horizon, seed):
    base = TutoringEnv(n_items=n_items, horizon=horizon, seed=seed)
    base = ActionMasker(base, mask_fn)
    base = Monitor(base)
    return base



def maybe_stat(env, name, default=None):
    """
    Try common ways to fetch custom stats from the env.
    Returns default if not found.
    """
    for attr in [name, f"get_{name}"]:
        if hasattr(env.unwrapped, attr):
            fn = getattr(env.unwrapped, attr)
            try:
                return fn() if callable(fn) else fn
            except Exception:
                pass
    return default


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="Path to best_model.zip")
    p.add_argument("--n_episodes", type=int, default=20)
    p.add_argument("--n_items", type=int, default=500)
    p.add_argument("--horizon", type=int, default=600)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = make_env(args.n_items, args.horizon, args.seed)

    print(f"Loading model: {args.model_path}")
    model = MaskablePPO.load(args.model_path, env=env, print_system_info=True)

    # Evaluate with masks-aware evaluator
    mean_r, std_r = evaluate_policy(
        model,
        env,
        n_eval_episodes=args.n_episodes,
        deterministic=True,
        render=False
    )
    print(f"\n=== PPO Evaluation ===")
    print(f"Episodes: {args.n_episodes}")
    print(f"Mean reward: {mean_r:.3f} ± {std_r:.3f}")

    # Optional: fetch domain metrics if your env exposes them
    learned = maybe_stat(env, "learned_items", default=None)
    seen    = maybe_stat(env, "seen_items",    default=None)
    recall  = maybe_stat(env, "recall_probability_hist", default=None)

    if learned is not None:
        print(f"Items learned: {learned}")
    if seen is not None:
        ratio = (learned / max(seen, 1)) if learned is not None else None
        print(f"Items seen: {seen}" + (f" | learned/seen: {ratio:.3f}" if ratio is not None else ""))
    if recall is not None:
        rp = np.array(recall)
        print(f"Recall prob: mean={rp.mean():.3f}, min={rp.min():.3f}, max={rp.max():.3f}")


if __name__ == "__main__":
    main()
