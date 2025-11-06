# -*- coding: utf-8 -*-
import os, glob, argparse, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor

from VocabularyLearningEnvironment.rl.env_tutoring_multi import MultiLearnerTutorEnv

# ---------- env factory ----------
def make_env(seed, n_learners, n_items, horizon):
    def _init():
        return MultiLearnerTutorEnv(
            n_learners=n_learners,
            n_items=n_items,
            horizon=horizon,
            seed=seed
        )
    return _init

# ---------- helpers ----------
def resolve_model_path(p: str) -> str:
    p = p.strip().strip('"').strip("'").replace(",", "")
    if p.lower().endswith(".zip.zip"):
        p = p[:-4]
    if not p.lower().endswith(".zip") and os.path.exists(p + ".zip"):
        return p + ".zip"
    if os.path.isdir(p):
        for c in [os.path.join(p, "best", "best_model.zip"),
                  os.path.join(p, "final_model.zip")]:
            if os.path.exists(c):
                return c
        zips = sorted(glob.glob(os.path.join(p, "**", "*.zip"), recursive=True))
        if zips:
            return zips[0]
        raise FileNotFoundError(f"No model zip under {p}")
    alt = p[:-4] if p.lower().endswith(".zip") else p + ".zip"
    return p if os.path.exists(p) else (alt if os.path.exists(alt) else p)

def build_eval_vec(n_envs, seed, n_learners, n_items, horizon):
    if n_envs == 1:
        return DummyVecEnv([make_env(seed, n_learners, n_items, horizon)])
    return SubprocVecEnv([make_env(seed+i, n_learners, n_items, horizon) for i in range(n_envs)])

def load_or_warm_vecnorm(vec, vecnorm_path, warmup_steps=10000):
    # If we have stats from training, use them. Otherwise, estimate obs stats
    # with a short random warmup (no reward normalization).
    if vecnorm_path and os.path.exists(vecnorm_path):
        vn = VecNormalize.load(vecnorm_path, vec)
        vn.training = False
        vn.norm_reward = False
        return vn, "loaded"
    # fresh stats: observe obs distribution with random actions
    vn = VecNormalize(vec, training=True, norm_obs=True, norm_reward=False, clip_obs=5.0)
    steps = 0
    done = np.array([False]*vn.num_envs)
    obs = vn.reset()
    while steps < warmup_steps:
        actions = np.array([vn.action_space.sample() for _ in range(vn.num_envs)])
        out = vn.step(actions)
        # support both gymnasium (5-tuple) and gym (4-tuple)
        if len(out) == 5:
            obs, rew, term, trunc, infos = out
            done = np.logical_or(term, trunc)
        else:
            obs, rew, done, infos = out
        steps += 1
    # freeze stats
    vn.training = False
    return vn, "warmed"

def mask_invalid_actions(env, actions):
    """
    Ask each sub-env for its current mask via env_method('get_action_mask').
    If chosen action is invalid, resample uniformly from valid indices.
    """
    fixed = np.array(actions).copy()
    try:
        masks = env.env_method("get_action_mask")  # returns list, one per sub-env
    except Exception:
        return fixed  # no masks available; do nothing

    for i, m in enumerate(masks):
        if m is None:
            continue
        if not bool(m[fixed[i]]):
            valid = np.flatnonzero(m)
            if valid.size > 0:
                fixed[i] = int(np.random.choice(valid))
    return fixed

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--vecnorm_path", type=str, default=None, help="Optional: path to vecnorm.pkl")
    ap.add_argument("--n_episodes", type=int, default=30)
    ap.add_argument("--n_envs", type=int, default=2)
    ap.add_argument("--n_learners", type=int, default=8)
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_mask_fix", action="store_true", help="Resample invalid actions during eval")
    ap.add_argument("--warmup_steps", type=int, default=10000, help="Obs-stat warmup if vecnorm.pkl missing")
    args = ap.parse_args()

    model_path = resolve_model_path(args.model_path)
    model = PPO.load(model_path)
    print(f"Loaded model: {model_path}")

    vec = build_eval_vec(args.n_envs, args.seed, args.n_learners, args.n_items, args.horizon)
    vec, how = load_or_warm_vecnorm(vec, args.vecnorm_path, warmup_steps=args.warmup_steps)
    print(f"VecNormalize stats: {how}")

    returns = []
    for _ in range(args.n_episodes):
        obs = vec.reset()
        done = np.array([False]*vec.num_envs)
        ret = np.zeros(vec.num_envs, dtype=np.float32)
        steps = 0
        while not done.all():
            actions, _ = model.predict(obs, deterministic=True)
            if args.use_mask_fix:
                actions = mask_invalid_actions(vec, np.array(actions))
            out = vec.step(actions)
            if len(out) == 5:
                obs, rew, term, trunc, infos = out
                done = np.logical_or(term, trunc)
            else:
                obs, rew, done, infos = out
            ret += rew
            steps += 1
            if steps > args.horizon + 5:
                done[:] = True
        returns.extend(ret.tolist())

    mean, std = float(np.mean(returns)), float(np.std(returns))
    print("\n=== PPO Multi (strict, comparable) ===")
    print(f"Episodes: {args.n_episodes}")
    print(f"Mean reward: {mean:.3f} ± {std:.3f}")

if __name__ == "__main__":
    main()
