# -*- coding: utf-8 -*-
import argparse, numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from VocabularyLearningEnvironment.rl.env_tutoring_multi import MultiLearnerTutorEnv

def make_env(seed, n_learners, n_items, horizon):
    def _init():
        return MultiLearnerTutorEnv(n_learners=n_learners, n_items=n_items, horizon=horizon, seed=seed)
    return _init

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=30)
    ap.add_argument("--n_envs", type=int, default=2)
    ap.add_argument("--n_learners", type=int, default=8)
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    vec = SubprocVecEnv([make_env(args.seed+i, args.n_learners, args.n_items, args.horizon) for i in range(args.n_envs)])
    vec = VecMonitor(vec)
    vec = VecNormalize(vec, training=False, norm_obs=True, norm_reward=False)

    returns = []
    for _ in range(args.n_episodes):
        obs = vec.reset()
        done = np.array([False]*vec.num_envs)
        ret = np.zeros(vec.num_envs, dtype=np.float32)
        while not done.all():
            actions = np.array([vec.action_space.sample() for _ in range(vec.num_envs)])
            out = vec.step(actions)
            if len(out) == 5:
                obs, rew, term, trunc, info = out
                done = np.logical_or(term, trunc)
            else:
                obs, rew, done, info = out
            ret += rew
        returns.extend(ret.tolist())

    print("\n=== Random Baseline (multi-learner) ===")
    print(f"Episodes: {args.n_episodes}")
    print(f"Mean reward: {np.mean(returns):.3f} ± {np.std(returns):.3f}")

if __name__ == "__main__":
    main()
