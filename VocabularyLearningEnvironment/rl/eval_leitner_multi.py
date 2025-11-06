# -*- coding: utf-8 -*-
import argparse, heapq, numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from VocabularyLearningEnvironment.rl.env_tutoring_multi import MultiLearnerTutorEnv

# --- utils ---
def make_env(seed, n_learners, n_items, horizon):
    def _init():
        return MultiLearnerTutorEnv(n_learners=n_learners, n_items=n_items, horizon=horizon, seed=seed)
    return _init

def encode_action(learner_id, item_id, n_items):
    return int(learner_id) * int(n_items) + int(item_id)

# --- per-learner Leitner policy ---
class Leitner:
    def __init__(self, n_items, delta_A=4, delta_B=2):
        self.n_items = n_items
        self.delta_A = delta_A
        self.delta_B = delta_B
        self.reset()

    def reset(self):
        self.t = 0
        self.box = {}          # item_id -> level k
        self.heap = []         # (next_due, seen_order, item_id)
        self.next_new = 0
        self._seen_order = 0

    def _delay(self, k): return int(self.delta_A * (self.delta_B ** k))
    def _sched(self, item, k):
        self._seen_order += 1
        heapq.heappush(self.heap, (self.t + max(1, self._delay(k)), self._seen_order, item))

    def observe(self, item, correct):
        k = self.box.get(item, 0)
        k = k + 1 if correct else max(0, k - 1)
        self.box[item] = k
        self._sched(item, k)

    def act(self, valid_items):
        self.t += 1
        valid = set(valid_items) if valid_items is not None else None

        # overdue selection
        pulled = []
        while self.heap and self.heap[0][0] <= self.t:
            pulled.append(heapq.heappop(self.heap))
        for tup in pulled:
            _, _, item = tup
            if valid is None or item in valid:
                # push back the rest
                for other in pulled:
                    if other is not tup:
                        heapq.heappush(self.heap, other)
                return item
        for other in pulled:
            heapq.heappush(self.heap, other)

        # introduce new
        while self.next_new < self.n_items:
            cand = self.next_new
            self.next_new += 1
            if valid is None or cand in valid:
                self.box[cand] = 0
                self._sched(cand, 0)
                return cand

        # fallback: any valid / random
        if valid and len(valid) > 0:
            return next(iter(valid))
        return np.random.randint(self.n_items)

def get_masks(vec):
    try:
        masks = vec.env_method("get_action_mask")
    except Exception:
        masks = [None] * vec.num_envs
    return masks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=30)
    ap.add_argument("--n_envs", type=int, default=1)  # DummyVecEnv for Windows
    ap.add_argument("--n_learners", type=int, default=8)
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delta_A", type=int, default=4)
    ap.add_argument("--delta_B", type=int, default=2)
    args = ap.parse_args()

    # Single-process for robustness
    vec = DummyVecEnv([make_env(args.seed, args.n_learners, args.n_items, args.horizon)])
    vec = VecMonitor(vec)
    vec = VecNormalize(vec, training=False, norm_obs=True, norm_reward=False)

    returns = []
    for ep in range(args.n_episodes):
        obs = vec.reset()
        done = np.array([False]*vec.num_envs)
        ret = np.zeros(vec.num_envs, dtype=np.float32)

        # one Leitner scheduler per learner
        schedulers = [Leitner(args.n_items, args.delta_A, args.delta_B) for _ in range(args.n_learners)]
        steps = 0
        while not done.all():
            masks = get_masks(vec)  # list per env; here n_envs=1
            # Build actions per env
            actions = []
            for env_i in range(vec.num_envs):
                # decode mask into per-learner valid item sets
                if masks[env_i] is None:
                    # no mask: choose arbitrary learner/item using Leitner
                    l = np.random.randint(args.n_learners)
                    item = schedulers[l].act(valid_items=None)
                    actions.append(encode_action(l, item, args.n_items))
                    continue
                mask = np.asarray(masks[env_i]).astype(bool)
                valid_idxs = np.flatnonzero(mask)
                # map valid action indices -> (learner, item)
                valid_pairs = {(idx // args.n_items, idx % args.n_items) for idx in valid_idxs}

                # pick a learner to act this step: round-robin or most overdue.
                # Simple round-robin works: l = steps % args.n_learners
                l = steps % args.n_learners
                # valid items for this learner:
                valid_items = [it for (ll, it) in valid_pairs if ll == l]
                if len(valid_items) == 0:
                    # pick any other learner that has valid items
                    fallback = None
                    for ll in range(args.n_learners):
                        cand = [it for (lll, it) in valid_pairs if lll == ll]
                        if cand:
                            l = ll; valid_items = cand; break
                    if len(valid_items) == 0:
                        # nothing valid? pick any valid action
                        actions.append(int(np.random.choice(valid_idxs)))
                        continue

                item = schedulers[l].act(valid_items=valid_items)
                actions.append(encode_action(l, item, args.n_items))

            # step
            out = vec.step(np.array(actions))
            if len(out) == 5:
                obs, rew, term, trunc, infos = out
                done = np.logical_or(term, trunc)
            else:
                obs, rew, done, infos = out
            # update schedulers with correctness per chosen (learner,item)
            # we rely on reward sign here; if env exposes info['correct'], use that instead
            for env_i, a in enumerate(actions):
                l = a // args.n_items; it = a % args.n_items
                was_correct = bool(rew[env_i] > 0)
                schedulers[l].observe(it, was_correct)

            ret += rew
            steps += 1
            if steps > args.horizon + 5:
                done[:] = True

        returns.extend(ret.tolist())

    print("\n=== Leitner Baseline (multi-learner, true) ===")
    print(f"Episodes: {args.n_episodes}")
    print(f"Mean reward: {np.mean(returns):.3f} ± {np.std(returns):.3f}")

if __name__ == "__main__":
    main()
