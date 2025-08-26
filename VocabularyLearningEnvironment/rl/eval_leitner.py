# eval_leitner.py
import argparse
import heapq
import numpy as np

from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import is_masking_supported, get_action_masks

from .env_tutoring import TutoringEnv


# ---------- wrappers / helpers ----------

def mask_fn(env):
    # robust to wrappers
    return env.unwrapped.action_masks()

def make_env(n_items, horizon, seed):
    base = TutoringEnv(n_items=n_items, horizon=horizon, seed=seed)
    base = ActionMasker(base, mask_fn)  # expose masks
    base = Monitor(base)
    return base

def maybe_stat(env, name, default=None):
    for attr in [name, f"get_{name}"]:
        if hasattr(env.unwrapped, attr):
            obj = getattr(env.unwrapped, attr)
            try:
                return obj() if callable(obj) else obj
            except Exception:
                pass
    return default


# ---------- Leitner policy core ----------

class LeitnerPolicy:
    """
    Classic Leitner 'boxes' with review intervals:
        next_review_delay = delta_A * (delta_B ** box_k)
    Rules:
      - New item starts in box 0 on first presentation
      - Correct answer -> box += 1
      - Incorrect answer -> box = max(0, box - 1)
      - At each step: choose the item past-due longest; if none due, introduce a new one
    We implement a min-heap keyed by next_due_step; a FIFO queue for overdue ties.
    """
    def __init__(self, n_items, delta_A=4, delta_B=2):
        self.n_items = n_items
        self.delta_A = delta_A
        self.delta_B = delta_B
        self.reset()

    def reset(self):
        self.t = 0
        self.box = {}                  # item_id -> box_k
        self.due_heap = []             # heap of (next_due_step, seen_order, item_id)
        self.seen = set()              # items that have been introduced
        self.next_new = 0              # next unseen item id to introduce (0..n_items-1)
        self._seen_counter = 0         # monotonically increasing to break ties

    def _next_delay(self, k):
        return self.delta_A * (self.delta_B ** k)

    def _schedule(self, item_id, k):
        delay = int(self._next_delay(k))
        next_due = self.t + max(1, delay)
        self._seen_counter += 1
        heapq.heappush(self.due_heap, (next_due, self._seen_counter, item_id))

    def observe(self, item_id, was_correct):
        # update box based on outcome and re-schedule
        k = self.box.get(item_id, 0)
        if was_correct:
            k = k + 1
        else:
            k = max(0, k - 1)
        self.box[item_id] = k
        self._schedule(item_id, k)

    def _pick_due_item(self, valid_set):
        # pop the most overdue item that is valid
        candidates = []
        while self.due_heap and self.due_heap[0][0] <= self.t:
            next_due, order, item_id = heapq.heappop(self.due_heap)
            candidates.append((next_due, order, item_id))
        # among overdue, pick the one with smallest next_due then oldest order that is valid
        for tup in candidates:
            if tup[2] in valid_set:
                # push back the others
                for o in candidates:
                    if o is not tup:
                        heapq.heappush(self.due_heap, o)
                return tup[2]
        # none valid; push them all back
        for o in candidates:
            heapq.heappush(self.due_heap, o)
        return None

    def act(self, valid_actions):
        """
        valid_actions: iterable of allowed action indices (from mask), or None to ignore masking
        returns chosen item index
        """
        self.t += 1
        valid_set = set(valid_actions) if valid_actions is not None else None

        # 1) try overdue review
        if valid_set is None:
            chosen = self._pick_due_item(valid_set=set(range(self.n_items)))
        else:
            chosen = self._pick_due_item(valid_set=valid_set)
        if chosen is not None:
            return chosen

        # 2) else introduce a new item (first unseen that is valid)
        while self.next_new < self.n_items:
            cand = self.next_new
            self.next_new += 1
            if (valid_set is None) or (cand in valid_set):
                # first time: put in box 0 and schedule next review
                self.seen.add(cand)
                self.box[cand] = 0
                self._schedule(cand, 0)
                return cand

        # 3) fallback: if all introduced and nothing overdue/valid, pick any valid (or random)
        if valid_set:
            return next(iter(valid_set))
        return np.random.randint(self.n_items)


# ---------- evaluation loop ----------

def evaluate_leitner(env, n_episodes=20, delta_A=4, delta_B=2):
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, trunc = False, False
        ep_r = 0.0

        policy = LeitnerPolicy(n_items=env.unwrapped.n_items, delta_A=delta_A, delta_B=delta_B)

        while not (done or trunc):
            if is_masking_supported(env):
                mask = get_action_masks(env)
                # vec env returns shape (1, n_actions) or (n_actions,)
                mask = np.asarray(mask)
                if mask.ndim == 2:
                    mask = mask[0]
                mask = np.atleast_1d(mask)
                if mask.shape == () or mask.ndim == 0:  # scalar mask (True/False)
                    valid = None if bool(mask) else []
                else:
                    valid = np.where(mask)[0].tolist()
            else:
                valid = None

            a = policy.act(valid_actions=valid)
            obs, r, done, trunc, info = env.step(a)
            ep_r += r

            # outcome for Leitner box updates: treat reward>0 as success (or check info flag if your env exposes it)
            was_correct = bool(r > 0)
            policy.observe(a, was_correct)

        rewards.append(ep_r)

    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))
    return mean_r, std_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=20)
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delta_A", type=int, default=4, help="base review delay")
    ap.add_argument("--delta_B", type=int, default=2, help="delay multiplier per box level")
    args = ap.parse_args()

    env = make_env(args.n_items, args.horizon, args.seed)
    mean_r, std_r = evaluate_leitner(env, n_episodes=args.n_episodes, delta_A=args.delta_A, delta_B=args.delta_B)

    print("\n=== Leitner Baseline ===")
    print(f"Episodes: {args.n_episodes}")
    print(f"Mean reward: {mean_r:.3f} ± {std_r:.3f}")

    # Optional: print env stats if available
    learned = maybe_stat(env, "learned_items", default=None)
    seen    = maybe_stat(env, "seen_items",    default=None)
    if learned is not None:
        print(f"Items learned: {learned}")
    if seen is not None:
        ratio = (learned / max(seen, 1)) if learned is not None else None
        print(f"Items seen: {seen}" + (f" | learned/seen: {ratio:.3f}" if ratio is not None else ""))


if __name__ == "__main__":
    main()
