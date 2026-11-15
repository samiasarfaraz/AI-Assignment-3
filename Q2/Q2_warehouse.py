"""
CS 351 - Assignment 03, Part B Q2
Warehouse robot planning with Value Iteration (and Policy Iteration bonus).

Map characters:
    . = free, # = shelf, G = packing station (+100),
    X = spill zone (-100), S = start

Complete the TODO sections according to the assignment instructions.

Terminal command: python .\Q2_warehouse.py --map .\warehouse-1.txt --bonus


"""


import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


ACTIONS = ["U", "D", "L", "R"]
MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
PERP = {"U": ("L", "R"), "D": ("L", "R"), "L": ("U", "D"), "R": ("U", "D")}
ARROW = {"U": (0, -0.35), "D": (0, 0.35), "L": (-0.35, 0), "R": (0.35, 0)}


# ----------------------------------------------------------------------
# Map loading
# ----------------------------------------------------------------------

def load_grid(path):
    """Load the warehouse map as an (H, W) array of single characters."""
    with open(path) as f:
        lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Map file '{path}' is empty")
    width = max(len(ln) for ln in lines)
    rows = [list(ln.ljust(width, "#")) for ln in lines]   # pad ragged rows
    grid = np.array(rows, dtype="<U1")
    bad = set(np.unique(grid)) - set(".#GXS")
    if bad:
        raise ValueError(f"Unknown map symbols: {sorted(bad)}")
    return grid


# ----------------------------------------------------------------------
# Warehouse environment
# ----------------------------------------------------------------------

class Warehouse:
    def __init__(self, grid, goal_reward=100.0, spill_reward=-100.0,
                 p_intended=0.8):
        self.grid = grid
        self.H, self.W = grid.shape
        self.N = self.H * self.W
        self.p = p_intended
        self.goal_reward = goal_reward
        self.spill_reward = spill_reward

        # Boolean masks, each of shape (H, W)
        self.is_wall = grid == "#"
        self.is_goal = grid == "G"
        self.is_spill = grid == "X"
        self.is_term = self.is_goal | self.is_spill
        self.is_free = ~(self.is_wall | self.is_term)   # cells where we act

        # Fixed value of terminal cells (0 everywhere else)
        self.term_value = np.zeros((self.H, self.W))
        self.term_value[self.is_goal] = goal_reward
        self.term_value[self.is_spill] = spill_reward

        # Start cell (flat index). Fall back to the first free cell.
        starts = np.argwhere(grid == "S")
        if len(starts) > 0:
            r0, c0 = starts[0]
        else:
            r0, c0 = np.argwhere(self.is_free)[0]
        self.start = (int(r0), int(c0))
        self.start_idx = self.start[0] * self.W + self.start[1]

        # NS[a, c] = flat index of the cell reached by moving in direction a
        # from flat cell c (staying put if blocked). Shape (4, N).
        self.NS = np.array([[self._move(c, a) for c in range(self.N)]
                            for a in ACTIONS], dtype=np.int64)

        # Transition tables for the deterministic and stochastic worlds
        self._tables = {False: self._transitions(False),
                        True: self._transitions(True)}

    # ---- dynamics ----------------------------------------------------

    def _move(self, c, action):
        """Flat cell c + action -> flat cell index (stay if blocked)."""
        r, col = divmod(c, self.W)
        dr, dc = MOVES[action]
        nr, nc = r + dr, col + dc
        if nr < 0 or nr >= self.H or nc < 0 or nc >= self.W:
            return c
        if self.is_wall[nr, nc]:
            return c
        return nr * self.W + nc

    def _transitions(self, stochastic):
        """For each action: a list of (probability, next_cell_array).
        next_cell_array has one entry per flat cell (shape (N,))."""
        table = []
        slip = (1.0 - self.p) / 2.0
        for a in ACTIONS:
            if not stochastic:
                table.append([(1.0, self.NS[ACTIONS.index(a)])])
            else:
                a1, a2 = PERP[a]
                table.append([
                    (self.p, self.NS[ACTIONS.index(a)]),
                    (slip, self.NS[ACTIONS.index(a1)]),
                    (slip, self.NS[ACTIONS.index(a2)]),
                ])
        return table

    def q_values(self, V, stochastic, gamma, r):
        """Q[a, row, col] = r + gamma * sum_s' P(s'|s,a) V(s').
        V is an (H, W) array; the result has shape (4, H, W)."""
        Vf = V.ravel()
        Q = np.empty((4, self.N))
        for ai, options in enumerate(self._tables[stochastic]):
            expected = sum(p * Vf[ns] for p, ns in options)
            Q[ai] = r + gamma * expected
        return Q.reshape(4, self.H, self.W)

    # ---- Value Iteration ----------------------------------------------

    def value_iteration(self, stochastic=False, gamma=0.9, r=-1.0,
                        theta=1e-4, max_iter=100000):
        """Returns (V, iterations). V has shape (H, W)."""
        V = self.term_value.copy()
        it = 0
        if not self.is_free.any():
            return V, it
        for it in range(1, max_iter + 1):
            Q = self.q_values(V, stochastic, gamma, r)
            V_new = np.where(self.is_free, Q.max(axis=0), self.term_value)
            delta = np.abs(V_new - V)[self.is_free].max()
            V = V_new
            if delta < theta:
                break
        return V, it

    def greedy(self, V, stochastic, gamma, r):
        """Greedy policy: (H, W) array of 'U','D','L','R' ('' if no action)."""
        Q = self.q_values(V, stochastic, gamma, r)
        best = Q.argmax(axis=0)
        policy = np.full((self.H, self.W), "", dtype="<U1")
        letters = np.array(ACTIONS)
        policy[self.is_free] = letters[best][self.is_free]
        return policy

    # ---- Policy Iteration (bonus) --------------------------------------

    def _policy_index(self, policy):
        """(H, W) letters -> flat array of action numbers 0..3."""
        idx = np.zeros(self.N, dtype=np.int64)
        for ai, a in enumerate(ACTIONS):
            idx[(policy == a).ravel()] = ai
        return idx

    def policy_evaluation(self, policy, stochastic, gamma, r):
        """Exact V^pi by solving the linear system (I - gamma P) V = b
        over the free cells. Returns an (H, W) array."""
        free_idx = np.flatnonzero(self.is_free.ravel())
        nf = len(free_idx)
        pos = np.full(self.N, -1, dtype=np.int64)
        pos[free_idx] = np.arange(nf)
        pol = self._policy_index(policy)[free_idx]
        tv = self.term_value.ravel()

        A = np.eye(nf)
        b = np.full(nf, float(r))
        for ai, options in enumerate(self._tables[stochastic]):
            rows = np.flatnonzero(pol == ai)
            for p, ns in options:
                nxt = ns[free_idx[rows]]
                to_free = pos[nxt] >= 0
                np.add.at(A, (rows[to_free], pos[nxt[to_free]]), -gamma * p)
                np.add.at(b, rows[~to_free], gamma * p * tv[nxt[~to_free]])
        Vf = np.linalg.solve(A, b)

        V = self.term_value.copy().ravel()
        V[free_idx] = Vf
        return V.reshape(self.H, self.W)

    def policy_iteration(self, stochastic=True, gamma=0.9, r=-1.0,
                         max_iter=1000):
        """Returns (policy, V, iterations)."""
        letters = np.array(ACTIONS)
        policy = np.full((self.H, self.W), "", dtype="<U1")
        policy[self.is_free] = "U"
        it = 0
        for it in range(1, max_iter + 1):
            V = self.policy_evaluation(policy, stochastic, gamma, r)
            Q = self.q_values(V, stochastic, gamma, r)
            cur = self._policy_index(policy).reshape(self.H, self.W)
            q_cur = np.take_along_axis(Q, cur[None], axis=0)[0]
            keep = q_cur >= Q.max(axis=0) - 1e-9      # avoid tie flip-flops
            new_idx = np.where(keep, cur, Q.argmax(axis=0))
            new_policy = np.full((self.H, self.W), "", dtype="<U1")
            new_policy[self.is_free] = letters[new_idx][self.is_free]
            if np.array_equal(new_policy, policy):
                break
            policy = new_policy
        return policy, V, it

    # ---- Simulation ----------------------------------------------------

    def simulate(self, policy, episodes=1000, r=-1.0,
                 max_steps=500, seed=0):
        """Run `episodes` episodes in the STOCHASTIC world from S.
        Returns a dict with avg_return, avg_steps, success_rate, spill_rate.
        All episodes run in parallel (vectorised over episodes)."""
        rng = np.random.default_rng(seed)
        E = episodes
        pol = self._policy_index(policy)
        tv = self.term_value.ravel()
        goal = self.is_goal.ravel()
        term = self.is_term.ravel()
        perp = np.array([[ACTIONS.index(x) for x in PERP[a]]
                         for a in ACTIONS])
        slip = (1.0 - self.p) / 2.0

        pos = np.full(E, self.start_idx, dtype=np.int64)
        alive = np.ones(E, dtype=bool)
        ret = np.zeros(E)
        steps = np.zeros(E, dtype=np.int64)
        success = np.zeros(E, dtype=bool)
        spilled = np.zeros(E, dtype=bool)

        for _ in range(max_steps):
            if not alive.any():
                break
            a = pol[pos]
            u = rng.random(E)
            a_eff = np.where(u < self.p, a,
                             np.where(u < self.p + slip,
                                      perp[a, 0], perp[a, 1]))
            nxt = self.NS[a_eff, pos]
            ret += np.where(alive, r, 0.0)
            steps += alive.astype(np.int64)
            pos = np.where(alive, nxt, pos)
            done = alive & term[pos]
            ret += np.where(done, tv[pos], 0.0)
            success |= done & goal[pos]
            spilled |= done & ~goal[pos]
            alive &= ~done

        return {"avg_return": float(ret.mean()),
                "avg_steps": float(steps.mean()),
                "success_rate": float(success.mean()),
                "spill_rate": float(spilled.mean())}

    # ---- Plotting ------------------------------------------------------

    def plot(self, V, policy, title, ax, mark=None):
        """Heatmap of V with policy arrows. `mark` = list of (row, col)
        cells to outline in blue."""
        playable = ~self.is_wall
        lo = min(float(V[playable].min()), -1e-6)
        hi = max(float(V[playable].max()), 1e-6)
        norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
        cmap = plt.get_cmap("RdYlGn").copy()
        cmap.set_bad("dimgray")
        im = ax.imshow(np.ma.masked_where(self.is_wall, V),
                       cmap=cmap, norm=norm)
        show_numbers = max(self.H, self.W) <= 12

        for rr in range(self.H):
            for cc in range(self.W):
                ch = self.grid[rr, cc]
                if ch in "GX":
                    ax.text(cc, rr, ch, ha="center", va="center",
                            fontweight="bold", fontsize=11)
                elif self.is_free[rr, cc]:
                    a = policy[rr, cc]
                    if a:
                        dx, dy = ARROW[a]
                        ax.annotate("", xy=(cc + dx, rr + dy),
                                    xytext=(cc - dx, rr - dy),
                                    arrowprops=dict(arrowstyle="->", lw=1.3))
                    if ch == "S":
                        ax.text(cc - 0.4, rr - 0.3, "S", fontsize=8,
                                fontweight="bold", va="center")
                    if show_numbers:
                        ax.text(cc, rr + 0.4, f"{V[rr, cc]:.0f}",
                                ha="center", va="center", fontsize=6)
        if mark is not None:
            for (rr, cc) in mark:
                ax.add_patch(plt.Rectangle((cc - 0.5, rr - 0.5), 1, 1,
                                           fill=False, edgecolor="blue",
                                           lw=2.5))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


# ----------------------------------------------------------------------
# Random warehouse map
# ----------------------------------------------------------------------

def random_map(n=20, shelf=0.22, spills=12, seed=3):
    """Generate a reachable random n x n warehouse map (S top-left,
    G bottom-right). A path from S to G avoiding shelves and spills is
    guaranteed."""
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        grid = np.full((n, n), ".", dtype="<U1")
        grid[rng.random((n, n)) < shelf] = "#"
        grid[0, 0] = "S"
        grid[n - 1, n - 1] = "G"
        free = [(i, j) for i in range(n) for j in range(n)
                if grid[i, j] == "."]
        k = min(spills, len(free))
        for idx in rng.choice(len(free), k, replace=False):
            grid[free[idx]] = "X"

        # breadth-first search from S, never entering shelves or spills
        seen = {(0, 0)}
        frontier = [(0, 0)]
        while frontier:
            r0, c0 = frontier.pop()
            for dr, dc in MOVES.values():
                nr, nc = r0 + dr, c0 + dc
                if (0 <= nr < n and 0 <= nc < n and (nr, nc) not in seen
                        and grid[nr, nc] not in "#X"):
                    seen.add((nr, nc))
                    frontier.append((nr, nc))
        if (n - 1, n - 1) in seen:
            return grid
    raise RuntimeError("Could not generate a reachable map")


# ----------------------------------------------------------------------
# Small output helpers
# ----------------------------------------------------------------------

def save_fig(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=130)
    plt.close(fig)


def save_table(rows, header, out, name, title=""):
    """Print a table, save it as a PNG (for report.pdf) and a CSV."""
    def fmt(v):
        return f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)
    cells = [[fmt(v) for v in r] for r in rows]
    fig, ax = plt.subplots(figsize=(1.7 * len(header) + 1,
                                    0.4 * len(rows) + 1.3))
    ax.axis("off")
    ax.set_title(title)
    tab = ax.table(cellText=cells, colLabels=header, loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1, 1.3)
    save_fig(fig, out, name)
    with open(os.path.join(out, name.replace(".png", ".csv")), "w") as f:
        f.write(",".join(header) + "\n")
        for r in cells:
            f.write(",".join(r) + "\n")
    print("\n" + title)
    print(" | ".join(header))
    for r in cells:
        print(" | ".join(r))


# ----------------------------------------------------------------------
# Main experiments
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="warehouse-1.txt")
    parser.add_argument("--out", default=os.path.join("out", "q2"))
    parser.add_argument("--bonus", action="store_true",
                        help="also run Policy Iteration vs Value Iteration")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    grid = load_grid(args.map)
    wh = Warehouse(grid)
    GAMMA, R = 0.9, -1.0
    print(f"Map '{args.map}': {wh.H} x {wh.W}, start = {wh.start}")

    # ---- Part (a): deterministic Value Iteration -------------------
    V_det, it_det = wh.value_iteration(False, GAMMA, R)
    pol_det = wh.greedy(V_det, False, GAMMA, R)
    print(f"\n(a) Deterministic VI converged in {it_det} iterations")
    fig, ax = plt.subplots(figsize=(7, 6))
    wh.plot(V_det, pol_det,
            f"Deterministic V* and policy ({it_det} iterations)", ax)
    save_fig(fig, args.out, "a_deterministic.png")

    # ---- Part (b): stochastic Value Iteration ----------------------
    V_sto, it_sto = wh.value_iteration(True, GAMMA, R)
    pol_sto = wh.greedy(V_sto, True, GAMMA, R)
    print(f"(b) Stochastic VI converged in {it_sto} iterations")
    diff = [tuple(int(x) for x in rc)
            for rc in np.argwhere((pol_det != pol_sto) & wh.is_free)]
    fig, ax = plt.subplots(figsize=(7, 6))
    wh.plot(V_sto, pol_sto,
            f"Stochastic V* and policy ({it_sto} iterations)\n"
            "blue = policy differs from deterministic", ax, mark=diff)
    save_fig(fig, args.out, "b_stochastic.png")
    print("Cells where the policy differs (row, col): "
          "deterministic -> stochastic")
    for (rr, cc) in diff:
        print(f"  ({rr}, {cc}): {pol_det[rr, cc]} -> {pol_sto[rr, cc]}")

    # ---- Part (c): evaluation by simulation ------------------------
    res_det = wh.simulate(pol_det, 1000, R)
    res_sto = wh.simulate(pol_sto, 1000, R)
    rows = [("deterministic-world policy", res_det["avg_return"],
             res_det["avg_steps"], res_det["success_rate"],
             res_det["spill_rate"]),
            ("stochastic-world policy", res_sto["avg_return"],
             res_sto["avg_steps"], res_sto["success_rate"],
             res_sto["spill_rate"])]
    save_table(rows, ["policy", "avg return", "avg steps",
                      "success rate", "spill rate"], args.out,
               "c_simulation.png",
               "Both policies run in the stochastic world "
               "(1000 episodes, gamma=0.9, r=-1)")

    # ---- Part (d): gamma x r sweep ---------------------------------
    gammas = [0.5, 0.9, 0.99]
    rewards = [-5, -1, 0, 1]
    sweep_rows = []
    fig, axes = plt.subplots(len(gammas), len(rewards), figsize=(5 * len(rewards), 4.6 * len(gammas)))
    for i, g in enumerate(gammas):
        for j, rv in enumerate(rewards):
            V, it = wh.value_iteration(True, g, rv)
            pol = wh.greedy(V, True, g, rv)
            sim = wh.simulate(pol, 1000, rv)
            sweep_rows.append((g, rv, it, sim["success_rate"],
                               sim["avg_steps"], sim["avg_return"]))
            wh.plot(V, pol, f"gamma={g}, r={rv}\n{it} iter, success "
                    f"{sim['success_rate']:.0%}, steps {sim['avg_steps']:.0f}",
                    axes[i, j])
    save_fig(fig, args.out, "d_policy_maps.png")
    save_table(sweep_rows, ["gamma", "r", "iterations", "success rate",
                            "avg steps", "avg return"], args.out,
               "d_sweep_table.png",
               "Stochastic VI: discount and living reward sweep")

    # ---- Bonus: Policy Iteration vs Value Iteration (20 x 20) ------
    if args.bonus:
        big = Warehouse(random_map(20))
        reps = 5
        t = time.perf_counter()
        for _ in range(reps):
            _, it_vi = big.value_iteration(True, GAMMA, R)
        t_vi = (time.perf_counter() - t) / reps
        t = time.perf_counter()
        for _ in range(reps):
            pol_pi, V_pi, it_pi = big.policy_iteration(True, GAMMA, R)
        t_pi = (time.perf_counter() - t) / reps
        V_vi, _ = big.value_iteration(True, GAMMA, R)
        same = np.array_equal(big.greedy(V_vi, True, GAMMA, R), pol_pi)
        save_table([("Value Iteration", it_vi, t_vi),
                    ("Policy Iteration", it_pi, t_pi)],
                   ["method", "iterations", "time (s)"], args.out,
                   "bonus_pi_vs_vi.png",
                   f"20x20 map, same policy: {same}")
        fig, ax = plt.subplots(figsize=(8, 7))
        big.plot(V_pi, pol_pi, "20x20 random map: Policy Iteration", ax)
        save_fig(fig, args.out, "bonus_map.png")

    print(f"\nDone. Results saved in '{args.out}'.")


if __name__ == "__main__":
    main()