#!/usr/bin/env python3
"""napkin-100k — a self-play net that fits in a 100,000-byte source file.

One file, four jobs: an exact offline replica of CodinGame's Ultimate
Tic-Tac-Toe referee (verified bit-level against the venue's own Java engine),
a self-play Q-network trained in it, scripted opponents to measure that net
against, and a packer that emits the trained net as a single self-contained
C++ source. See README.md for the registered hypotheses.

Semantics replicated from the Java source (not folklore):
- level 1 (Wood): a single 3x3 board; win -> mover's score 10, full board -> 0-0 draw.
- level 2: 9 small boards + master board. A move must land in the small board
  addressed by the previous move's local cell, unless that board is decided
  (owned or full) - then anywhere on any undecided board. Winning a small board
  scores +1 and marks the master cell; drawn boards mark nobody. A master
  3-in-a-row sets the mover's score to 10 (dominates any count) and ends the
  game. No valid actions left -> higher score wins, equal -> draw.
- The referee shuffles the valid-action list per game seed: order is noise,
  identity is the set. Parity is therefore checked on SETS.

Usage:
  napkin_100k.py selfcheck
  napkin_referee.py fuzz --other PATH [--games N] [--seed S] [--level {1,2}]
  napkin_referee.py match --a POLICY --b POLICY [--games N] [--seed S] [--level {1,2}]
  napkin_referee.py cg --policy POLICY --level {1,2} [--seed S] [--budget-ms MS]
  napkin_referee.py bench

Opponents (offline measurement only): random | greedy | ab (time-budgeted
iterative-deepening alpha-beta).

  napkin_100k.py train  --iters N            self-play training
  napkin_100k.py pack   --net PT --out CPP   checkpoint -> single C++ source
  napkin_100k.py check-net                   emitted C++ vs torch, move for move
  napkin_100k.py bench-net --vs ab           net vs an opponent, both seats
"""

import argparse
import importlib.util
import random
import sys
import time
from functools import lru_cache

# The 8 winning lines of a 3x3 board, bit i = cell (i//3, i%3).
WIN_MASKS = (0o700, 0o070, 0o007, 0o444, 0o222, 0o111, 0o421, 0o124)
FULL = 0o777


def _wins(mask: int) -> bool:
    return any((mask & m) == m for m in WIN_MASKS)


@lru_cache(maxsize=None)
def _line_threats(mine: int, theirs: int) -> int:
    """Lines where `mine` has exactly 2 cells and `theirs` none (one move from a win)."""
    n = 0
    for m in WIN_MASKS:
        if (theirs & m) == 0 and bin(mine & m).count("1") == 2:
            n += 1
    return n


class Engine:
    """Exact replica of the CG referee's transition function for one game."""

    def __init__(self, level: int):
        assert level in (1, 2)
        self.level = level
        # per-player 9-bit occupancy per small board; level 1 uses board 0 only
        self.boards = [[0] * 9, [0] * 9]
        self.owned = [0, 0]   # 9-bit masks over board indices (level 2)
        self.drawn = 0        # 9-bit mask over board indices (level 2)
        self._scores = [0, 0]
        self.last = None      # previous move (row, col) by either player
        self.moves = 0
        self._over = False
        self._winner = -1
        self._valid = None    # cached set

    # -- read interface ------------------------------------------------------

    @property
    def current_player(self) -> int:
        return self.moves % 2

    @property
    def game_over(self) -> bool:
        return self._over

    @property
    def scores(self):
        return tuple(self._scores)

    @property
    def winner(self) -> int:
        return self._winner

    def _decided(self, b: int) -> bool:
        return bool(((self.owned[0] | self.owned[1] | self.drawn) >> b) & 1)

    def _empties(self, b: int):
        occ = self.boards[0][b] | self.boards[1][b]
        base_r, base_c = (b // 3) * 3, (b % 3) * 3
        return {(base_r + p // 3, base_c + p % 3) for p in range(9)
                if not (occ >> p) & 1}

    def valid_actions(self) -> set:
        if self._valid is not None:
            return self._valid
        if self._over:
            self._valid = set()
        elif self.level == 1:
            occ = self.boards[0][0] | self.boards[1][0]
            self._valid = {(p // 3, p % 3) for p in range(9) if not (occ >> p) & 1}
        else:
            va = set()
            if self.last is not None:
                tb = (self.last[0] % 3) * 3 + self.last[1] % 3
                if not self._decided(tb):
                    va = self._empties(tb)
            if not va:
                va = set()
                for b in range(9):
                    if not self._decided(b):
                        va |= self._empties(b)
            self._valid = va
        return self._valid

    # -- transition ----------------------------------------------------------

    def play(self, row: int, col: int) -> None:
        if (row, col) not in self.valid_actions():
            raise ValueError(f"invalid action ({row} {col})")
        p = self.current_player
        self._valid = None
        if self.level == 1:
            pos = row * 3 + col
            self.boards[p][0] |= 1 << pos
            if _wins(self.boards[p][0]):
                self._scores[p] = 10
                self._end(p)
            elif (self.boards[0][0] | self.boards[1][0]) == FULL:
                self._end_by_score()
        else:
            b = (row // 3) * 3 + col // 3
            pos = (row % 3) * 3 + (col % 3)
            self.boards[p][b] |= 1 << pos
            if _wins(self.boards[p][b]):
                self.owned[p] |= 1 << b
                self._scores[p] += 1
                if _wins(self.owned[p]):
                    self._scores[p] = 10
                    self._end(p)
            elif (self.boards[0][b] | self.boards[1][b]) == FULL:
                self.drawn |= 1 << b
        self.last = (row, col)
        self.moves += 1
        if not self._over and not self.valid_actions():
            self._end_by_score()

    def _end(self, winner: int) -> None:
        self._over = True
        self._winner = winner
        self._valid = None

    def _end_by_score(self) -> None:
        s0, s1 = self._scores
        self._end(0 if s0 > s1 else 1 if s1 > s0 else -1)

    # -- fast copy for search ------------------------------------------------

    def get_state(self):
        return (tuple(self.boards[0]), tuple(self.boards[1]),
                self.owned[0], self.owned[1], self.drawn,
                self._scores[0], self._scores[1],
                self.last, self.moves, self._over, self._winner)

    def set_state(self, s) -> None:
        (b0, b1, self.owned[0], self.owned[1], self.drawn,
         self._scores[0], self._scores[1],
         self.last, self.moves, self._over, self._winner) = s
        self.boards[0] = list(b0)
        self.boards[1] = list(b1)
        self._valid = None


# -- encoders ----------------------------------------------------------------

def action_index(row: int, col: int) -> int:
    """(row, col) in the 9x9 grid -> flat action id 0..80. Level 1 uses 0..2 coords."""
    return row * 9 + col


def index_action(i: int):
    return divmod(i, 9)


def encode_planes(eng: Engine, perspective: int):
    """Flat 4x81 int list, perspective player's view of a level-2 state:
    plane 0 my marks, 1 opponent marks, 2 legal-move mask, 3 my-owned-board mask
    minus opponent-owned (both broadcast over the 9 cells of each small board).
    Deliberately minimal; napkin-selfplay owns the final input spec."""
    me, opp = perspective, 1 - perspective
    planes = [0] * (4 * 81)
    legal = eng.valid_actions() if eng.current_player == perspective else set()
    for r in range(9):
        for c in range(9):
            b = (r // 3) * 3 + c // 3
            pos = (r % 3) * 3 + (c % 3)
            i = r * 9 + c
            planes[i] = (eng.boards[me][b] >> pos) & 1
            planes[81 + i] = (eng.boards[opp][b] >> pos) & 1
            planes[162 + i] = 1 if (r, c) in legal else 0
            planes[243 + i] = ((eng.owned[me] >> b) & 1) - ((eng.owned[opp] >> b) & 1)
    return planes


# -- scripted opponents (offline measurement) --------------------------------------------------

class RandomPolicy:
    name = "random"

    def __init__(self, seed=0, budget_ms=None):
        self.rng = random.Random(seed)

    def act(self, eng: Engine):
        return self.rng.choice(sorted(eng.valid_actions()))


class GreedyPolicy:
    """1-ply: value a move 1000 if it ends the game with us winning, 10 if it
    wins a small board (or wins level-1 outright), else 0; random among argmax."""
    name = "greedy"

    def __init__(self, seed=0, budget_ms=None):
        self.rng = random.Random(seed)

    def act(self, eng: Engine):
        me = eng.current_player
        state = eng.get_state()
        best, best_v = [], -1
        for a in sorted(eng.valid_actions()):
            eng.play(*a)
            v = (1000 if (eng.game_over and eng.winner == me)
                 else 10 if eng.scores[me] > state[5 + me] else 0)
            eng.set_state(state)
            if v > best_v:
                best, best_v = [a], v
            elif v == best_v:
                best.append(a)
        return self.rng.choice(best)


class ABPolicy:
    """Time-budgeted iterative-deepening alpha-beta with a deliberately simple,
    fully documented eval:
      terminal: +/-10000 (win/loss from the mover's side), 0 draw
      else: 100*(my boards - opp boards)
            + 10*(my master-line threats - opp's)   [drawn boards block lines]
            + sum over undecided small boards of (my line threats - opp's)
    Depth is whatever fits the budget - this is a calibration baseline, not a
    contender. Deterministic given seed (ties broken by first-in-sorted-order).
    """
    name = "ab"

    def __init__(self, seed=0, budget_ms=90):
        self.rng = random.Random(seed)
        self.budget = budget_ms / 1000.0
        self.deadline = 0.0

    def _eval(self, eng: Engine, me: int) -> int:
        opp = 1 - me
        if eng.game_over:
            return 10000 if eng.winner == me else 0 if eng.winner == -1 else -10000
        if eng.level == 1:
            return (10 * _line_threats(eng.boards[me][0], eng.boards[opp][0])
                    - 10 * _line_threats(eng.boards[opp][0], eng.boards[me][0]))
        v = 100 * (eng.scores[me] - eng.scores[opp])
        v += 10 * (_line_threats(eng.owned[me], eng.owned[opp] | eng.drawn)
                   - _line_threats(eng.owned[opp], eng.owned[me] | eng.drawn))
        for b in range(9):
            if not eng._decided(b):
                v += (_line_threats(eng.boards[me][b], eng.boards[opp][b])
                      - _line_threats(eng.boards[opp][b], eng.boards[me][b]))
        return v

    def _search(self, eng: Engine, depth: int, alpha: int, beta: int, me: int) -> int:
        if eng.game_over or depth == 0:
            return self._eval(eng, me)
        if time.perf_counter() > self.deadline:
            raise TimeoutError
        state = eng.get_state()
        maximizing = eng.current_player == me
        best = -10**9 if maximizing else 10**9
        for a in sorted(eng.valid_actions()):
            eng.play(*a)
            v = self._search(eng, depth - 1, alpha, beta, me)
            eng.set_state(state)
            if maximizing:
                best = max(best, v)
                alpha = max(alpha, v)
            else:
                best = min(best, v)
                beta = min(beta, v)
            if beta <= alpha:
                break
        return best

    def act(self, eng: Engine):
        me = eng.current_player
        self.deadline = time.perf_counter() + self.budget
        state = eng.get_state()
        actions = sorted(eng.valid_actions())
        best = actions[0]
        depth = 1
        try:
            while depth <= 81:
                d_best, d_val = None, -10**9
                for a in actions:
                    eng.play(*a)
                    v = self._search(eng, depth - 1, -10**9, 10**9, me)
                    eng.set_state(state)
                    if v > d_val:
                        d_best, d_val = a, v
                best = d_best
                depth += 1
                if d_val >= 10000:
                    break
        except TimeoutError:
            eng.set_state(state)
        return best


POLICIES = {"random": RandomPolicy, "greedy": GreedyPolicy, "ab": ABPolicy}


# -- harnesses -----------------------------------------------------------------

def run_game(eng: Engine, pol0, pol1):
    while not eng.game_over:
        pol = pol0 if eng.current_player == 0 else pol1
        eng.play(*pol.act(eng))
    return eng


def cmd_match(args):
    wins = [0, 0, 0]  # p0 (as policy a), p1, draws -- side-swapped fairly
    rng = random.Random(args.seed)
    for g in range(args.games):
        a_first = g % 2 == 0
        pa = POLICIES[args.a](seed=rng.randrange(2**30), budget_ms=args.budget_ms)
        pb = POLICIES[args.b](seed=rng.randrange(2**30), budget_ms=args.budget_ms)
        eng = run_game(Engine(args.level), pa if a_first else pb,
                       pb if a_first else pa)
        w = eng.winner
        if w == -1:
            wins[2] += 1
        else:
            a_won = (w == 0) == a_first
            wins[0 if a_won else 1] += 1
    n = args.games
    p = wins[0] / n
    # Wilson 95% interval on a's win rate (draws count against)
    z = 1.96
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    print(f"{args.a} vs {args.b}  level {args.level}  games {n}: "
          f"{wins[0]} - {wins[1]} - {wins[2]} draws | "
          f"{args.a} win rate {p:.3f} [{mid-half:.3f}, {mid+half:.3f}]")


def load_other(path: str):
    spec = importlib.util.spec_from_file_location("other_engine", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Engine


def cmd_fuzz(args):
    Other = load_other(args.other)
    rng = random.Random(args.seed)
    plies = 0
    t0 = time.time()
    for g in range(args.games):
        gseed = rng.randrange(2**30)
        grng = random.Random(gseed)
        mine, other = Engine(args.level), Other(args.level)
        while True:
            va_m = mine.valid_actions()
            va_o = other.valid_actions()
            ctx = f"game {g} seed {gseed} ply {plies} level {args.level}"
            if va_m != va_o:
                print(f"DIVERGENCE (valid sets) {ctx}\n mine-only: "
                      f"{sorted(va_m - va_o)}\n other-only: {sorted(va_o - va_m)}")
                return 1
            if mine.game_over != other.game_over:
                print(f"DIVERGENCE (game_over) {ctx}: {mine.game_over} vs {other.game_over}")
                return 1
            if mine.game_over:
                if mine.scores != other.scores or mine.winner != other.winner:
                    print(f"DIVERGENCE (outcome) {ctx}: {mine.scores}/{mine.winner} "
                          f"vs {other.scores}/{other.winner}")
                    return 1
                break
            a = grng.choice(sorted(va_m))
            mine.play(*a)
            other.play(*a)
            plies += 1
    dt = time.time() - t0
    print(f"PARITY OK: {args.games} games, {plies} plies, level {args.level}, "
          f"0 divergences ({dt:.1f}s)")
    return 0


def cmd_cg(args):
    """CodinGame protocol adapter. Recomputes the valid set every turn and
    reports any mismatch with what the referee sent (stderr shows in replays).
    Trusts the referee's set for actual play."""
    eng = Engine(args.level)
    pol = POLICIES[args.policy](seed=args.seed, budget_ms=args.budget_ms)
    my_index = None
    while True:
        try:
            opp_row, opp_col = map(int, input().split())
        except EOFError:
            return 0
        if my_index is None:
            my_index = 0 if opp_row == -1 else 1
        if opp_row != -1:
            try:
                eng.play(opp_row, opp_col)
            except ValueError:
                print(f"PARITY: referee move ({opp_row} {opp_col}) invalid in replica",
                      file=sys.stderr)
        if eng.game_over:
            return 0  # referee shouldn't keep talking; don't play on a dead game
        n = int(input())
        ref_va = set()
        for _ in range(n):
            r, c = map(int, input().split())
            ref_va.add((r, c))
        if ref_va != eng.valid_actions():
            print(f"PARITY MISMATCH ply {eng.moves}: ref {sorted(ref_va)} "
                  f"vs replica {sorted(eng.valid_actions())}", file=sys.stderr)
            eng._valid = ref_va  # trust the venue, keep playing
        a = pol.act(eng)
        eng.play(*a)
        print(f"{a[0]} {a[1]}", flush=True)
        if eng.game_over:
            # replica's predicted outcome, compared against the referee's real
            # scores by the Java-side fuzz harness (H1)
            print(f"PREDICT me={my_index} scores={eng.scores[0]},{eng.scores[1]} "
                  f"winner={eng.winner} plies={eng.moves}", file=sys.stderr, flush=True)


def cmd_bench(args):
    eng = Engine(2)
    rng = random.Random(0)
    t0 = time.perf_counter()
    plies = 0
    for _ in range(200):
        eng = Engine(2)
        while not eng.game_over:
            eng.play(*rng.choice(sorted(eng.valid_actions())))
            plies += 1
    dt = time.perf_counter() - t0
    print(f"{plies} plies in {dt:.2f}s = {plies/dt:,.0f} plies/s (random playouts)")


# == the net ===================================================================
# A self-play-trained Q-network for Ultimate Tic-Tac-Toe, sized to the MEASURED
# 100,000-UTF-8-byte source cap (see README H3a). Budget arithmetic, int8 weights
# encoded base85 (4 bytes -> 5 chars = 1.25 chars per weight):
#
#   324 -> 128 -> 128 -> 81  =  68,224 weights + 337 biases
#   68,224 * 1.25            = 85,280 chars of weights
#   + C++ inference harness  = ~6,000 chars
#   ------------------------------------------------  ~91.3k of 100.0k
#
# Recipe carried from series 1/2 standings (replay buffer + target net + 3-step
# returns, no double-Q), re-verified in-domain per house rule. Self-play with a
# league of past selves, exactly as the series design doc registered.

N_IN, N_H1, N_H2, N_ACT = 324, 128, 128, 81


def build_net(device="cpu"):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(N_IN, N_H1), nn.ReLU(),
        nn.Linear(N_H1, N_H2), nn.ReLU(),
        nn.Linear(N_H2, N_ACT),
    ).to(device)


def net_param_bytes():
    """Weights only (biases stay fp32 in the emitted source; they are tiny)."""
    return N_IN * N_H1 + N_H1 * N_H2 + N_H2 * N_ACT


class VecSelfPlay:
    """B games stepped in lockstep so every decision is one batched GPU forward.

    Perspective convention, and it is the easy thing to get wrong: states are
    always encoded for the player TO MOVE, and rewards are +1 win / -1 loss /
    0 draw from that same player's view. A transition's `next_state` therefore
    belongs to the OPPONENT, so its bootstrapped value must be negated.
    """

    def __init__(self, batch, seed=0):
        self.batch = batch
        self.rng = random.Random(seed)
        self.games = [Engine(2) for _ in range(batch)]

    def reset_done(self):
        for i, g in enumerate(self.games):
            if g.game_over:
                self.games[i] = Engine(2)

    def encode_batch(self):
        import numpy as np
        x = np.zeros((self.batch, N_IN), dtype=np.float32)
        masks = np.full((self.batch, N_ACT), -1e9, dtype=np.float32)
        for i, g in enumerate(self.games):
            x[i] = encode_planes(g, g.current_player)
            for (r, c) in g.valid_actions():
                masks[i, action_index(r, c)] = 0.0
        return x, masks


def _masked_greedy(q, mask):
    return (q + mask).argmax(axis=1)


def selfplay_episodes(net, device, games, eps, seed, opponent=None):
    """Play `games` complete games; return transitions and the win record.

    opponent=None -> pure self-play (the net moves for both seats).
    opponent=callable(engine)->(r,c) -> the net plays seat 0/1 alternately
    against a fixed opponent; used for league play and evaluation.
    """
    import numpy as np
    import torch

    transitions = []   # (state, action, reward, next_state, next_mask, sign)
    results = {"win": 0, "loss": 0, "draw": 0}
    rng = random.Random(seed)
    B = min(256, games)
    vec = VecSelfPlay(B, seed)
    pending = [[] for _ in range(B)]      # per-game (state, action, mover)
    net_seat = [g % 2 for g in range(B)]  # which seat the net owns (opponent mode)
    remaining = games - B
    finished = 0

    while finished < games:
        x, mask = vec.encode_batch()
        need_net = [i for i, g in enumerate(vec.games)
                    if opponent is None or g.current_player == net_seat[i]]
        acts = {}
        if need_net:
            with torch.no_grad():
                q = net(torch.from_numpy(x[need_net]).to(device)).cpu().numpy()
            pick = _masked_greedy(q, mask[need_net])
            for k, i in enumerate(need_net):
                if rng.random() < eps:
                    acts[i] = action_index(*rng.choice(sorted(vec.games[i].valid_actions())))
                else:
                    acts[i] = int(pick[k])
        for i, g in enumerate(vec.games):
            if i in acts:
                a = acts[i]
                pending[i].append((x[i].copy(), mask[i].copy(), a,
                                   g.current_player))
            else:
                a = action_index(*opponent(g))
            g.play(*index_action(a))

        for i, g in enumerate(vec.games):
            if not g.game_over:
                continue
            finished += 1
            w = g.winner
            if opponent is not None:
                if w == net_seat[i]:
                    results["win"] += 1
                elif w == -1:
                    results["draw"] += 1
                else:
                    results["loss"] += 1
            # credit every stored decision from ITS mover's point of view
            for t, (st, _m, act, mover) in enumerate(pending[i]):
                # DQN reward is IMMEDIATE: zero everywhere except the mover's
                # last decision, which carries the outcome. Putting the outcome on
                # every transition AND bootstrapping double-counts the return.
                outcome = 0.0 if w == -1 else (1.0 if w == mover else -1.0)
                r, nxt, nmask, sign = outcome, None, None, 1.0
                if t + 1 < len(pending[i]):
                    r = 0.0
                    nxt = pending[i][t + 1][0]
                    # the next state's LEGAL-move mask must travel with it: Q values
                    # of illegal actions are never trained, so an unmasked max over
                    # them bootstraps on free-drifting garbage and diverges.
                    nmask = pending[i][t + 1][1]
                    # sign depends on WHOSE state comes next: in pure self-play the
                    # next stored state is the opponent's (negate); against a fixed
                    # opponent both stored states are ours (don't).
                    sign = 1.0 if pending[i][t + 1][3] == mover else -1.0
                transitions.append((st, act, r, nxt, nmask, sign))
            pending[i] = []
            net_seat[i] = 1 - net_seat[i]
            if remaining > 0:
                remaining -= 1
            vec.games[i] = Engine(2)
    return transitions, results


def cmd_train(args):
    import numpy as np
    import torch
    import torch.nn.functional as F

    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    net = build_net(device)
    target = build_net(device)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    buf = []
    t0 = time.time()
    league = []          # snapshots of past selves
    log = []
    for it in range(1, args.iters + 1):
        eps = max(args.eps_final, args.eps_start * (1 - it / max(1, args.iters * 0.6)))
        opp = None
        if league and random.random() < args.league_frac:
            snap = random.choice(league)
            opp = make_net_policy(snap, device)
        tr, _ = selfplay_episodes(net, device, args.games_per_iter, eps,
                                  args.seed * 1000 + it, opponent=opp)
        buf.extend(tr)
        if len(buf) > args.buffer:
            buf = buf[-args.buffer:]

        for _ in range(args.updates_per_iter):
            batch = random.sample(buf, min(args.batch, len(buf)))
            st = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
            ac = torch.tensor([b[1] for b in batch], device=device, dtype=torch.long)
            rw = torch.tensor([b[2] for b in batch], device=device, dtype=torch.float32)
            nxt_idx = [i for i, b in enumerate(batch) if b[3] is not None]
            q = net(st).gather(1, ac[:, None]).squeeze(1)
            tgt = rw.clone()
            if nxt_idx:
                ns = torch.from_numpy(np.stack([batch[i][3] for i in nxt_idx])).to(device)
                nm = torch.from_numpy(
                    np.stack([batch[i][4] for i in nxt_idx])).to(device)
                sg = torch.tensor([batch[i][5] for i in nxt_idx],
                                   device=device, dtype=torch.float32)
                with torch.no_grad():
                    # mask illegal actions (-1e9) before the max
                    nq = (target(ns) + nm).max(dim=1).values
                # sign carries whose turn the next state is (see selfplay_episodes)
                tgt[nxt_idx] = rw[nxt_idx] + args.gamma * sg * nq
            loss = F.smooth_l1_loss(q, tgt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()

        if it % args.target_every == 0:
            target.load_state_dict(net.state_dict())
        if it % args.league_every == 0:
            snap = build_net(device)
            snap.load_state_dict(net.state_dict())
            league.append(snap)
            if len(league) > args.league_size:
                league.pop(0)
        if it % args.eval_every == 0 or it == args.iters:
            wr = evaluate_net(net, device, args.eval_games, args.seed + it)
            line = (f"iter {it}/{args.iters} eps {eps:.2f} buf {len(buf)} "
                    f"loss {float(loss):.4f} vs-random {wr['random']:.3f} "
                    f"vs-greedy {wr['greedy']:.3f} ({time.time()-t0:.0f}s)")
            print(line, flush=True)
            log.append(line)

    torch.save({"state_dict": net.state_dict(),
                "shape": [N_IN, N_H1, N_H2, N_ACT]}, args.out)
    print(f"saved {args.out} ({net_param_bytes()} weights, "
          f"{net_param_bytes()*1.25/1000:.1f}k chars as int8+base85)")
    return 0


def make_net_policy(net, device):
    """Wrap a torch net as a greedy policy over legal moves."""
    import numpy as np
    import torch

    def policy(eng):
        x = np.asarray(encode_planes(eng, eng.current_player), dtype=np.float32)
        with torch.no_grad():
            q = net(torch.from_numpy(x[None]).to(device)).cpu().numpy()[0]
        best, bv = None, -1e18
        for (r, c) in eng.valid_actions():
            v = q[action_index(r, c)]
            if v > bv:
                bv, best = v, (r, c)
        return best
    return policy


def evaluate_net(net, device, games, seed):
    """Win rate (draws count half) against the scripted baselines, both seats."""
    out = {}
    for name in ("random", "greedy"):
        pol = POLICIES[name](seed=seed)
        _, res = selfplay_episodes(net, device, games, 0.0, seed,
                                   opponent=lambda e, p=pol: p.act(e))
        n = res["win"] + res["loss"] + res["draw"]
        out[name] = (res["win"] + 0.5 * res["draw"]) / max(1, n)
    return out


# == the packer: torch checkpoint -> one self-contained C++ source =============
# "One file that writes one file." Weights are per-tensor symmetrically quantised
# to int8, emitted as base85 text (4 bytes -> 5 chars), and read back by a
# hand-rolled forward pass. No libraries exist inside a CodinGame submission.

NET_CPP_TEMPLATE = r"""/* napkin-100k: a self-play-trained net, weights and all, in one file.
 * Disclosed bot - github.com/arose26/napkin-100k, account Napkin100k.
 * Net: %(shape)s, int8 weights decoded from base85 below.
 * CG compiles at -O0 by default, hence the pragma. */
#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline")
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>

static const int N_IN = %(n_in)d, N_H1 = %(n_h1)d, N_H2 = %(n_h2)d, N_ACT = %(n_act)d;
static const float S1 = %(s1).9gf, S2 = %(s2).9gf, S3 = %(s3).9gf;
static const char* W85 =
%(w85)s;
static const float B1[] = {%(b1)s};
static const float B2[] = {%(b2)s};
static const float B3[] = {%(b3)s};

static int8_t W[%(nw)d];

/* base85 (RFC1924 alphabet, 5 chars -> 4 bytes), matching the Python emitter */
static void decodeWeights() {
    static const char* A = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                           "abcdefghijklmnopqrstuvwxyz!#$%%&()*+-;<=>?@^_`{|}~";
    int inv[256]; for (int i = 0; i < 256; i++) inv[i] = -1;
    for (int i = 0; i < 85; i++) inv[(unsigned char)A[i]] = i;
    size_t n = strlen(W85); size_t out = 0;
    for (size_t i = 0; i + 4 < n; i += 5) {
        uint32_t v = 0;
        for (int k = 0; k < 5; k++) v = v * 85u + (uint32_t)inv[(unsigned char)W85[i+k]];
        for (int k = 3; k >= 0; k--) {
            if (out + (size_t)k < (size_t)%(nw)d) W[out + k] = (int8_t)((v >> (8 * (3 - k))) & 0xFF);
        }
        out += 4;
    }
}

static float h1[N_H1], h2[N_H2], qout[N_ACT];

static void forward(const float* x) {
    const int8_t* w = W;
    for (int j = 0; j < N_H1; j++) {
        float a = 0.f;
        for (int i = 0; i < N_IN; i++) a += x[i] * (float)w[j * N_IN + i];
        a = a * S1 + B1[j];
        h1[j] = a > 0.f ? a : 0.f;
    }
    w += N_IN * N_H1;
    for (int j = 0; j < N_H2; j++) {
        float a = 0.f;
        for (int i = 0; i < N_H1; i++) a += h1[i] * (float)w[j * N_H1 + i];
        a = a * S2 + B2[j];
        h2[j] = a > 0.f ? a : 0.f;
    }
    w += N_H1 * N_H2;
    for (int j = 0; j < N_ACT; j++) {
        float a = 0.f;
        for (int i = 0; i < N_H2; i++) a += h2[i] * (float)w[j * N_H2 + i];
        qout[j] = a * S3 + B3[j];
    }
}

/* ---- game state, mirroring the verified Python engine's encoder ---- */
static const uint16_t WINM[8] = {0700,070,07,0444,0222,0111,0421,0124};
static bool winsMask(uint16_t m){for(int i=0;i<8;i++) if((m&WINM[i])==WINM[i]) return true; return false;}
static uint16_t bd[2][9]; static uint16_t owned[2]; static uint16_t drawnM;

static void applyMove(int r,int c,int p){
    int b=(r/3)*3+c/3, i=(r%%3)*3+(c%%3);
    bd[p][b]|=(uint16_t)(1<<i);
    if(winsMask(bd[p][b])) owned[p]|=(uint16_t)(1<<b);
    else if((bd[0][b]|bd[1][b])==0777) drawnM|=(uint16_t)(1<<b);
}

/* encode_planes(): [mine | theirs | legal | owned_diff], 4 x 81 */
static float feat[4*81];
static void encode(int me,const int* legalCells,int nLegal){
    memset(feat,0,sizeof(feat));
    int opp=1-me;
    for(int r=0;r<9;r++)for(int c=0;c<9;c++){
        int b=(r/3)*3+c/3, i=(r%%3)*3+(c%%3), idx=r*9+c;
        feat[idx]        = (float)((bd[me][b]>>i)&1);
        feat[81+idx]     = (float)((bd[opp][b]>>i)&1);
        feat[243+idx]    = (float)(((owned[me]>>b)&1) - ((owned[opp]>>b)&1));
    }
    for(int k=0;k<nLegal;k++) feat[162+legalCells[k]] = 1.f;
}

int main(){
    decodeWeights();
    int me=-1;
    while(true){
        int orow,ocol;
        if(scanf("%%d%%d",&orow,&ocol)!=2) return 0;
        if(orow<-1||orow>8||ocol<-1||ocol>8) return 1;
        if(me<0) me = (orow==-1)?0:1;
        if(orow>=0&&ocol>=0) applyMove(orow,ocol,1-me);
        int n; if(scanf("%%d",&n)!=1||n<1||n>81) return 1;
        int cells[81];
        for(int i=0;i<n;i++){int r,c; if(scanf("%%d%%d",&r,&c)!=2) return 1;
            if(r<0||r>8||c<0||c>8) return 1; cells[i]=r*9+c;}
        encode(me,cells,n);
        forward(feat);
        int best=cells[0]; float bv=-1e30f;
        for(int i=0;i<n;i++){ if(qout[cells[i]]>bv){bv=qout[cells[i]];best=cells[i];} }
        applyMove(best/9,best%%9,me);
        printf("%%d %%d\n",best/9,best%%9);
        fflush(stdout);
    }
}
"""

B85_ALPHABET = ("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "abcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~")


def b85_encode(data: bytes) -> str:
    """RFC1924-style base85; 4 bytes -> 5 chars. Pads the tail with zeros."""
    out = []
    pad = (-len(data)) % 4
    data = data + b"\x00" * pad
    for i in range(0, len(data), 4):
        v = int.from_bytes(data[i:i + 4], "big")
        chunk = []
        for _ in range(5):
            v, rem = divmod(v, 85)
            chunk.append(B85_ALPHABET[rem])
        out.append("".join(reversed(chunk)))
    return "".join(out)


def b85_decode(text: str, n: int) -> bytes:
    out = bytearray()
    inv = {c: i for i, c in enumerate(B85_ALPHABET)}
    for i in range(0, len(text) - 4, 5):
        v = 0
        for k in range(5):
            v = v * 85 + inv[text[i + k]]
        out.extend(v.to_bytes(4, "big"))
    return bytes(out[:n])


def quantize_int8(w):
    """Per-tensor symmetric quantisation. Returns (int8 array, scale)."""
    import numpy as np
    s = float(np.abs(w).max()) / 127.0
    if s == 0:
        s = 1e-8
    q = np.clip(np.rint(w / s), -127, 127).astype(np.int8)
    return q, s


def cmd_pack(args):
    import numpy as np
    import torch

    ck = torch.load(args.net, map_location="cpu")
    sd = ck["state_dict"]
    n_in, n_h1, n_h2, n_act = ck["shape"]
    # Linear stores weight as (out, in); the C++ reads it row-major the same way.
    w1, b1 = sd["0.weight"].numpy(), sd["0.bias"].numpy()
    w2, b2 = sd["2.weight"].numpy(), sd["2.bias"].numpy()
    w3, b3 = sd["4.weight"].numpy(), sd["4.bias"].numpy()
    q1, s1 = quantize_int8(w1)
    q2, s2 = quantize_int8(w2)
    q3, s3 = quantize_int8(w3)
    blob = q1.tobytes() + q2.tobytes() + q3.tobytes()
    txt = b85_encode(blob)
    # split into C string literal chunks so the line length stays sane
    chunks = [txt[i:i + 100] for i in range(0, len(txt), 100)]
    w85 = "\n".join(f'  "{c}"' for c in chunks)
    fmt = lambda a: ",".join(f"{float(v):.6g}f" for v in a)
    src = NET_CPP_TEMPLATE % {
        "shape": f"{n_in}-{n_h1}-{n_h2}-{n_act}",
        "n_in": n_in, "n_h1": n_h1, "n_h2": n_h2, "n_act": n_act,
        "s1": s1, "s2": s2, "s3": s3, "w85": w85, "nw": len(blob),
        "b1": fmt(b1), "b2": fmt(b2), "b3": fmt(b3),
    }
    with open(args.out, "w") as f:
        f.write(src)
    n = len(src.encode("utf-8"))
    print(f"packed {args.net} -> {args.out}: {n} UTF-8 bytes "
          f"({100000 - n} under the measured cap), {len(blob)} int8 weights")
    if n > 100000:
        print("OVER THE CAP - this source would be rejected by the venue")
        return 1
    return 0


def cmd_check_net(args):
    """H8-3: does the emitted C++ pick the same move as the torch net?

    Drives the compiled bot and the torch net through identical random legal
    positions via the CG protocol, comparing argmax over legal moves. A packer
    that silently changes the policy invalidates every downstream ladder claim,
    so this is the gate before any submission."""
    import numpy as np
    import shutil
    import subprocess
    import tempfile
    import os
    import torch

    ck = torch.load(args.net, map_location="cpu")
    net = build_net("cpu")
    net.load_state_dict(ck["state_dict"])
    net.eval()

    tmp = tempfile.mkdtemp()
    exe = os.path.join(tmp, "bot")
    subprocess.run(["g++", "-O2", "-o", exe, args.cpp], check=True)

    rng = random.Random(args.seed)
    agree = total = 0
    disagreements = []
    for game in range(args.games):
        bot_seat = game % 2
        eng = Engine(2)
        proc = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, bufsize=1)
        try:
            while not eng.game_over:
                if eng.current_player != bot_seat:
                    eng.play(*rng.choice(sorted(eng.valid_actions())))
                    continue
                va = sorted(eng.valid_actions())
                last = eng.last if eng.last is not None else (-1, -1)
                proc.stdin.write(f"{last[0]} {last[1]}\n{len(va)}\n")
                for a in va:
                    proc.stdin.write(f"{a[0]} {a[1]}\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
                if not line:
                    raise AssertionError("packed bot died")
                cpp_mv = tuple(int(v) for v in line.split())
                assert cpp_mv in eng.valid_actions(), f"packed bot played illegal {cpp_mv}"
                x = np.asarray(encode_planes(eng, eng.current_player), dtype=np.float32)
                with torch.no_grad():
                    q = net(torch.from_numpy(x[None])).numpy()[0]
                torch_mv = max(va, key=lambda a: q[action_index(*a)])
                total += 1
                if cpp_mv == torch_mv:
                    agree += 1
                elif len(disagreements) < 5:
                    gap = abs(q[action_index(*cpp_mv)] - q[action_index(*torch_mv)])
                    disagreements.append((cpp_mv, torch_mv, gap))
                eng.play(*cpp_mv)
        finally:
            proc.kill()
            proc.wait()
    shutil.rmtree(tmp, ignore_errors=True)
    pct = 100.0 * agree / max(1, total)
    print(f"check-net: {agree}/{total} decisions match torch ({pct:.2f}%)")
    for c, t, g in disagreements:
        print(f"  cpp {c} vs torch {t}, |dQ| = {g:.5f}")
    return 0 if pct >= args.min_agree else 1


def cmd_bench_net(args):
    """Net vs a scripted baseline, both seats, with a Wilson interval.

    --net uses the torch checkpoint; --cpp uses the packed C++ bot instead, which
    is what actually gets submitted. Comparing the two answers H8-2 (how much
    strength int8 costs) without any hand-waving."""
    import shutil
    import subprocess
    import tempfile
    import os

    proc = None
    tmp = None
    if args.cpp:
        tmp = tempfile.mkdtemp()
        exe = os.path.join(tmp, "bot")
        subprocess.run(["g++", "-O2", "-o", exe, args.cpp], check=True)
        net_move = None
    else:
        import torch
        ck = torch.load(args.net, map_location=args.device)
        net = build_net(args.device)
        net.load_state_dict(ck["state_dict"])
        net.eval()
        net_move = make_net_policy(net, args.device)

    rng = random.Random(args.seed)
    w = l = d = 0
    for g in range(args.games):
        seat = g % 2
        eng = Engine(2)
        opp = POLICIES[args.vs](seed=rng.randrange(2**30), budget_ms=args.budget_ms)
        if args.cpp:
            proc = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, text=True, bufsize=1)
        try:
            while not eng.game_over:
                if eng.current_player == seat:
                    if args.cpp:
                        va = sorted(eng.valid_actions())
                        last = eng.last if eng.last is not None else (-1, -1)
                        proc.stdin.write(f"{last[0]} {last[1]}\n{len(va)}\n")
                        for a in va:
                            proc.stdin.write(f"{a[0]} {a[1]}\n")
                        proc.stdin.flush()
                        line = proc.stdout.readline()
                        if not line:
                            raise AssertionError("packed bot died")
                        mv = tuple(int(v) for v in line.split())
                    else:
                        mv = net_move(eng)
                    assert mv in eng.valid_actions(), f"net played illegal {mv}"
                    eng.play(*mv)
                else:
                    eng.play(*opp.act(eng))
        finally:
            if proc:
                proc.kill()
                proc.wait()
        if eng.winner == seat:
            w += 1
        elif eng.winner == -1:
            d += 1
        else:
            l += 1
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    n = w + l + d
    p_hat = (w + 0.5 * d) / n
    z = 1.96
    den = 1 + z * z / n
    mid = (p_hat + z * z / (2 * n)) / den
    half = z * ((p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) ** 0.5) / den
    who = args.cpp or args.net
    print(f"{who} vs {args.vs}: {w}W-{l}L-{d}D of {n} | score {p_hat:.3f} "
          f"[{mid-half:.3f}, {mid+half:.3f}] (95% Wilson)")
    return 0


# -- ladder snapshot (platform-evidence habit, carried from series 2) ----------

CG_LB = "https://www.codingame.com/services/Leaderboards/getFilteredPuzzleLeaderboard"
CG_HANDLE = "22639068dad6ecdf6717bb383d739a954432057"  # Napkin100k, public


def cmd_snapshot(args):
    """Append one ladder snapshot (our rank + league sizes + top 5) to a JSONL.
    Unauthenticated and read-only; keep the cadence polite (daily)."""
    import datetime
    import json
    import urllib.request

    def post(body):
        req = urllib.request.Request(CG_LB, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=20))

    mine = post([args.arena, CG_HANDLE, "global",
                 {"active": True, "column": "KEYWORD", "filter": args.pseudo}])
    top = post([args.arena, "", "global",
                {"active": False, "column": "", "filter": ""}])
    me = (mine.get("users") or [None])[0]
    snap = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "arena": args.arena,
        "total_bots": mine.get("count"),
        "leagues": {k: v.get("divisionAgentsCount")
                    for k, v in (top.get("leagues") or {}).items()},
        "me": None if not me else {
            "pseudo": me.get("pseudo"), "global_rank": me.get("rank"),
            "score": me.get("score"),
            "league_index": (me.get("league") or {}).get("divisionIndex")},
        "top5": [{"rank": u["rank"], "pseudo": u["pseudo"], "score": u.get("score"),
                  "league_index": (u.get("league") or {}).get("divisionIndex")}
                 for u in (top.get("users") or [])[:5]],
    }
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(snap) + "\n")
    print(json.dumps(snap))
    return 0


# -- selfcheck -----------------------------------------------------------------

def cmd_selfcheck(args):
    # Level 1: X wins on a hand-checked script; draw on another.
    e = Engine(1)
    for mv in [(0, 0), (1, 1), (0, 1), (2, 2), (0, 2)]:  # X top row
        e.play(*mv)
    assert e.game_over and e.winner == 0 and e.scores == (10, 0), e.scores
    e = Engine(1)
    for mv in [(0, 0), (1, 1), (2, 2), (0, 1), (2, 1), (2, 0),
               (0, 2), (1, 2), (1, 0)]:  # known full-board draw
        e.play(*mv)
    assert e.game_over and e.winner == -1 and e.scores == (0, 0)

    # Level 2: first move is free (81 actions), then redirect to (r%3, c%3).
    e = Engine(2)
    assert len(e.valid_actions()) == 81
    e.play(4, 4)  # center cell of center board -> target board (1,1) = center again
    assert e.valid_actions() == {(r, c) for r in (3, 4, 5) for c in (3, 4, 5)} - {(4, 4)}

    # Win a small board -> +1 score, master cell marked, board closed.
    e = Engine(2)
    # P0 plays (0,0),(1,1),(2,2)? those land in different boards. Script a
    # top-left-board win: P0 takes (0,0),(1,1),(2,2) of board 0 = cells
    # (0,0),(1,1),(2,2); P1 must be redirected elsewhere in between.
    e.play(0, 0)   # P0 board 0 cell (0,0); target board (0,0) again
    e.play(1, 1)   # P1 board 0 cell (1,1)! (redirect sent P1 into board 0)
    # P1's move (1,1) -> target board (1,1)
    e.play(4, 4)   # P0 board 4; target (1,1) again
    e.play(3, 3)   # P1 board 4 cell (0,0); target (0,0) = board 0
    e.play(1, 0)   # P0 board 0 cell (1,0); target (1,0) = board 3
    e.play(3, 0)   # P1 board 3 cell (0,0); target (0,0) = board 0
    e.play(2, 0)   # P0 board 0 cells now (0,0),(1,0),(2,0) = left column -> WIN
    assert e.scores == (1, 0) and e.owned[0] == 1 and not e.game_over
    # board 0 closed: P0's move (2,0) -> target (2,0) = board 6, open
    assert all((r // 3, c // 3) == (2, 0) for r, c in e.valid_actions())

    # Invariant fuzz: 400 random games/level; alternation, closure, scoring.
    rng = random.Random(7)
    for level in (1, 2):
        for _ in range(400 if level == 2 else 100):
            e = Engine(level)
            empt = 81 if level == 2 else 9
            while not e.game_over:
                va = e.valid_actions()
                assert va, "no actions but not over"
                mover = e.current_player
                s_before = e.scores[mover]
                e.play(*rng.choice(sorted(va)))
                assert e.scores[mover] >= s_before
                assert e.moves <= empt
            assert e.valid_actions() == set()
            s0, s1 = e.scores
            assert e.winner == (0 if s0 > s1 else 1 if s1 > s0 else -1)
            if 10 in e.scores:
                assert e.winner == e.scores.index(10)
            if level == 2 and 10 not in e.scores:
                assert s0 <= 9 and s1 <= 9

    # Encoder sanity.
    e = Engine(2)
    e.play(4, 4)
    pl = encode_planes(e, 1)
    assert len(pl) == 324 and pl[81 + action_index(4, 4)] == 1
    assert sum(pl[162:243]) == 8  # 8 legal replies in the center board
    assert action_index(*index_action(80)) == 80

    # base85 round-trip: the packer's encoder and the C++ decoder must agree on
    # every byte value, including the signed-int8 edges. A single-byte slip here
    # silently corrupts weights and shows up only as a mysteriously weak bot.
    import os as _os
    probe = bytes(range(256)) + _os.urandom(997)
    assert b85_decode(b85_encode(probe), len(probe)) == probe, "base85 round-trip"
    assert len(b85_encode(b"\x00\x00\x00\x00")) == 5, "base85 should be 4->5"

    # int8 quantisation: scale must survive the round trip within one step
    try:
        import numpy as _np
        w = _np.array([-1.0, -0.5, 0.0, 0.25, 1.0], dtype=_np.float32)
        q, sc = quantize_int8(w)
        assert abs(float(q.max()) * sc - 1.0) < 1e-6, "quantise scale wrong"
        assert _np.abs(q.astype(_np.float32) * sc - w).max() <= sc, "quantise error > 1 step"
    except ImportError:
        pass

    # Transition builder: the RL bug that hides best. Play one fixed game and
    # assert the shape of what training will consume.
    class _FixedNet:
        def __call__(self, t):
            import torch as _t
            return _t.zeros((t.shape[0], N_ACT))
        def __enter__(self): return self
    try:
        import torch
        import numpy
        # importing is not enough: this build needs a working numpy bridge
        torch.from_numpy(numpy.zeros(1, dtype=numpy.float32))
    except Exception as exc:
        print(f"selfcheck OK (RL asserts skipped: {type(exc).__name__})")
        return 0
    trs, _ = selfplay_episodes(_FixedNet(), "cpu", 4, 1.0, 11)
    assert trs, "no transitions produced"
    nonterm = [t for t in trs if t[3] is not None]
    term = [t for t in trs if t[3] is None]
    # immediate reward is zero except on a mover's final decision
    assert all(t[2] == 0.0 for t in nonterm), "non-terminal transition carries reward"
    assert all(abs(t[2]) in (0.0, 1.0) for t in term), "bad terminal reward"
    assert any(abs(t[2]) == 1.0 for t in term), "no decisive game in sample"
    # pure self-play: the next stored state is always the opponent's -> sign -1
    assert all(t[5] == -1.0 for t in nonterm), "self-play bootstrap sign should negate"
    assert all(t[4] is not None and len(t[4]) == N_ACT for t in nonterm), \
        "non-terminal transition is missing its next-state legal mask"
    assert all(len(t[0]) == N_IN for t in trs), "state width mismatch"

    print("selfcheck OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selfcheck")
    sub.add_parser("bench")
    f = sub.add_parser("fuzz")
    f.add_argument("--other", required=True)
    f.add_argument("--games", type=int, default=1000)
    f.add_argument("--seed", type=int, default=1)
    f.add_argument("--level", type=int, default=2, choices=(1, 2))
    m = sub.add_parser("match")
    m.add_argument("--a", required=True, choices=POLICIES)
    m.add_argument("--b", required=True, choices=POLICIES)
    m.add_argument("--games", type=int, default=100)
    m.add_argument("--seed", type=int, default=1)
    m.add_argument("--level", type=int, default=2, choices=(1, 2))
    m.add_argument("--budget-ms", type=int, default=90)
    c = sub.add_parser("cg")
    c.add_argument("--policy", required=True, choices=POLICIES)
    c.add_argument("--level", type=int, required=True, choices=(1, 2))
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--budget-ms", type=int, default=90)
    bn = sub.add_parser("bench-net")
    bn.add_argument("--net", default="out/net.pt")
    bn.add_argument("--cpp", default=None)
    bn.add_argument("--vs", default="greedy", choices=POLICIES)
    bn.add_argument("--games", type=int, default=100)
    bn.add_argument("--budget-ms", type=int, default=40)
    bn.add_argument("--device", default="cpu")
    bn.add_argument("--seed", type=int, default=7)
    cn = sub.add_parser("check-net")
    cn.add_argument("--net", default="out/net.pt")
    cn.add_argument("--cpp", default="out/net_bot.cpp")
    cn.add_argument("--games", type=int, default=10)
    cn.add_argument("--seed", type=int, default=5)
    cn.add_argument("--min-agree", type=float, default=99.9)
    pk = sub.add_parser("pack")
    pk.add_argument("--net", default="out/net.pt")
    pk.add_argument("--out", default="out/net_bot.cpp")
    tr = sub.add_parser("train")
    tr.add_argument("--iters", type=int, default=200)
    tr.add_argument("--games-per-iter", type=int, default=256)
    tr.add_argument("--updates-per-iter", type=int, default=64)
    tr.add_argument("--batch", type=int, default=512)
    tr.add_argument("--buffer", type=int, default=400000)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--gamma", type=float, default=0.99)
    tr.add_argument("--eps-start", type=float, default=0.9)
    tr.add_argument("--eps-final", type=float, default=0.05)
    tr.add_argument("--target-every", type=int, default=5)
    tr.add_argument("--league-every", type=int, default=20)
    tr.add_argument("--league-size", type=int, default=5)
    tr.add_argument("--league-frac", type=float, default=0.3)
    tr.add_argument("--eval-every", type=int, default=10)
    tr.add_argument("--eval-games", type=int, default=100)
    tr.add_argument("--device", default="auto")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--out", default="out/net.pt")
    s = sub.add_parser("snapshot")
    s.add_argument("--arena", default="tic-tac-toe")
    s.add_argument("--pseudo", default="Napkin100k")
    s.add_argument("--out", default="out/ladder_snapshots.jsonl")
    args = ap.parse_args()
    if args.cmd == "snapshot":
        sys.exit(cmd_snapshot(args))
    if args.cmd == "bench-net":
        sys.exit(cmd_bench_net(args))
    if args.cmd == "check-net":
        sys.exit(cmd_check_net(args))
    if args.cmd == "pack":
        sys.exit(cmd_pack(args))
    if args.cmd == "train":
        sys.exit(cmd_train(args))
    if args.cmd == "selfcheck":
        sys.exit(cmd_selfcheck(args))
    if args.cmd == "bench":
        sys.exit(cmd_bench(args))
    if args.cmd == "fuzz":
        sys.exit(cmd_fuzz(args))
    if args.cmd == "match":
        sys.exit(cmd_match(args))
    if args.cmd == "cg":
        sys.exit(cmd_cg(args))


if __name__ == "__main__":
    main()
