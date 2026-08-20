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


class NetSearchPolicy:
    """H4: the trained net as a leaf evaluator inside negamax with alpha-beta.

    The net is unchanged and untrained further; the only added ingredient is
    lookahead. Leaf value of a position, from the side-to-move's view, is
    max_a Q(s,a) over legal a -- the net's own estimate of how good it is to be
    here. Terminal positions use the true result, so the search is exact near
    the end of the game and learned in the middle.
    """
    name = "netsearch"

    def __init__(self, net, device, depth=3, seed=0):
        self.net = net
        self.device = device
        self.depth = depth
        self.rng = random.Random(seed)
        self._cache = {}

    def _leaf_value(self, eng):
        import numpy as np
        import torch
        key = eng.get_state()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        x = np.asarray(encode_planes(eng, eng.current_player), dtype=np.float32)
        with torch.no_grad():
            q = self.net(torch.from_numpy(x[None]).to(self.device)).cpu().numpy()[0]
        v = max(q[action_index(r, c)] for (r, c) in eng.valid_actions())
        v = float(max(-1.0, min(1.0, v)))
        self._cache[key] = v
        return v

    def _negamax(self, eng, depth, alpha, beta, me):
        if eng.game_over:
            # from the perspective of the player to move at the parent call
            if eng.winner == -1:
                return 0.0
            return 1.0 if eng.winner == me else -1.0
        if depth == 0:
            v = self._leaf_value(eng)
            return v if eng.current_player == me else -v
        state = eng.get_state()
        best = -2.0
        for a in sorted(eng.valid_actions()):
            eng.play(*a)
            v = -self._negamax(eng, depth - 1, -beta, -alpha, 1 - me)
            eng.set_state(state)
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def act(self, eng):
        me = eng.current_player
        state = eng.get_state()
        best, bv = None, -2.0
        for a in sorted(eng.valid_actions()):
            eng.play(*a)
            v = -self._negamax(eng, self.depth - 1, -2.0, 2.0, 1 - me)
            eng.set_state(state)
            if v > bv:
                bv, best = v, a
        return best


def cmd_bench_search(args):
    """H4: same net, with lookahead, against a scripted opponent."""
    import torch
    ck = torch.load(args.net, map_location=args.device)
    net = build_net(args.device)
    net.load_state_dict(ck["state_dict"])
    net.eval()

    rng = random.Random(args.seed)
    w = l = d = 0
    for g in range(args.games):
        seat = g % 2
        eng = Engine(2)
        me = NetSearchPolicy(net, args.device, args.depth, seed=g)
        opp = POLICIES[args.vs](seed=rng.randrange(2**30), budget_ms=args.budget_ms)
        while not eng.game_over:
            mv = me.act(eng) if eng.current_player == seat else opp.act(eng)
            eng.play(*mv)
        if eng.winner == seat:
            w += 1
        elif eng.winner == -1:
            d += 1
        else:
            l += 1
    n = w + l + d
    p_hat = (w + 0.5 * d) / n
    z = 1.96
    den = 1 + z * z / n
    mid = (p_hat + z * z / (2 * n)) / den
    half = z * ((p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) ** 0.5) / den
    print(f"net+search(depth {args.depth}) vs {args.vs}: {w}W-{l}L-{d}D of {n} | "
          f"score {p_hat:.3f} [{mid-half:.3f}, {mid+half:.3f}]")
    return 0


# == GPU self-play: the whole game, batched =====================================
# The Python engine above is the verified reference, but it is also the reason a
# T4 sat idle: self-play spent its time in per-object Python, not in tensors.
# This is the same rules re-expressed as batched bitmask arithmetic, so tens of
# thousands of games advance per kernel launch and the accelerator does the work.
# It is checked against the reference engine move-for-move (`gpu-parity`).

def _tables(device):
    """Precomputed constants: cell->board, cell->bit, and 3-in-a-row lookup."""
    import torch
    c2b = torch.tensor([(r // 3) * 3 + c // 3 for r in range(9) for c in range(9)],
                       dtype=torch.long, device=device)
    c2i = torch.tensor([(r % 3) * 3 + (c % 3) for r in range(9) for c in range(9)],
                       dtype=torch.long, device=device)
    win = torch.zeros(512, dtype=torch.bool, device=device)
    for m in range(512):
        win[m] = any((m & w) == w for w in WIN_MASKS)
    return c2b, c2i, win


class TensorUTTT:
    """B independent Ultimate Tic-Tac-Toe games advanced in lockstep on GPU.

    Layout (all torch tensors, first dim = game):
      boards [B,2,9] int32   9-bit occupancy per player per small board
      owned  [B,2]   int32   9-bit mask of small boards won
      drawn  [B]     int32   9-bit mask of small boards full with no winner
      last   [B]     int64   previous cell 0..80, -1 before the first move
      side   [B]     int64   player to move
      cnt    [B,2]   int32   small boards won (the referee's running score)
      done   [B]     bool
      winner [B]     int64   0/1, or -1 for draw / still running
    """

    def __init__(self, batch, device):
        import torch
        self.B = batch
        self.device = device
        self.c2b, self.c2i, self.WIN = _tables(device)
        z = lambda *s, dt=torch.int32: torch.zeros(*s, dtype=dt, device=device)
        self.boards = z(batch, 2, 9)
        self.owned = z(batch, 2)
        self.drawn = z(batch)
        self.cnt = z(batch, 2)
        self.last = torch.full((batch,), -1, dtype=torch.long, device=device)
        self.side = torch.zeros(batch, dtype=torch.long, device=device)
        self.done = torch.zeros(batch, dtype=torch.bool, device=device)
        self.winner = torch.full((batch,), -1, dtype=torch.long, device=device)
        self.ar = torch.arange(batch, device=device)

    # -- observation ---------------------------------------------------------

    def decided(self):
        """[B,9] bool: board is owned by someone or drawn, so unplayable."""
        import torch
        d = self.owned[:, 0] | self.owned[:, 1] | self.drawn
        bits = torch.arange(9, device=self.device)
        return ((d.unsqueeze(1) >> bits) & 1).bool()

    def legal_mask(self):
        """[B,81] bool. Mirrors Engine.valid_actions exactly, including the
        'target board is decided -> play anywhere' rule."""
        import torch
        occ = self.boards[:, 0] | self.boards[:, 1]                  # [B,9]
        cell_occ = ((occ[:, self.c2b] >> self.c2i) & 1).bool()        # [B,81]
        dec = self.decided()                                          # [B,9]
        cell_board_dead = dec[:, self.c2b]                            # [B,81]

        tb = torch.where(self.last >= 0,
                         ((self.last // 9) % 3) * 3 + (self.last % 9) % 3,
                         torch.full_like(self.last, -1))              # [B]
        tb_ok = tb >= 0
        tb_dead = torch.zeros_like(tb_ok)
        tb_dead[tb_ok] = dec[self.ar[tb_ok], tb[tb_ok]]
        constrained = tb_ok & ~tb_dead                                # [B]

        in_target = self.c2b.unsqueeze(0) == tb.unsqueeze(1)          # [B,81]
        allowed = torch.where(constrained.unsqueeze(1), in_target,
                              torch.ones_like(in_target))
        legal = (~cell_occ) & (~cell_board_dead) & allowed
        return legal & ~self.done.unsqueeze(1)

    def encode(self):
        """[B,324] float32, matching encode_planes() for the side to move."""
        import torch
        me, opp = self.side, 1 - self.side
        bm = self.boards[self.ar, me]                                 # [B,9]
        bo = self.boards[self.ar, opp]
        mine = ((bm[:, self.c2b] >> self.c2i) & 1).float()            # [B,81]
        theirs = ((bo[:, self.c2b] >> self.c2i) & 1).float()
        legal = self.legal_mask().float()
        bits = torch.arange(9, device=self.device)
        om = ((self.owned[self.ar, me].unsqueeze(1) >> bits) & 1).float()
        oo = ((self.owned[self.ar, opp].unsqueeze(1) >> bits) & 1).float()
        odiff = (om - oo)[:, self.c2b]                                # [B,81]
        return torch.cat([mine, theirs, legal, odiff], dim=1)

    # -- transition ----------------------------------------------------------

    def step(self, cells):
        """Apply one move per live game. `cells` [B] long; ignored where done."""
        import torch
        live = ~self.done
        if not bool(live.any()):
            return
        b = self.c2b[cells]
        i = self.c2i[cells]
        s = self.side
        one = torch.ones_like(b, dtype=torch.int32)

        cur = self.boards[self.ar, s, b] | (one << i.to(torch.int32))
        self.boards[self.ar, s, b] = torch.where(live, cur,
                                                 self.boards[self.ar, s, b])

        won_small = self.WIN[cur.long()] & live
        self.owned[self.ar, s] |= (one << b.to(torch.int32)) * won_small
        self.cnt[self.ar, s] += won_small.to(torch.int32)

        both = self.boards[self.ar, 0, b] | self.boards[self.ar, 1, b]
        full = (both == 511) & live
        self.drawn |= (one << b.to(torch.int32)) * (full & ~won_small)

        master = self.WIN[self.owned[self.ar, s].long()] & won_small
        self.last = torch.where(live, cells, self.last)
        self.side = torch.where(live, 1 - s, self.side)

        # a completed master line ends it immediately, mover wins
        self.winner = torch.where(master, s, self.winner)
        self.done = self.done | master

        # otherwise: no legal move anywhere ends it, most boards won decides
        stuck = (~self.done) & (~self.legal_mask().any(dim=1))
        if bool(stuck.any()):
            c0, c1 = self.cnt[:, 0], self.cnt[:, 1]
            by_count = torch.where(c0 > c1, torch.zeros_like(self.winner),
                                   torch.where(c1 > c0,
                                               torch.ones_like(self.winner),
                                               torch.full_like(self.winner, -1)))
            self.winner = torch.where(stuck, by_count, self.winner)
            self.done = self.done | stuck

    def reset_done(self):
        """Restart finished games in place, so the batch never idles."""
        import torch
        d = self.done
        if not bool(d.any()):
            return int(0)
        n = int(d.sum())
        self.boards[d] = 0
        self.owned[d] = 0
        self.drawn[d] = 0
        self.cnt[d] = 0
        self.last[d] = -1
        self.side[d] = 0
        self.winner[d] = -1
        self.done[d] = False
        return n
    def clone_repeat(self, n):
        """A new batch with every game duplicated n times (contiguous blocks).
        Used to evaluate every candidate move of every game in one shot."""
        import torch
        c = TensorUTTT(self.B * n, self.device)
        rep = lambda x: x.repeat_interleave(n, dim=0)
        c.boards = rep(self.boards).clone()
        c.owned = rep(self.owned).clone()
        c.drawn = rep(self.drawn).clone()
        c.cnt = rep(self.cnt).clone()
        c.last = rep(self.last).clone()
        c.side = rep(self.side).clone()
        c.done = rep(self.done).clone()
        c.winner = rep(self.winner).clone()
        return c


def improved_policy(t, net, tau=1.0):
    """One-ply policy improvement, entirely on GPU.

    For every game and EVERY one of the 81 moves at once: play it in a cloned
    batch, then score the resulting position. Terminal children use the exact
    result; the rest use the value head. Q(s,a) is negated because the child is
    the opponent's turn. The softmax over those Q values is a strictly better
    policy than the network's raw prior, which is what makes it a useful
    training target -- the same reason AlphaZero trains on search output.

    Returns (Q [B,81], pi [B,81], legal [B,81]).
    """
    import torch
    B, dev = t.B, t.device
    legal = t.legal_mask()
    kids = t.clone_repeat(81)
    cells = torch.arange(81, device=dev).repeat(B)
    kids.step(cells)

    with torch.no_grad():
        _, v_child = net(kids.encode())
    v_child = v_child.view(B, 81)
    kdone = kids.done.view(B, 81)
    kwin = kids.winner.view(B, 81)
    me = t.side.unsqueeze(1).expand(B, 81)

    exact = torch.where(kwin == me, torch.ones_like(v_child),
                        torch.where(kwin < 0, torch.zeros_like(v_child),
                                    -torch.ones_like(v_child)))
    q = torch.where(kdone, exact, -v_child)          # our value after playing a
    q = q.masked_fill(~legal, -1e9)
    pi = torch.softmax(q / tau, dim=1)
    pi = pi * legal
    pi = pi / pi.sum(dim=1, keepdim=True).clamp_min(1e-9)
    return q, pi, legal


def _sym_perms(device):
    """The 8 dihedral symmetries of Ultimate Tic-Tac-Toe as cell permutations.

    A symmetry must act on the master board AND identically on every small
    board, or the "your move picks my board" rule breaks. Applying transform g
    to (board_row, board_col) and to (inner_row, inner_col) together is exactly
    such a symmetry, so a position and its image are the same game -- which
    makes this free training data: 8x the samples for no extra self-play.
    """
    import torch

    def t(g, r, c):
        if g & 4:
            r, c = c, r          # transpose
        for _ in range(g & 3):
            r, c = c, 2 - r      # rotate
        return r, c

    perms = []
    for g in range(8):
        p = [0] * 81
        for cell in range(81):
            R, C = cell // 9, cell % 9
            br, bc, ir, ic = R // 3, C // 3, R % 3, C % 3
            nbr, nbc = t(g, br, bc)
            nir, nic = t(g, ir, ic)
            p[cell] = (nbr * 3 + nir) * 9 + (nbc * 3 + nic)
        perms.append(p)
    return torch.tensor(perms, dtype=torch.long, device=device)


def augment(x, pi, perm):
    """Apply one symmetry to features [N,324] (4 planes of 81) and policy [N,81].
    The owned-difference plane is constant within a small board, so permuting
    cells transforms it correctly too."""
    n = x.shape[0]
    xp = x.view(n, 4, 81)[:, :, perm].reshape(n, 324)
    return xp, pi[:, perm]


def cmd_train_gpu(args):
    """Self-play on GPU with one-ply-improved policy targets (H6).

    Everything except the optimiser step is batched tensor work: the games, the
    legal-move logic, the candidate expansion and the evaluation.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    net = build_aznet(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    B = args.batch_games
    t = TensorUTTT(B, device)

    # Trajectory staging and the replay buffer live on GPU. Nothing crosses to
    # the host inside the step loop: writes are scatters, harvesting a finished
    # game is a boolean mask. (Per-step .cpu() calls were the previous ceiling --
    # they held throughput at ~8 games/s while the engine could do 15,000.)
    MAXP = 81
    st_x = torch.zeros(B, MAXP, N_IN, device=device)
    st_p = torch.zeros(B, MAXP, N_ACT, device=device)
    st_m = torch.zeros(B, MAXP, dtype=torch.long, device=device)
    plies = torch.zeros(B, dtype=torch.long, device=device)

    cap = args.buffer
    bf_x = torch.zeros(cap, N_IN, device=device)
    bf_p = torch.zeros(cap, N_ACT, device=device)
    bf_z = torch.zeros(cap, device=device)
    ptr = 0
    filled = 0
    t0 = time.time()
    games_done = 0
    ar = torch.arange(B, device=device)
    perms = _sym_perms(device)

    for it in range(1, args.iters + 1):
        for _ in range(args.steps_per_iter):
            x = t.encode()
            _, pi, legal = improved_policy(t, net, tau=args.tau)
            # Exploration belongs in the opening. Sampling every ply (plus a
            # uniform mix) makes outcomes nearly unpredictable, and then the
            # value head is being asked to regress noise -- which is exactly
            # what a value loss stuck near 0.88 looks like.
            u = legal.float()
            u = u / u.sum(dim=1, keepdim=True).clamp_min(1e-9)
            act_p = (1 - args.eps) * pi + args.eps * u
            mv_s = torch.multinomial(act_p.clamp_min(1e-12), 1).squeeze(1)
            mv_g = pi.argmax(dim=1)
            opening = plies < args.opening_plies
            mv = torch.where(opening, mv_s, mv_g)

            live = ~t.done
            slot = plies.clamp(max=MAXP - 1)
            st_x[ar, slot] = torch.where(live.unsqueeze(1), x, st_x[ar, slot])
            st_p[ar, slot] = torch.where(live.unsqueeze(1), pi, st_p[ar, slot])
            st_m[ar, slot] = torch.where(live, t.side, st_m[ar, slot])
            plies = plies + live.long()

            t.step(mv)

            fin = t.done
            if bool(fin.any()):
                # z per stored ply: +1 if that ply's mover won the game
                w = t.winner.unsqueeze(1).expand(B, MAXP)
                z = torch.where(w < 0, torch.zeros(B, MAXP, device=device),
                                torch.where(w == st_m, torch.ones(B, MAXP, device=device),
                                            -torch.ones(B, MAXP, device=device)))
                valid = (torch.arange(MAXP, device=device).unsqueeze(0)
                         < plies.unsqueeze(1)) & fin.unsqueeze(1)
                nsel = int(valid.sum())
                if nsel:
                    sx = st_x[valid]
                    sp = st_p[valid]
                    sz = z[valid]
                    idx = (ptr + torch.arange(nsel, device=device)) % cap
                    bf_x[idx] = sx
                    bf_p[idx] = sp
                    bf_z[idx] = sz
                    ptr = int((ptr + nsel) % cap)
                    filled = min(cap, filled + nsel)
                games_done += int(fin.sum())
                plies = torch.where(fin, torch.zeros_like(plies), plies)
                t.reset_done()

        if filled < args.batch:
            continue
        for _ in range(args.updates_per_iter):
            idx = torch.randint(0, filled, (args.batch,), device=device)
            xb, pb = bf_x[idx], bf_p[idx]
            if args.augment:
                g = int(torch.randint(0, 8, (1,)).item())
                xb, pb = augment(xb, pb, perms[g])
            logits, v = net(xb)
            loss_p = -(pb * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            loss_v = F.mse_loss(v, bf_z[idx])
            loss = loss_p + args.value_weight * loss_v
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

        if it % args.eval_every == 0 or it == args.iters:
            wr = evaluate_aznet_greedy(net, device, args.eval_games, args.seed + it)
            print(f"iter {it}/{args.iters} games {games_done} buf {filled} "
                  f"loss {float(loss):.4f} (p {float(loss_p):.4f} v {float(loss_v):.4f}) "
                  f"vs-greedy {wr:.3f} ({time.time()-t0:.0f}s)", flush=True)
        torch.save({"state_dict": net.state_dict(),
                    "shape": [N_IN, AZ_TRUNK1, AZ_TRUNK2, N_ACT]}, args.out)
    rate = games_done / max(1e-9, time.time() - t0)
    print(f"saved {args.out} after {games_done} self-play games "
          f"({rate:,.0f} games/s, {aznet_param_bytes()} weights)")
    return 0


def evaluate_aznet_greedy(net, device, games, seed):
    """Win rate vs `greedy`, both seats, using the net's 1-ply improved policy."""
    import torch
    rng = random.Random(seed)
    w = d = n = 0
    for g in range(games):
        seat = g % 2
        eng = Engine(2)
        opp = POLICIES["greedy"](seed=rng.randrange(2**30))
        while not eng.game_over:
            if eng.current_player == seat:
                t = TensorUTTT(1, device)
                t.boards[0, 0] = torch.tensor(eng.boards[0], dtype=torch.int32, device=device)
                t.boards[0, 1] = torch.tensor(eng.boards[1], dtype=torch.int32, device=device)
                t.owned[0, 0] = int(eng.owned[0]); t.owned[0, 1] = int(eng.owned[1])
                t.drawn[0] = int(eng.drawn)
                t.cnt[0, 0] = int(eng.scores[0] if eng.scores[0] < 10 else eng.scores[0])
                t.cnt[0, 1] = int(eng.scores[1] if eng.scores[1] < 10 else eng.scores[1])
                t.last[0] = -1 if eng.last is None else action_index(*eng.last)
                t.side[0] = eng.current_player
                _, pi, _ = improved_policy(t, net)
                mv = int(pi[0].argmax())
                a = index_action(mv)
                if a not in eng.valid_actions():
                    a = sorted(eng.valid_actions())[0]
                eng.play(*a)
            else:
                eng.play(*opp.act(eng))
        n += 1
        if eng.winner == seat:
            w += 1
        elif eng.winner == -1:
            d += 1
    return (w + 0.5 * d) / max(1, n)


def cmd_gpu_parity(args):
    """A new engine is worth nothing until it agrees with the verified one.

    Plays random games with the tensor engine and the reference Python engine in
    lockstep, comparing the legal-move SET every ply and the outcome every game.
    """
    import numpy as np
    import torch

    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    B = args.games
    t = TensorUTTT(B, device)
    refs = [Engine(2) for _ in range(B)]
    plies = 0
    while True:
        gl = t.legal_mask().cpu().numpy()
        alive = [k for k in range(B) if not refs[k].game_over]
        if not alive:
            break
        for k in alive:
            ref = {action_index(*a) for a in refs[k].valid_actions()}
            got = set(np.nonzero(gl[k])[0].tolist())
            if ref != got:
                print(f"DIVERGENCE game {k} ply {plies}\\n"
                      f"  ref-only {sorted(ref - got)}\\n  gpu-only {sorted(got - ref)}")
                return 1
            # encoder parity too: the net must see identical features
            if args.check_encode:
                enc_gpu = t.encode()[k].cpu().numpy()
                enc_ref = np.asarray(encode_planes(refs[k], refs[k].current_player),
                                     dtype=np.float32)
                if not np.array_equal(enc_gpu, enc_ref):
                    bad = int(np.abs(enc_gpu - enc_ref).argmax())
                    print(f"ENCODER DIVERGENCE game {k} ply {plies} at feature {bad}")
                    return 1
        choice = torch.zeros(B, dtype=torch.long, device=device)
        for k in alive:
            a = rng.choice(sorted(refs[k].valid_actions()))
            choice[k] = action_index(*a)
            refs[k].play(*a)
        t.step(choice)
        plies += 1
        for k in alive:
            if refs[k].game_over != bool(t.done[k]):
                print(f"DONE MISMATCH game {k} ply {plies}: "
                      f"ref {refs[k].game_over} gpu {bool(t.done[k])}")
                return 1
            if refs[k].game_over and refs[k].winner != int(t.winner[k]):
                print(f"WINNER MISMATCH game {k}: ref {refs[k].winner} "
                      f"gpu {int(t.winner[k])}")
                return 1
    print(f"GPU PARITY OK: {B} games, {plies} plies, legal sets + outcomes"
          f"{' + encoder' if args.check_encode else ''} identical to the "
          f"verified engine")
    return 0


# == H5: the AlphaZero loop ====================================================
# A policy+value network guides an MCTS; the search's visit distribution becomes
# the policy target and the game outcome becomes the value target, so each round
# of self-play trains on a policy STRONGER than the one that generated it. The
# net supplies every learned quantity; the search contains no hand-tuned
# evaluation whatsoever (leaves are the net's value head, terminals are exact).
#
# Sized to the same measured budget: trunk 324->128->96, policy head 96->81,
# value head 96->1 = 61,632 weights = ~77.0k base85 chars, leaving room for the
# MCTS harness inside 100,000 bytes.

AZ_TRUNK1, AZ_TRUNK2 = 128, 96


def build_aznet(device="cpu"):
    import torch
    import torch.nn as nn

    class _AZ(nn.Module):
        def __init__(self):
            super().__init__()
            self.t1 = nn.Linear(N_IN, AZ_TRUNK1)
            self.t2 = nn.Linear(AZ_TRUNK1, AZ_TRUNK2)
            self.ph = nn.Linear(AZ_TRUNK2, N_ACT)
            self.vh = nn.Linear(AZ_TRUNK2, 1)

        def forward(self, x):
            h = torch.relu(self.t1(x))
            h = torch.relu(self.t2(h))
            return self.ph(h), torch.tanh(self.vh(h)).squeeze(-1)

    return _AZ().to(device)


def aznet_param_bytes():
    return (N_IN * AZ_TRUNK1 + AZ_TRUNK1 * AZ_TRUNK2
            + AZ_TRUNK2 * N_ACT + AZ_TRUNK2)


class MCTS:
    """PUCT search over the verified engine, evaluated by the net.

    Batched across games: every game contributes at most one leaf per round, so
    leaf evaluation is a single batched forward. Values are always stored from
    the point of view of the player to move at that node.
    """

    def __init__(self, net, device, sims, c_puct=1.5, dirichlet=0.0, seed=0):
        self.net = net
        self.device = device
        self.sims = sims
        self.c_puct = c_puct
        self.dirichlet = dirichlet
        self.rng = random.Random(seed)

    def _evaluate(self, engines):
        """Batched net evaluation. Returns (priors over legal moves, values)."""
        import numpy as np
        import torch
        if not engines:
            return [], []
        x = np.stack([np.asarray(encode_planes(e, e.current_player), dtype=np.float32)
                      for e in engines])
        with torch.no_grad():
            logits, vals = self.net(torch.from_numpy(x).to(self.device))
            logits = logits.cpu().numpy()
            vals = vals.cpu().numpy()
        priors = []
        for i, e in enumerate(engines):
            legal = sorted(e.valid_actions())
            lg = np.array([logits[i][action_index(*a)] for a in legal], dtype=np.float64)
            lg -= lg.max()
            ex = np.exp(lg)
            priors.append(dict(zip(legal, ex / ex.sum())))
        return priors, list(vals)

    def run(self, engines):
        """Return one visit-count distribution per engine."""
        import numpy as np
        roots = [{"N": {}, "W": {}, "P": None, "visits": 0} for _ in engines]
        states = [e.get_state() for e in engines]

        pri, _ = self._evaluate(engines)
        for i, r in enumerate(roots):
            r["P"] = dict(pri[i])
            if self.dirichlet > 0:
                legal = list(r["P"])
                noise = np.random.default_rng(self.rng.randrange(2**31)).dirichlet(
                    [0.3] * len(legal))
                for k, a in enumerate(legal):
                    r["P"][a] = 0.75 * r["P"][a] + 0.25 * noise[k]
            for a in r["P"]:
                r["N"][a] = 0
                r["W"][a] = 0.0

        # Depth-1 PUCT with net-evaluated children: each simulation picks a root
        # action by PUCT, evaluates the resulting position with the net, and backs
        # the (negated) value up. Cheap, batched, and enough to improve the policy.
        for _ in range(self.sims):
            picks, children = [], []
            for i, e in enumerate(engines):
                r = roots[i]
                tot = max(1, r["visits"])
                best, bs = None, -1e18
                for a, p in r["P"].items():
                    n = r["N"][a]
                    q = (r["W"][a] / n) if n else 0.0
                    u = self.c_puct * p * (tot ** 0.5) / (1 + n)
                    if q + u > bs:
                        bs, best = q + u, a
                picks.append(best)
                e.set_state(states[i])
                e.play(*best)
                children.append(e)
            # terminal children need the true result, not the net
            live = [i for i, e in enumerate(children) if not e.game_over]
            pri2, vals = self._evaluate([children[i] for i in live])
            vmap = dict(zip(live, vals))
            for i, e in enumerate(children):
                if e.game_over:
                    # value from the CHILD's mover perspective; terminal -> outcome
                    v_child = 0.0 if e.winner == -1 else (
                        1.0 if e.winner == e.moves % 2 else -1.0)
                else:
                    v_child = float(vmap[i])
                a = picks[i]
                roots[i]["N"][a] += 1
                roots[i]["W"][a] += -v_child     # child's value is the opponent's
                roots[i]["visits"] += 1
        for i, e in enumerate(engines):
            e.set_state(states[i])
        out = []
        for r in roots:
            tot = sum(r["N"].values()) or 1
            out.append({a: n / tot for a, n in r["N"].items()})
        return out


def cmd_train_az(args):
    """Self-play with MCTS-improved targets (H5)."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    net = build_aznet(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    buf = []
    t0 = time.time()

    for it in range(1, args.iters + 1):
        mcts = MCTS(net, device, args.sims, dirichlet=0.25, seed=args.seed * 977 + it)
        engines = [Engine(2) for _ in range(args.batch_games)]
        histories = [[] for _ in engines]
        live = list(range(len(engines)))
        while live:
            pis = mcts.run([engines[i] for i in live])
            for k, i in enumerate(live):
                e = engines[i]
                pi = pis[k]
                x = np.asarray(encode_planes(e, e.current_player), dtype=np.float32)
                target = np.zeros(N_ACT, dtype=np.float32)
                for a, pr in pi.items():
                    target[action_index(*a)] = pr
                histories[i].append((x, target, e.current_player))
                moves = list(pi)
                probs = np.array([pi[a] for a in moves], dtype=np.float64)
                probs = probs / probs.sum()
                idx = np.random.default_rng(
                    mcts.rng.randrange(2**31)).choice(len(moves), p=probs)
                e.play(*moves[idx])
            live = [i for i in live if not engines[i].game_over]
        for i, e in enumerate(engines):
            w = e.winner
            for x, target, mover in histories[i]:
                z = 0.0 if w == -1 else (1.0 if w == mover else -1.0)
                buf.append((x, target, z))
        if len(buf) > args.buffer:
            buf = buf[-args.buffer:]

        for _ in range(args.updates_per_iter):
            b = random.sample(buf, min(args.batch, len(buf)))
            xb = torch.from_numpy(np.stack([q[0] for q in b])).to(device)
            pb = torch.from_numpy(np.stack([q[1] for q in b])).to(device)
            zb = torch.tensor([q[2] for q in b], device=device, dtype=torch.float32)
            logits, v = net(xb)
            logp = F.log_softmax(logits, dim=1)
            loss_p = -(pb * logp).sum(dim=1).mean()
            loss_v = F.mse_loss(v, zb)
            loss = loss_p + loss_v
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

        if it % args.eval_every == 0 or it == args.iters:
            wr = evaluate_aznet(net, device, args.sims, args.eval_games, args.seed + it)
            print(f"iter {it}/{args.iters} buf {len(buf)} loss {float(loss):.4f} "
                  f"(p {float(loss_p):.4f} v {float(loss_v):.4f}) "
                  f"vs-greedy {wr:.3f} ({time.time()-t0:.0f}s)", flush=True)
        torch.save({"state_dict": net.state_dict(),
                    "shape": [N_IN, AZ_TRUNK1, AZ_TRUNK2, N_ACT]}, args.out)
    print(f"saved {args.out} ({aznet_param_bytes()} weights, "
          f"{aznet_param_bytes()*1.25/1000:.1f}k chars as int8+base85)")
    return 0


def az_policy(net, device, sims, seed=0):
    mcts = MCTS(net, device, sims, seed=seed)

    def policy(eng):
        pi = mcts.run([eng])[0]
        return max(pi, key=pi.get)
    return policy


def evaluate_aznet(net, device, sims, games, seed):
    rng = random.Random(seed)
    pol = az_policy(net, device, sims, seed)
    w = d = n = 0
    for g in range(games):
        seat = g % 2
        eng = Engine(2)
        opp = POLICIES["greedy"](seed=rng.randrange(2**30))
        while not eng.game_over:
            eng.play(*(pol(eng) if eng.current_player == seat else opp.act(eng)))
        n += 1
        if eng.winner == seat:
            w += 1
        elif eng.winner == -1:
            d += 1
    return (w + 0.5 * d) / max(1, n)


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

NET_SEARCH_CPP_TEMPLATE = r"""/* napkin-100k: a self-play-trained net, weights and all, in one file,
 * used as the evaluation function inside a negamax search.
 * Every learned quantity is the net's; the search adds lookahead and exact
 * terminal results, and contains no hand-written position evaluation.
 * Disclosed bot - github.com/arose26/napkin-100k, account Napkin100k.
 * Net: %(shape)s, int8 weights decoded from base85 below. */
#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline")
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <string>

static const int N_IN = %(n_in)d, N_H1 = %(n_h1)d, N_H2 = %(n_h2)d, N_ACT = %(n_act)d;
static const float S1 = %(s1).9gf, S2 = %(s2).9gf, S3 = %(s3).9gf;
static const char* W85 =
%(w85)s;
static const float B1[] = {%(b1)s};
static const float B2[] = {%(b2)s};
static const float B3[] = {%(b3)s};
static int8_t W[%(nw)d];

static void decodeWeights() {
    static const char* A = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                           "abcdefghijklmnopqrstuvwxyz!#$%%&()*+-;<=>?@^_`{|}~";
    int inv[256]; for (int i = 0; i < 256; i++) inv[i] = -1;
    for (int i = 0; i < 85; i++) inv[(unsigned char)A[i]] = i;
    size_t n = strlen(W85); size_t out = 0;
    for (size_t i = 0; i + 4 < n; i += 5) {
        uint32_t v = 0;
        for (int k = 0; k < 5; k++) v = v * 85u + (uint32_t)inv[(unsigned char)W85[i+k]];
        for (int k = 3; k >= 0; k--)
            if (out + (size_t)k < (size_t)%(nw)d) W[out + k] = (int8_t)((v >> (8 * (3 - k))) & 0xFF);
        out += 4;
    }
}

static float h1_[N_H1], h2_[N_H2], qout[N_ACT];
static void forward(const float* x) {
    const int8_t* w = W;
    for (int j = 0; j < N_H1; j++) {
        float a = 0.f;
        for (int i = 0; i < N_IN; i++) a += x[i] * (float)w[j * N_IN + i];
        a = a * S1 + B1[j]; h1_[j] = a > 0.f ? a : 0.f;
    }
    w += N_IN * N_H1;
    for (int j = 0; j < N_H2; j++) {
        float a = 0.f;
        for (int i = 0; i < N_H1; i++) a += h1_[i] * (float)w[j * N_H1 + i];
        a = a * S2 + B2[j]; h2_[j] = a > 0.f ? a : 0.f;
    }
    w += N_H1 * N_H2;
    for (int j = 0; j < N_ACT; j++) {
        float a = 0.f;
        for (int i = 0; i < N_H2; i++) a += h2_[i] * (float)w[j * N_H2 + i];
        qout[j] = a * S3 + B3[j];
    }
}

/* ---- game (mirrors the verified Python engine) ---- */
static const uint16_t WINM[8] = {0700,070,07,0444,0222,0111,0421,0124};
static bool winsM(uint16_t m){for(int i=0;i<8;i++) if((m&WINM[i])==WINM[i]) return true; return false;}
struct Pos { uint16_t b[2][9], own[2], drawn; int last, side, cnt[2]; };
struct Undo { uint16_t o0,o1,dr; int last,c0,c1; };

static inline bool decided(const Pos& p,int b){return ((p.own[0]|p.own[1]|p.drawn)>>b)&1;}
static int legal(const Pos& p,int* out){
    int n=0, ab=-1;
    if(p.last>=0){ int t=((p.last/9)%%3)*3+(p.last%%9)%%3; if(!decided(p,t)) ab=t; }
    for(int b=0;b<9;b++){
        if(ab>=0&&b!=ab) continue;
        if(decided(p,b)) continue;
        uint16_t occ=(uint16_t)(p.b[0][b]|p.b[1][b]);
        int br=(b/3)*3, bc=(b%%3)*3;
        for(int i=0;i<9;i++) if(!((occ>>i)&1)) out[n++]=(br+i/3)*9+bc+i%%3;
    }
    return n;
}
static inline bool mk(Pos& p,int cell,Undo& u){
    u.o0=p.own[0];u.o1=p.own[1];u.dr=p.drawn;u.last=p.last;u.c0=p.cnt[0];u.c1=p.cnt[1];
    int r=cell/9,c=cell%%9,b=(r/3)*3+c/3,i=(r%%3)*3+(c%%3),s=p.side;
    p.b[s][b]|=(uint16_t)(1<<i);
    bool mw=false;
    if(winsM(p.b[s][b])){p.own[s]|=(uint16_t)(1<<b);p.cnt[s]++;if(winsM(p.own[s]))mw=true;}
    else if((p.b[0][b]|p.b[1][b])==0777) p.drawn|=(uint16_t)(1<<b);
    p.last=cell;p.side=1-s;return mw;
}
static inline void unmk(Pos& p,int cell,const Undo& u){
    int r=cell/9,c=cell%%9,b=(r/3)*3+c/3,i=(r%%3)*3+(c%%3);
    p.side=1-p.side;p.b[p.side][b]&=(uint16_t)~(1<<i);
    p.own[0]=u.o0;p.own[1]=u.o1;p.drawn=u.dr;p.last=u.last;p.cnt[0]=u.c0;p.cnt[1]=u.c1;
}

/* encode_planes(): [mine | theirs | legal | owned_diff] for the side to move */
static float feat[4*81];
static void encode(const Pos& p,const int* mv,int n){
    memset(feat,0,sizeof(feat));
    int me=p.side, opp=1-me;
    for(int r=0;r<9;r++)for(int c=0;c<9;c++){
        int b=(r/3)*3+c/3,i=(r%%3)*3+(c%%3),idx=r*9+c;
        feat[idx]=(float)((p.b[me][b]>>i)&1);
        feat[81+idx]=(float)((p.b[opp][b]>>i)&1);
        feat[243+idx]=(float)(((p.own[me]>>b)&1)-((p.own[opp]>>b)&1));
    }
    for(int k=0;k<n;k++) feat[162+mv[k]]=1.f;
}

/* leaf value from the side-to-move's view: the net's own best Q here */
static float leafValue(const Pos& p,const int* mv,int n){
    encode(p,mv,n); forward(feat);
    float best=-1e30f;
    for(int k=0;k<n;k++) if(qout[mv[k]]>best) best=qout[mv[k]];
    if(best>1.f) best=1.f; if(best<-1.f) best=-1.f;
    return best;
}

static std::chrono::steady_clock::time_point deadline;
static bool timeUp=false; static long evals=0;
static inline bool tick(){ if((++evals&63)==0 && std::chrono::steady_clock::now()>deadline) timeUp=true; return timeUp; }

static float negamax(Pos& p,int depth,float alpha,float beta){
    int mv[81]; int n=legal(p,mv);
    if(n==0){ int d=p.cnt[p.side]-p.cnt[1-p.side]; return d>0?1.f:(d<0?-1.f:0.f); }
    if(depth==0||tick()) return leafValue(p,mv,n);
    float best=-2.f;
    for(int k=0;k<n;k++){
        Undo u; bool mw=mk(p,mv[k],u);
        float v = mw ? 1.f : -negamax(p,depth-1,-beta,-alpha);
        unmk(p,mv[k],u);
        if(v>best) best=v;
        if(best>alpha) alpha=best;
        if(alpha>=beta) break;
        if(timeUp) break;
    }
    return best;
}

int main(){
    decodeWeights();
    Pos p; memset(&p,0,sizeof(p)); p.last=-1; p.side=0;
    bool first=true; int me=-1;
    while(true){
        int orow,ocol;
        if(scanf("%%d%%d",&orow,&ocol)!=2) return 0;
        if(orow<-1||orow>8||ocol<-1||ocol>8) return 1;
        if(me<0) me=(orow==-1)?0:1;
        if(orow>=0&&ocol>=0){ Undo u; mk(p,orow*9+ocol,u); }
        int n; if(scanf("%%d",&n)!=1||n<1||n>81) return 1;
        int mv[81];
        for(int i=0;i<n;i++){int r,c; if(scanf("%%d%%d",&r,&c)!=2) return 1;
            if(r<0||r>8||c<0||c>8) return 1; mv[i]=r*9+c;}
        /* venue parity: our own move generator must agree with the referee's
         * list every turn. If it ever does not, our internal position has
         * drifted and every search below is about the wrong game. */
        { int own[81]; int m=legal(p,own);
          bool same = (m==n);
          if(same) for(int i=0;i<m && same;i++){ bool f=false;
              for(int j=0;j<n;j++) if(own[i]==mv[j]){f=true;break;} same=f; }
          if(!same) fprintf(stderr,"PARITY MISMATCH ours=%%d ref=%%d\n",m,n); }
        int budget = first ? 900 : 85; first=false;
        deadline = std::chrono::steady_clock::now()+std::chrono::milliseconds(budget);
        timeUp=false; evals=0;
        /* Fallback must never be "the first legal move": under CPU contention the
         * clock can expire inside depth 1 and leave the choice uninitialised.
         * Start from the net's own preference, and let depth 1 always finish -
         * it is ~n leaf evaluations and it is what guarantees a forced win is
         * seen at all. Only depth >= 2 may be abandoned on time. */
        /* An immediately winning move is exact knowledge, not something worth
         * spending a time-budgeted search on. Check it directly so this case can
         * never depend on the clock. A win is either a master line (mk reports it)
         * or exhaustion with more small boards. */
        int winNow=-1;
        { for(int k=0;k<n;k++){
              Undo u; bool mw=mk(p,mv[k],u);
              bool won=mw;
              if(!won){ int tmp[81]; if(legal(p,tmp)==0)
                        won = p.cnt[1-p.side] > p.cnt[p.side]; }
              unmk(p,mv[k],u);
              if(won){ winNow=mv[k]; break; } } }
        if(winNow>=0){
            Undo u2; mk(p,winNow,u2);
            printf("%%d %%d\n",winNow/9,winNow%%9); fflush(stdout);
            continue;
        }

        encode(p,mv,n); forward(feat);
        int best=mv[0]; { float bq=-1e30f;
            for(int k=0;k<n;k++) if(qout[mv[k]]>bq){bq=qout[mv[k]];best=mv[k];} }
        int reached=0;
        for(int depth=1; depth<=12; depth++){
            float bv=-2.f; int bm=best;
            bool aborted=false;
            for(int k=0;k<n;k++){
                Undo u; bool mw=mk(p,mv[k],u);
                float v = mw ? 1.f : -negamax(p,depth-1,-2.f,2.f);
                unmk(p,mv[k],u);
                if(timeUp && depth>1){ aborted=true; break; }
                if(v>bv){bv=v;bm=mv[k];}
            }
            if(aborted) break;
            best=bm; reached=depth;
            if(bv>=1.f) break;
            if(timeUp) break;
        }
        fprintf(stderr,"depth=%%d evals=%%ld\n",reached,evals);
        { Undo u; mk(p,best,u); }
        printf("%%d %%d\n",best/9,best%%9); fflush(stdout);
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


AZ_SEARCH_CPP_TEMPLATE = r"""/* napkin-100k: a GPU-self-play-trained policy+value net, weights and all, in one
 * file, driving a negamax search. The value head scores leaves, the policy head
 * orders moves, terminals are exact. No hand-written position evaluation.
 * Disclosed bot - github.com/arose26/napkin-100k, account Napkin100k.
 * Trunk %(shape)s, int8 weights decoded from base85 below. */
#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline")
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <string>
#include <cmath>
#include <algorithm>

static const int N_IN=%(n_in)d, T1=%(t1)d, T2=%(t2)d, N_ACT=%(n_act)d;
static const float S1=%(s1).9gf, S2=%(s2).9gf, SP=%(sp).9gf, SV=%(sv).9gf;
static const char* W85 =
%(w85)s;
static const float B1[]={%(b1)s};
static const float B2[]={%(b2)s};
static const float BP[]={%(bp)s};
static const float BV=%(bv).9gf;
static int8_t W[%(nw)d];

static void decodeWeights(){
    static const char* A="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                         "abcdefghijklmnopqrstuvwxyz!#$%%&()*+-;<=>?@^_`{|}~";
    int inv[256]; for(int i=0;i<256;i++) inv[i]=-1;
    for(int i=0;i<85;i++) inv[(unsigned char)A[i]]=i;
    size_t n=strlen(W85), out=0;
    for(size_t i=0;i+4<n;i+=5){
        uint32_t v=0;
        for(int k=0;k<5;k++) v=v*85u+(uint32_t)inv[(unsigned char)W85[i+k]];
        for(int k=3;k>=0;k--)
            if(out+(size_t)k<(size_t)%(nw)d) W[out+k]=(int8_t)((v>>(8*(3-k)))&0xFF);
        out+=4;
    }
}

static float h1_[T1], h2_[T2], pol[N_ACT], val;
static void forward(const float* x){
    const int8_t* w=W;
    for(int j=0;j<T1;j++){ float a=0.f;
        for(int i=0;i<N_IN;i++) a+=x[i]*(float)w[j*N_IN+i];
        a=a*S1+B1[j]; h1_[j]=a>0.f?a:0.f; }
    w+=N_IN*T1;
    for(int j=0;j<T2;j++){ float a=0.f;
        for(int i=0;i<T1;i++) a+=h1_[i]*(float)w[j*T1+i];
        a=a*S2+B2[j]; h2_[j]=a>0.f?a:0.f; }
    w+=T1*T2;
    for(int j=0;j<N_ACT;j++){ float a=0.f;
        for(int i=0;i<T2;i++) a+=h2_[i]*(float)w[j*T2+i];
        pol[j]=a*SP+BP[j]; }
    w+=T2*N_ACT;
    { float a=0.f; for(int i=0;i<T2;i++) a+=h2_[i]*(float)w[i];
      a=a*SV+BV; val=tanhf(a); }
}

static const uint16_t WINM[8]={0700,070,07,0444,0222,0111,0421,0124};
static bool winsM(uint16_t m){for(int i=0;i<8;i++) if((m&WINM[i])==WINM[i]) return true; return false;}
struct Pos{ uint16_t b[2][9],own[2],drawn; int last,side,cnt[2]; };
struct Undo{ uint16_t o0,o1,dr; int last,c0,c1; };
static inline bool decided(const Pos&p,int b){return ((p.own[0]|p.own[1]|p.drawn)>>b)&1;}
static int legal(const Pos&p,int* out){
    int n=0,ab=-1;
    if(p.last>=0){int t=((p.last/9)%%3)*3+(p.last%%9)%%3; if(!decided(p,t)) ab=t;}
    for(int b=0;b<9;b++){
        if(ab>=0&&b!=ab) continue;
        if(decided(p,b)) continue;
        uint16_t occ=(uint16_t)(p.b[0][b]|p.b[1][b]);
        int br=(b/3)*3,bc=(b%%3)*3;
        for(int i=0;i<9;i++) if(!((occ>>i)&1)) out[n++]=(br+i/3)*9+bc+i%%3;
    }
    return n;
}
static inline bool mk(Pos&p,int cell,Undo&u){
    u.o0=p.own[0];u.o1=p.own[1];u.dr=p.drawn;u.last=p.last;u.c0=p.cnt[0];u.c1=p.cnt[1];
    int r=cell/9,c=cell%%9,b=(r/3)*3+c/3,i=(r%%3)*3+(c%%3),s=p.side;
    p.b[s][b]|=(uint16_t)(1<<i);
    bool mw=false;
    if(winsM(p.b[s][b])){p.own[s]|=(uint16_t)(1<<b);p.cnt[s]++;if(winsM(p.own[s]))mw=true;}
    else if((p.b[0][b]|p.b[1][b])==0777) p.drawn|=(uint16_t)(1<<b);
    p.last=cell;p.side=1-s;return mw;
}
static inline void unmk(Pos&p,int cell,const Undo&u){
    int r=cell/9,c=cell%%9,b=(r/3)*3+c/3,i=(r%%3)*3+(c%%3);
    p.side=1-p.side;p.b[p.side][b]&=(uint16_t)~(1<<i);
    p.own[0]=u.o0;p.own[1]=u.o1;p.drawn=u.dr;p.last=u.last;p.cnt[0]=u.c0;p.cnt[1]=u.c1;
}
static float feat[4*81];
static void encode(const Pos&p,const int* mv,int n){
    memset(feat,0,sizeof(feat));
    int me=p.side,opp=1-me;
    for(int r=0;r<9;r++)for(int c=0;c<9;c++){
        int b=(r/3)*3+c/3,i=(r%%3)*3+(c%%3),idx=r*9+c;
        feat[idx]=(float)((p.b[me][b]>>i)&1);
        feat[81+idx]=(float)((p.b[opp][b]>>i)&1);
        feat[243+idx]=(float)(((p.own[me]>>b)&1)-((p.own[opp]>>b)&1));
    }
    for(int k=0;k<n;k++) feat[162+mv[k]]=1.f;
}

static std::chrono::steady_clock::time_point deadline;
static bool timeUp=false; static long evals=0;
static inline bool tick(){ if((++evals&63)==0&&std::chrono::steady_clock::now()>deadline) timeUp=true; return timeUp; }

/* leaf: the value head, from the side-to-move's view */
static float negamax(Pos&p,int depth,float alpha,float beta){
    int mv[81]; int n=legal(p,mv);
    if(n==0){ int d=p.cnt[p.side]-p.cnt[1-p.side]; return d>0?1.f:(d<0?-1.f:0.f); }
    if(depth==0||tick()){ encode(p,mv,n); forward(feat); return val; }
    /* order by the policy head: better moves first prunes far more */
    encode(p,mv,n); forward(feat);
    float pr[81]; for(int k=0;k<n;k++) pr[k]=pol[mv[k]];
    for(int k=1;k<n;k++){ int c=mv[k]; float q=pr[k]; int j=k-1;
        while(j>=0&&pr[j]<q){mv[j+1]=mv[j];pr[j+1]=pr[j];j--;} mv[j+1]=c;pr[j+1]=q; }
    float best=-2.f;
    for(int k=0;k<n;k++){
        Undo u; bool mw=mk(p,mv[k],u);
        float v = mw ? 1.f : -negamax(p,depth-1,-beta,-alpha);
        unmk(p,mv[k],u);
        if(v>best) best=v;
        if(best>alpha) alpha=best;
        if(alpha>=beta) break;
        if(timeUp) break;
    }
    return best;
}

int main(){
    decodeWeights();
    Pos p; memset(&p,0,sizeof(p)); p.last=-1; p.side=0;
    bool first=true; int me=-1;
    while(true){
        int orow,ocol;
        if(scanf("%%d%%d",&orow,&ocol)!=2) return 0;
        if(orow<-1||orow>8||ocol<-1||ocol>8) return 1;
        if(me<0) me=(orow==-1)?0:1;
        if(orow>=0&&ocol>=0){ Undo u; mk(p,orow*9+ocol,u); }
        int n; if(scanf("%%d",&n)!=1||n<1||n>81) return 1;
        int mv[81];
        for(int i=0;i<n;i++){int r,c; if(scanf("%%d%%d",&r,&c)!=2) return 1;
            if(r<0||r>8||c<0||c>8) return 1; mv[i]=r*9+c;}

        /* an immediate win is exact knowledge; never let the clock decide it */
        int winNow=-1;
        for(int k=0;k<n;k++){
            Undo u; bool mw=mk(p,mv[k],u);
            bool won=mw;
            if(!won){ int tmp[81]; if(legal(p,tmp)==0) won=p.cnt[1-p.side]>p.cnt[p.side]; }
            unmk(p,mv[k],u);
            if(won){ winNow=mv[k]; break; }
        }
        if(winNow>=0){ Undo u; mk(p,winNow,u);
            printf("%%d %%d\n",winNow/9,winNow%%9); fflush(stdout); continue; }

        int budget = first?900:85; first=false;
        deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(budget);
        timeUp=false; evals=0;
        encode(p,mv,n); forward(feat);
        int best=mv[0]; { float bp=-1e30f;
            for(int k=0;k<n;k++) if(pol[mv[k]]>bp){bp=pol[mv[k]];best=mv[k];} }
        int reached=0;
        for(int depth=1;depth<=12;depth++){
            float bv=-2.f; int bm=best; bool aborted=false;
            for(int k=0;k<n;k++){
                Undo u; bool mw=mk(p,mv[k],u);
                float v = mw?1.f:-negamax(p,depth-1,-2.f,2.f);
                unmk(p,mv[k],u);
                if(timeUp&&depth>1){aborted=true;break;}
                if(v>bv){bv=v;bm=mv[k];}
            }
            if(aborted) break;
            best=bm; reached=depth;
            if(bv>=1.f) break;
            if(timeUp) break;
        }
        fprintf(stderr,"depth=%%d evals=%%ld\n",reached,evals);
        { Undo u; mk(p,best,u); }
        printf("%%d %%d\n",best/9,best%%9); fflush(stdout);
    }
}
"""


def cmd_pack_az(args):
    """Pack the GPU-trained policy+value net into one searching C++ source."""
    import numpy as np
    import torch

    ck = torch.load(args.net, map_location="cpu")
    sd = ck["state_dict"]
    n_in, t1, t2, n_act = ck["shape"]
    w1, b1 = sd["t1.weight"].numpy(), sd["t1.bias"].numpy()
    w2, b2 = sd["t2.weight"].numpy(), sd["t2.bias"].numpy()
    wp, bp = sd["ph.weight"].numpy(), sd["ph.bias"].numpy()
    wv, bv = sd["vh.weight"].numpy(), sd["vh.bias"].numpy()
    q1, s1 = quantize_int8(w1)
    q2, s2 = quantize_int8(w2)
    qp, sp = quantize_int8(wp)
    qv, sv = quantize_int8(wv)
    blob = q1.tobytes() + q2.tobytes() + qp.tobytes() + qv.tobytes()
    txt = b85_encode(blob)
    chunks = [txt[i:i + 500] for i in range(0, len(txt), 500)]
    w85 = "\n".join(f'  "{c}"' for c in chunks)
    fmt = lambda a: ",".join(f"{float(v):.6g}f" for v in np.asarray(a).ravel())
    src = AZ_SEARCH_CPP_TEMPLATE % {
        "shape": f"{n_in}-{t1}-{t2}", "n_in": n_in, "t1": t1, "t2": t2,
        "n_act": n_act, "s1": s1, "s2": s2, "sp": sp, "sv": sv,
        "w85": w85, "nw": len(blob), "b1": fmt(b1), "b2": fmt(b2),
        "bp": fmt(bp), "bv": float(bv[0]),
    }
    with open(args.out, "w") as f:
        f.write(src)
    n = len(src.encode("utf-8"))
    print(f"packed {args.net} -> {args.out}: {n} UTF-8 bytes "
          f"({100000 - n} under the cap), {len(blob)} int8 weights")
    return 0 if n <= 100000 else 1


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
    # Longer literals mean fewer quote+newline bytes: at 100 chars/line the
    # wrapping alone cost ~4.3KB of the budget, at 500 it costs ~0.9KB.
    chunks = [txt[i:i + 500] for i in range(0, len(txt), 500)]
    w85 = "\n".join(f'  "{c}"' for c in chunks)
    fmt = lambda a: ",".join(f"{float(v):.6g}f" for v in a)
    template = NET_SEARCH_CPP_TEMPLATE if args.search else NET_CPP_TEMPLATE
    src = template % {
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

    # Independent int8 reference: the same arithmetic the emitted C++ performs,
    # sharing none of its code. The C++ must match THIS (that is the packer being
    # correct); its drift from fp32 is quantisation, which we measure, not a bug.
    sd = ck["state_dict"]
    qw, sc = [], []
    for k in ("0.weight", "2.weight", "4.weight"):
        q, s_ = quantize_int8(sd[k].numpy())
        qw.append(q.astype(np.float32))
        sc.append(s_)
    bs = [sd[k].numpy() for k in ("0.bias", "2.bias", "4.bias")]

    def int8_ref(x):
        h = np.maximum(qw[0] @ x * sc[0] + bs[0], 0.0)
        h = np.maximum(qw[1] @ h * sc[1] + bs[1], 0.0)
        return qw[2] @ h * sc[2] + bs[2]

    tmp = tempfile.mkdtemp()
    exe = os.path.join(tmp, "bot")
    subprocess.run(["g++", "-O2", "-o", exe, args.cpp], check=True)

    rng = random.Random(args.seed)
    agree = total = 0            # C++ vs the int8 reference  -> packer correctness
    agree_fp = 0                 # int8 reference vs fp32     -> quantisation drift
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
                qi = int8_ref(x)
                torch_mv = max(va, key=lambda a: q[action_index(*a)])
                ref_mv = max(va, key=lambda a: qi[action_index(*a)])
                total += 1
                if cpp_mv == ref_mv:
                    agree += 1
                elif len(disagreements) < 5:
                    gap = abs(qi[action_index(*cpp_mv)] - qi[action_index(*ref_mv)])
                    disagreements.append((cpp_mv, ref_mv, gap))
                if ref_mv == torch_mv:
                    agree_fp += 1
                eng.play(*cpp_mv)
        finally:
            proc.kill()
            proc.wait()
    shutil.rmtree(tmp, ignore_errors=True)
    pct = 100.0 * agree / max(1, total)
    pct_fp = 100.0 * agree_fp / max(1, total)
    print(f"check-net: C++ vs int8 reference {agree}/{total} ({pct:.2f}%)  "
          f"<- packer correctness, gated at {args.min_agree}%")
    print(f"           int8 vs fp32          {agree_fp}/{total} ({pct_fp:.2f}%)  "
          f"<- quantisation drift, measured not gated")
    for c, t, g in disagreements:
        print(f"  cpp {c} vs ref {t}, |dQ| = {g:.5f}")
    return 0 if pct >= args.min_agree else 1


def cmd_check_bot(args):
    """Correctness gate for any packed C++ bot, searching or not.

    `check-net` compares against the net's raw argmax and is therefore only
    meaningful for the argmax bot -- a searching bot is *supposed* to disagree
    with it. What must hold for either is: every move legal, and every forced
    win taken (the search knows terminals exactly, so declining one means the
    value signs are wrong)."""
    import shutil
    import subprocess
    import tempfile
    import os

    tmp = tempfile.mkdtemp()
    exe = os.path.join(tmp, "bot")
    subprocess.run(["g++", "-O2", "-o", exe, args.cpp], check=True)
    rng = random.Random(args.seed)
    offered = taken = moves = 0
    for g in range(args.games):
        seat = g % 2
        eng = Engine(2)
        proc = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, bufsize=1)
        try:
            while not eng.game_over:
                if eng.current_player != seat:
                    eng.play(*rng.choice(sorted(eng.valid_actions())))
                    continue
                va = sorted(eng.valid_actions())
                st = eng.get_state()
                wins = []
                for a in va:
                    eng.play(*a)
                    if eng.game_over and eng.winner == seat:
                        wins.append(a)
                    eng.set_state(st)
                last = eng.last if eng.last is not None else (-1, -1)
                proc.stdin.write(f"{last[0]} {last[1]}\n{len(va)}\n")
                for a in va:
                    proc.stdin.write(f"{a[0]} {a[1]}\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
                if not line:
                    raise AssertionError("bot died mid-game")
                mv = tuple(int(v) for v in line.split())
                assert mv in eng.valid_actions(), f"ILLEGAL move {mv}; legal={va}"
                moves += 1
                if wins:
                    offered += 1
                    if mv in wins:
                        taken += 1
                eng.play(*mv)
        finally:
            proc.kill()
            proc.wait()
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"check-bot {args.cpp}: {moves} moves all legal, "
          f"forced wins taken {taken}/{offered}")
    return 0 if offered == taken else 1


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

    # Symmetry augmentation: replaying a game through a symmetry must produce
    # exactly the permuted position. If it does not, augmentation is feeding the
    # net mislabelled data -- silently, and forever.
    _perms = _sym_perms("cpu")
    for _g in range(8):
        _p = _perms[_g].tolist()
        assert sorted(_p) == list(range(81)), f"symmetry {_g} is not a bijection"
        _rng = random.Random(100 + _g)
        _a, _b = Engine(2), Engine(2)
        for _ in range(14):
            if _a.game_over or _b.game_over:
                break
            _mv = _rng.choice(sorted(_a.valid_actions()))
            _a.play(*_mv)
            _im = _p[action_index(*_mv)]
            _b.play(*index_action(_im))
        _la = {_p[action_index(*x)] for x in _a.valid_actions()}
        _lb = {action_index(*x) for x in _b.valid_actions()}
        assert _la == _lb, (f"symmetry {_g} does not preserve legal moves: "
                            f"{sorted(_la ^ _lb)}")

    # Test `augment` itself, not just the permutation table. Tensor indexing
    # x[:, :, perm] GATHERS, so it applies perm's INVERSE -- still a symmetry
    # (the group is closed under inverse) and still consistent between features
    # and policy, but the only way to be sure is to compare against a game
    # actually replayed through that map.
    import numpy as _np
    import torch as _t
    for _g in range(8):
        _p = _perms[_g].tolist()
        _inv = [0] * 81
        for _i, _v in enumerate(_p):
            _inv[_v] = _i
        _rng = random.Random(200 + _g)
        _a, _b = Engine(2), Engine(2)
        for _ in range(12):
            if _a.game_over or _b.game_over:
                break
            _mv = _rng.choice(sorted(_a.valid_actions()))
            _a.play(*_mv)
            _b.play(*index_action(_inv[action_index(*_mv)]))
        _xa = _t.tensor([encode_planes(_a, _a.current_player)], dtype=_t.float32)
        _pa = _t.zeros(1, N_ACT)
        for _act in _a.valid_actions():
            _pa[0, action_index(*_act)] = 1.0
        _xg, _pg = augment(_xa, _pa, _perms[_g])
        _xb = _t.tensor([encode_planes(_b, _b.current_player)], dtype=_t.float32)
        assert _t.equal(_xg, _xb), f"augment({_g}) features != replayed position"
        _legal_b = {action_index(*x) for x in _b.valid_actions()}
        assert set(_t.nonzero(_pg[0]).flatten().tolist()) == _legal_b, \
            f"augment({_g}) policy target misaligned with its own features"

    # GPU engine: a second implementation of the rules is only trustworthy if it
    # agrees with the verified one. Same discipline as blind_engine.py, but the
    # divergence here would be silent -- training data would just be wrong.
    _rc = cmd_gpu_parity(argparse.Namespace(games=8, seed=0, device="cpu",
                                            check_encode=True))
    assert _rc == 0, "tensor engine diverged from the reference engine"

    # MCTS: with an UNTRAINED net the value head is noise, so only the exact
    # terminal values can carry the search. If it still finds every forced win,
    # the tree, the terminal handling and the backup SIGN are all correct --
    # a flipped sign would make it actively avoid winning.
    torch.manual_seed(0)
    _az = build_aznet("cpu")
    _az.eval()
    _mcts = MCTS(_az, "cpu", sims=48, seed=3)
    _rng = random.Random(9)
    _offered = _taken = 0
    for _g in range(6):
        e = Engine(2)
        while not e.game_over and _offered < 12:
            va = sorted(e.valid_actions())
            me = e.current_player
            st = e.get_state()
            wins = []
            for a in va:
                e.play(*a)
                if e.game_over and e.winner == me:
                    wins.append(a)
                e.set_state(st)
            if wins:
                pi = _mcts.run([e])[0]
                _offered += 1
                if max(pi, key=pi.get) in wins:
                    _taken += 1
            e.play(*_rng.choice(va))
    assert _offered == 0 or _taken == _offered, \
        f"MCTS declined a forced win ({_taken}/{_offered}) - check backup sign"

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
    bs = sub.add_parser("bench-search")
    bs.add_argument("--net", default="out/net.pt")
    bs.add_argument("--vs", default="ab", choices=POLICIES)
    bs.add_argument("--depth", type=int, default=3)
    bs.add_argument("--games", type=int, default=40)
    bs.add_argument("--budget-ms", type=int, default=30)
    bs.add_argument("--device", default="cpu")
    bs.add_argument("--seed", type=int, default=7)
    cb = sub.add_parser("check-bot")
    cb.add_argument("--cpp", default="out/net_search_bot.cpp")
    cb.add_argument("--games", type=int, default=6)
    cb.add_argument("--seed", type=int, default=5)
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
    pa = sub.add_parser("pack-az")
    pa.add_argument("--net", default="out/gpunet.pt")
    pa.add_argument("--out", default="out/az_bot.cpp")
    pk = sub.add_parser("pack")
    pk.add_argument("--net", default="out/net.pt")
    pk.add_argument("--out", default="out/net_bot.cpp")
    pk.add_argument("--search", action="store_true",
                    help="emit the net inside a negamax search")
    tg = sub.add_parser("train-gpu")
    tg.add_argument("--iters", type=int, default=200)
    tg.add_argument("--batch-games", type=int, default=512)
    tg.add_argument("--steps-per-iter", type=int, default=12)
    tg.add_argument("--updates-per-iter", type=int, default=48)
    tg.add_argument("--batch", type=int, default=2048)
    tg.add_argument("--buffer", type=int, default=600000)
    tg.add_argument("--lr", type=float, default=1e-3)
    tg.add_argument("--tau", type=float, default=0.5)
    tg.add_argument("--eps", type=float, default=0.08)
    tg.add_argument("--opening-plies", type=int, default=12)
    tg.add_argument("--value-weight", type=float, default=1.0)
    tg.add_argument("--augment", action="store_true", default=True)
    tg.add_argument("--eval-every", type=int, default=10)
    tg.add_argument("--eval-games", type=int, default=30)
    tg.add_argument("--device", default="auto")
    tg.add_argument("--seed", type=int, default=0)
    tg.add_argument("--out", default="out/gpunet.pt")
    gp = sub.add_parser("gpu-parity")
    gp.add_argument("--games", type=int, default=64)
    gp.add_argument("--seed", type=int, default=0)
    gp.add_argument("--device", default="auto")
    gp.add_argument("--check-encode", action="store_true")
    ta = sub.add_parser("train-az")
    ta.add_argument("--iters", type=int, default=60)
    ta.add_argument("--sims", type=int, default=48)
    ta.add_argument("--batch-games", type=int, default=64)
    ta.add_argument("--updates-per-iter", type=int, default=48)
    ta.add_argument("--batch", type=int, default=512)
    ta.add_argument("--buffer", type=int, default=200000)
    ta.add_argument("--lr", type=float, default=1e-3)
    ta.add_argument("--eval-every", type=int, default=5)
    ta.add_argument("--eval-games", type=int, default=40)
    ta.add_argument("--device", default="auto")
    ta.add_argument("--seed", type=int, default=0)
    ta.add_argument("--out", default="out/aznet.pt")
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
    if args.cmd == "bench-search":
        sys.exit(cmd_bench_search(args))
    if args.cmd == "check-bot":
        sys.exit(cmd_check_bot(args))
    if args.cmd == "bench-net":
        sys.exit(cmd_bench_net(args))
    if args.cmd == "check-net":
        sys.exit(cmd_check_net(args))
    if args.cmd == "pack-az":
        sys.exit(cmd_pack_az(args))
    if args.cmd == "pack":
        sys.exit(cmd_pack(args))
    if args.cmd == "train-gpu":
        sys.exit(cmd_train_gpu(args))
    if args.cmd == "gpu-parity":
        sys.exit(cmd_gpu_parity(args))
    if args.cmd == "train-az":
        sys.exit(cmd_train_az(args))
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
