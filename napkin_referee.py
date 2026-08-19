#!/usr/bin/env python3
"""napkin-referee — repo 1 of the napkin-100k series.

Exact offline replica of CodinGame's Ultimate Tic-Tac-Toe referee
(github.com/CodinGame/game-ultimate-tictactoe, read 2026-08-19), plus the
state/action encoders, scripted baselines, parity fuzz harnesses, and the CG
stdin/stdout protocol adapter. See README.md for the registered hypotheses.

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
  napkin_referee.py selfcheck
  napkin_referee.py fuzz --other PATH [--games N] [--seed S] [--level {1,2}]
  napkin_referee.py match --a POLICY --b POLICY [--games N] [--seed S] [--level {1,2}]
  napkin_referee.py cg --policy POLICY --level {1,2} [--seed S] [--budget-ms MS]
  napkin_referee.py bench

Policies: random | greedy | ab (time-budgeted iterative-deepening alpha-beta).
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


# -- encoders (consumed by napkin-selfplay / napkin-forge) --------------------

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


# -- scripted baselines --------------------------------------------------------

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


# -- probe bot (emitted C++, submitted to the real CG sandbox) ------------------

PROBE_CPP = r'''/* napkin-100k series probe bot (napkin-referee H3). Fully disclosed:
 * this account (Napkin100k) runs open-science experiments - see the profile bio.
 * This bot measures the CG sandbox (stderr lines prefixed PROBE) and plays a
 * simple heuristic game. Emitted by napkin_referee.py (repo: napkin-referee). */
#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline")
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>

// ---- measurement: scalar int8 MLP, series-2 deployed shape 10-128-128-3 ----
static int8_t W1[10*128], W2[128*128], W3[128*3];
static void fill(int8_t* w, int n, uint32_t& s) {
    for (int i = 0; i < n; i++) { s = s*1664525u + 1013904223u; w[i] = (int8_t)(s >> 24); }
}
static int32_t mlp_eval(const int8_t* x) {
    int32_t h1[128], h2[128], out = 0;
    for (int j = 0; j < 128; j++) {
        int32_t a = 0;
        for (int i = 0; i < 10; i++) a += (int32_t)x[i] * W1[i*128+j];
        h1[j] = a > 0 ? a >> 4 : 0;
    }
    for (int j = 0; j < 128; j++) {
        int32_t a = 0;
        for (int i = 0; i < 128; i++) a += h1[i] * W2[i*128+j];
        h2[j] = a > 0 ? a >> 8 : 0;
    }
    for (int j = 0; j < 3; j++) {
        int32_t a = 0;
        for (int i = 0; i < 128; i++) a += h2[i] * W3[i*3+j];
        out += a;
    }
    return out;
}

#if defined(__GNUC__)
__attribute__((target("avx2")))
static int32_t avx2_touch() {
    // trivial AVX2 use; only called after a runtime cpu check
    volatile int32_t v = 0;
    for (int i = 0; i < 32; i++) v += W2[i];
    return v;
}
#endif

static void run_probe() {
    bool avx2 = __builtin_cpu_supports("avx2");
    bool avx512 = __builtin_cpu_supports("avx512f");
    uint32_t s = 42;
    fill(W1, 10*128, s); fill(W2, 128*128, s); fill(W3, 128*3, s);
    int8_t x[10];
    for (int i = 0; i < 10; i++) x[i] = (int8_t)(i * 7 + 1);
    volatile int32_t sink = 0;
    // warmup
    for (int k = 0; k < 100; k++) { x[0] = (int8_t)k; sink += mlp_eval(x); }
    auto t0 = std::chrono::steady_clock::now();
    const int N = 10000;
    for (int k = 0; k < N; k++) { x[0] = (int8_t)k; x[1] = (int8_t)(k>>8); sink += mlp_eval(x); }
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    int32_t touch = avx2 ? avx2_touch() : -1;
    fprintf(stderr, "PROBE avx2=%d avx512f=%d mlp18k_per_eval_us=%.3f checksum=%d touch=%d\n",
            (int)avx2, (int)avx512, us / N, (int)sink, (int)touch);
}

// ---- play: from the referee-provided valid list + tracked 9x9 grid ----------
static int grid[9][9]; // 0 empty, 1 me, 2 opp

static bool wins3(int b[3][3], int who, int r, int c) {
    b[r][c] = who;
    bool w = false;
    for (int i = 0; i < 3 && !w; i++) {
        if (b[i][0] == who && b[i][1] == who && b[i][2] == who) w = true;
        if (b[0][i] == who && b[1][i] == who && b[2][i] == who) w = true;
    }
    if (b[0][0] == who && b[1][1] == who && b[2][2] == who) w = true;
    if (b[2][0] == who && b[1][1] == who && b[0][2] == who) w = true;
    b[r][c] = 0;
    return w;
}

// full negamax for the plain 3x3 Wood game: perfect play, instant
static int nega3(int b[3][3], int who, int* br, int* bc) {
    int best = -2, r0 = -1, c0 = -1;
    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) {
        if (b[r][c]) continue;
        int v;
        if (wins3(b, who, r, c)) v = 1;
        else {
            b[r][c] = who;
            int orr, occ;
            v = -nega3(b, 3 - who, &orr, &occ);
            b[r][c] = 0;
        }
        if (v > best) { best = v; r0 = r; c0 = c; }
        if (best == 1) break;
    }
    if (br) { *br = r0; *bc = c0; }
    return r0 < 0 ? 0 : best;
}

int main() {
    bool first = true;
    bool level1 = true; // falsified the moment any coordinate exceeds 2
    while (1) {
        int orow, ocol;
        if (scanf("%d%d", &orow, &ocol) != 2) return 0;
        if (orow < -1 || orow > 8 || ocol < -1 || ocol > 8) return 1; // never trust stdin
        if (orow >= 0 && ocol >= 0) grid[orow][ocol] = 2;
        if (orow > 2 || ocol > 2) level1 = false;
        int n; if (scanf("%d", &n) != 1 || n < 1 || n > 81) return 1;
        if (n > 9) level1 = false;
        std::vector<std::pair<int,int>> va(n);
        for (int i = 0; i < n; i++) {
            if (scanf("%d%d", &va[i].first, &va[i].second) != 2) return 1;
            if (va[i].first < 0 || va[i].first > 8 || va[i].second < 0 || va[i].second > 8) return 1;
            if (va[i].first > 2 || va[i].second > 2) level1 = false;
        }
        if (first) { run_probe(); first = false; }
        int mr = va[0].first, mc = va[0].second;
        if (level1) {
            // plain 3x3 (or the indistinguishable board-0 opening of level 2):
            // perfect negamax move - valid under both interpretations
            int b[3][3];
            for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) b[r][c] = grid[r][c];
            nega3(b, 1, &mr, &mc);
            // guard: only trust it if the referee lists it as valid
            bool ok = false;
            for (auto& a : va) if (a.first == mr && a.second == mc) ok = true;
            if (!ok) { mr = va[0].first; mc = va[0].second; }
        } else {
            // level 2 heuristic: win the small board > block their win > center > corner
            int bestv = -1;
            for (auto& a : va) {
                int r = a.first, c = a.second, br = r / 3 * 3, bc = c / 3 * 3;
                int b[3][3];
                for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
                    b[i][j] = grid[br+i][bc+j];
                int lr = r % 3, lc = c % 3, v = 0;
                if (wins3(b, 1, lr, lc)) v = 1000;
                else if (wins3(b, 2, lr, lc)) v = 900;
                else if (lr == 1 && lc == 1) v = 5;
                else if (lr != 1 && lc != 1) v = 3;
                if (v > bestv) { bestv = v; mr = r; mc = c; }
            }
        }
        grid[mr][mc] = 1;
        printf("%d %d\n", mr, mc);
        fflush(stdout);
    }
}
'''


AB_CPP = r'''/* napkin-100k series: the `ab` baseline ported to C++ (napkin-referee).
 * Same search and same eval terms as the Python ABPolicy in napkin_referee.py -
 * the ONLY intended difference is nodes searched per turn. Disclosed bot, see
 * the Napkin100k profile and github.com/arose26/napkin-referee.
 * CG compiles at -O0 by default, hence the pragma. */
#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline")
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <algorithm>

typedef uint16_t u16;
static const u16 WIN[8] = {0700, 070, 07, 0444, 0222, 0111, 0421, 0124};
static const u16 FULL = 0777;

static inline bool wins(u16 m) {
    for (int i = 0; i < 8; i++) if ((m & WIN[i]) == WIN[i]) return true;
    return false;
}
// lines where `mine` has exactly 2 and `block` none  (same term as the Python eval)
static inline int threats(u16 mine, u16 block) {
    int n = 0;
    for (int i = 0; i < 8; i++)
        if (!(block & WIN[i]) && __builtin_popcount((unsigned)(mine & WIN[i])) == 2) n++;
    return n;
}

struct Pos {
    u16 b[2][9];   // per-player occupancy of each small board
    u16 own[2];    // boards won, as a 9-bit mask
    u16 drawn;     // boards full with no winner
    int last;      // previous move cell 0..80, -1 if none
    int side;      // player to move: 0 or 1
    int cnt[2];    // small boards won (the referee's running score)
};

static inline bool decided(const Pos& p, int b) {
    return ((p.own[0] | p.own[1] | p.drawn) >> b) & 1;
}

// writes legal cells (0..80) into out[], returns how many
static int legal(const Pos& p, int* out) {
    int n = 0;
    int tb = p.last < 0 ? -1 : ((p.last / 9) % 3) * 3 + (p.last % 9) % 3;
    if (tb >= 0 && !decided(p, tb)) {
        u16 occ = p.b[0][tb] | p.b[1][tb];
        int br = (tb / 3) * 3, bc = (tb % 3) * 3;
        for (int i = 0; i < 9; i++)
            if (!((occ >> i) & 1)) out[n++] = (br + i / 3) * 9 + bc + i % 3;
        return n;
    }
    for (int b = 0; b < 9; b++) {
        if (decided(p, b)) continue;
        u16 occ = p.b[0][b] | p.b[1][b];
        int br = (b / 3) * 3, bc = (b % 3) * 3;
        for (int i = 0; i < 9; i++)
            if (!((occ >> i) & 1)) out[n++] = (br + i / 3) * 9 + bc + i % 3;
    }
    return n;
}

// returns true if this move ended the game with `side` winning the master board
static bool apply(Pos& p, int cell) {
    int r = cell / 9, c = cell % 9;
    int b = (r / 3) * 3 + c / 3, i = (r % 3) * 3 + (c % 3);
    int s = p.side;
    p.b[s][b] |= (u16)(1 << i);
    bool masterWin = false;
    if (wins(p.b[s][b])) {
        p.own[s] |= (u16)(1 << b);
        p.cnt[s]++;
        if (wins(p.own[s])) masterWin = true;
    } else if ((p.b[0][b] | p.b[1][b]) == FULL) {
        p.drawn |= (u16)(1 << b);
    }
    p.last = cell;
    p.side = 1 - s;
    return masterWin;
}

static int evaluate(const Pos& p, int me) {
    int opp = 1 - me;
    int v = 100 * (p.cnt[me] - p.cnt[opp]);
    v += 10 * (threats(p.own[me], (u16)(p.own[opp] | p.drawn))
             - threats(p.own[opp], (u16)(p.own[me] | p.drawn)));
    for (int b = 0; b < 9; b++)
        if (!decided(p, b))
            v += threats(p.b[me][b], p.b[opp][b]) - threats(p.b[opp][b], p.b[me][b]);
    return v;
}

static std::chrono::steady_clock::time_point deadline;
static bool timeUp = false;
static long nodes = 0;
static inline bool outOfTime() {
    if ((++nodes & 1023) == 0 && std::chrono::steady_clock::now() > deadline) timeUp = true;
    return timeUp;
}

static const int WIN_SCORE = 30000;

static int search(Pos& p, int depth, int alpha, int beta, int me, bool ended, int winner) {
    if (ended) return winner == me ? WIN_SCORE - (64 - depth) : -(WIN_SCORE - (64 - depth));
    int mv[81];
    int n = legal(p, mv);
    if (n == 0) {  // no moves: most small boards won decides it
        int d = p.cnt[me] - p.cnt[1 - me];
        return d > 0 ? WIN_SCORE - 100 : d < 0 ? -(WIN_SCORE - 100) : 0;
    }
    if (depth == 0) return evaluate(p, me);
    if (outOfTime()) return evaluate(p, me);
    bool maxing = (p.side == me);
    int best = maxing ? -WIN_SCORE * 2 : WIN_SCORE * 2;
    for (int k = 0; k < n; k++) {
        Pos q = p;
        bool w = apply(q, mv[k]);
        int v = search(q, depth - 1, alpha, beta, me, w, w ? p.side : -1);
        if (maxing) { if (v > best) best = v; if (best > alpha) alpha = best; }
        else        { if (v < best) best = v; if (best < beta)  beta  = best; }
        if (beta <= alpha) break;
        if (timeUp) break;
    }
    return best;
}

int main() {
    Pos p;
    memset(&p, 0, sizeof(p));
    p.last = -1; p.side = 0;
    bool first = true;
    while (true) {
        int orow, ocol;
        if (scanf("%d%d", &orow, &ocol) != 2) return 0;
        if (orow < -1 || orow > 8 || ocol < -1 || ocol > 8) return 1;
        if (orow >= 0 && ocol >= 0) apply(p, orow * 9 + ocol);
        int n;
        if (scanf("%d", &n) != 1 || n < 1 || n > 81) return 1;
        int refmv[81];
        for (int i = 0; i < n; i++) {
            int r, c;
            if (scanf("%d%d", &r, &c) != 2) return 1;
            if (r < 0 || r > 8 || c < 0 || c > 8) return 1;
            refmv[i] = r * 9 + c;
        }
        // budget: 100ms per turn, 1000ms on the first - stay well inside both
        int budget = first ? 900 : 88;
        first = false;
        deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(budget);
        timeUp = false; nodes = 0;
        int me = p.side;
        int best = refmv[0], depthReached = 0;
        for (int depth = 1; depth <= 81 && !timeUp; depth++) {
            int localBest = -WIN_SCORE * 2, localMove = refmv[0];
            for (int k = 0; k < n; k++) {
                Pos q = p;
                bool w = apply(q, refmv[k]);
                int v = search(q, depth - 1, -WIN_SCORE * 2, WIN_SCORE * 2, me, w, w ? me : -1);
                if (timeUp) break;
                if (v > localBest) { localBest = v; localMove = refmv[k]; }
            }
            if (!timeUp) { best = localMove; depthReached = depth; }
            if (localBest >= WIN_SCORE - 200) break;   // forced win found
        }
        fprintf(stderr, "depth=%d nodes=%ld\n", depthReached, nodes);
        apply(p, best);
        printf("%d %d\n", best / 9, best % 9);
        fflush(stdout);
    }
}
'''


def cmd_emit_cpp(args):
    sys.stdout.write(AB_CPP)
    print(f"emitted ab-cpp: {len(AB_CPP.encode('utf-8'))} UTF-8 bytes (cap 100000)",
          file=sys.stderr)
    return 0


def cmd_emit(args):
    """Print a standalone CodinGame Python bot: this file plus a bootstrap that
    runs the chosen policy over the CG protocol. (A file that writes a file.)"""
    with open(__file__, encoding="utf-8") as f:
        src = f.read()
    src = src.replace('if __name__ == "__main__":\n    main()\n', "")
    src += (
        "\n# ---- CodinGame entry point (emitted by `napkin_referee.py emit`) ----\n"
        "# Disclosed bot for the napkin-100k series (github.com/arose26).\n"
        "import argparse as _ap\n"
        f"cmd_cg(_ap.Namespace(policy={args.policy!r}, level={args.level}, "
        f"seed={args.seed}, budget_ms={args.budget_ms}))\n"
    )
    sys.stdout.write(src)
    print(f"emitted {args.policy}: {len(src.encode('utf-8'))} UTF-8 bytes "
          f"(cap 100000)", file=sys.stderr)
    return 0


def cmd_probe(args):
    src = PROBE_CPP
    if args.utf16_blob:
        # H3a discriminator: pad with U+0100 chars so UTF-8 bytes > 100KB while
        # UTF-16 code units stay < 100k. Accepted <=> the cap counts UTF-16 units.
        n = 60000
        src += "/* UTF-16 counting probe: " + "Ā" * n + " */\n"
    sys.stdout.write(src)
    units = len(src.encode("utf-16-le")) // 2
    print(f"probe source: {len(src.encode('utf-8'))} UTF-8 bytes, "
          f"{units} UTF-16 units", file=sys.stderr)
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

    # Probe emitter: the two things that silently break it are a missing
    # optimize pragma (CG compiles -O0) and miscounted source size.
    assert '#pragma GCC optimize' in PROBE_CPP
    assert len(PROBE_CPP.encode("utf-8")) < 100000  # the venue's real cap (measured)

    # Emitted bots must be standalone: no argparse main left to swallow argv, a
    # bootstrap call present, and still under the byte cap.
    import io
    from contextlib import redirect_stdout, redirect_stderr
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(io.StringIO()):
        cmd_emit(argparse.Namespace(policy="greedy", level=2, seed=0, budget_ms=85))
    bot = buf.getvalue()
    assert 'if __name__ == "__main__":\n    main()' not in bot
    assert "cmd_cg(_ap.Namespace(policy='greedy'" in bot
    assert len(bot.encode("utf-8")) < 100000
    compile(bot, "emitted_bot", "exec")  # must at least parse

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
    p = sub.add_parser("probe")
    p.add_argument("--utf16-blob", action="store_true")
    ec = sub.add_parser("emit-cpp")
    e = sub.add_parser("emit")
    e.add_argument("--policy", required=True, choices=POLICIES)
    e.add_argument("--level", type=int, default=2, choices=(1, 2))
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--budget-ms", type=int, default=85)
    s = sub.add_parser("snapshot")
    s.add_argument("--arena", default="tic-tac-toe")
    s.add_argument("--pseudo", default="Napkin100k")
    s.add_argument("--out", default="out/ladder_snapshots.jsonl")
    args = ap.parse_args()
    if args.cmd == "probe":
        sys.exit(cmd_probe(args))
    if args.cmd == "snapshot":
        sys.exit(cmd_snapshot(args))
    if args.cmd == "emit":
        sys.exit(cmd_emit(args))
    if args.cmd == "emit-cpp":
        sys.exit(cmd_emit_cpp(args))
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
