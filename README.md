# napkin-referee

Repo 1 of the **napkin-100k series** (this → napkin-selfplay → napkin-shrink →
napkin-forge → napkin-100k). The series question: how much playing strength fits in a
single ≤100k-character CodinGame source file, trained by self-play, deployed honestly on
a live public ladder? This repo builds **the world**: an exact offline replica of the
chosen game, verified bit-level against the venue's own referee — because every later
claim ("Elo in the sim", "strength per kilobyte") inherits its meaning from this parity.

**The game (registered choice, 2026-08-19):** Ultimate Tic-Tac-Toe, CodinGame arena
`tic-tac-toe` — 10,069 ranked bots at selection time, the largest board-game ladder on
the platform, official referee source public, famously an MCTS/NN playground with the
scene's strong names at the top (karliso, RoboStac, TomAlard). Backup if UTTT fails
in some unforeseen way: `othello-1`. Finale stretch target: Mad Pod Racing.

Rules were extracted from the official referee source
([CodinGame/game-ultimate-tictactoe](https://github.com/CodinGame/game-ultimate-tictactoe),
Java, read 2026-08-19), not from folklore. Three details folklore tends to miss, all
load-bearing: the valid-action list the referee sends is **shuffled** per game seed
(order carries no information; parity compares sets); a master-board 3-in-a-row sets the
mover's score to 10, which **dominates** any small-board count (max 9); drawn small
boards mark **nobody's** master cell, so games can end with no master line — then most
small boards won wins, equal counts draw. Wood leagues play plain 3×3 (referee "level
1"); the real game (level 2) starts at promotion out of Wood.

**Account:** one, disclosed — [Napkin100k](https://www.codingame.com/profile) (Kole's).
No alt accounts (staff-stated rule + series rule). Arena ladders only, never timed
contests (their "no technical assistance" clause is the one place the rules and this
project's assisted development collide — registered constraint, doc §fine-print 3).

## What lives here

One file, `napkin_referee.py`:

- **Engine** — exact replica of both referee levels (plain + ultimate), bitboard-based.
- **Encoders** — state→planes and action↔index mappings the later repos will consume.
- **Scripted baselines** — `random`, `greedy` (1-ply, documented), `ab-k` (alpha-beta
  depth k, simple documented eval). These are the sim→ladder calibration instruments.
- **Fuzz harnesses** — engine-vs-blind-reimplementation and engine-vs-official-Java-referee
  parity over seeded random games.
- **CG protocol adapter** — plays any built-in policy over CodinGame's stdin/stdout
  protocol; in live play it recomputes the valid-action set each turn and logs any
  mismatch with what the referee actually sent (venue parity checked on every real
  ladder game, not just offline).
- **Probe emitter** — prints the C++ probe bot used to measure the sandbox (a preview
  of napkin-forge's file-that-writes-a-file trick).

## Registered hypotheses (written 2026-08-19, before any result below existed)

**H1 — venue parity.** The engine will match the official Java referee bit-level —
identical valid-action sets every ply, identical winner and scores every game — over
≥ 100,000 plies of seeded random games at both levels. Any divergence is a bug in our
engine, to be fixed and the whole fuzz re-run from zero. Target: **0 divergences**.

**H2 — blind-reimplementation parity.** An independently coded second engine (written
from the extracted rules text only, never seeing our engine's code) will agree with the
primary engine on 100% of plies over ≥ 1,000,000 plies. This is the series' standard
selfcheck: two implementations, one spec, zero tolerated disagreement.

**H3 — sandbox measurements** (probe bot, run in the real CG submission sandbox):
- (a) The 100k source cap counts **UTF-16 code units** (community claim, unconfirmed by
  staff): a probe source whose byte length exceeds 100KB but whose UTF-16-unit count is
  under 100k will be **accepted**. Prediction: confirmed.
- (b) AVX2 intrinsics compile and run (top-player claims, 2021–2025). Prediction:
  confirmed.
- (c) With `#pragma GCC optimize` enabled, a scalar int8 forward pass of an
  18k-parameter MLP (series-2's deployed net shape) will run in **< 20 µs**, i.e.
  ≥ 5,000 evals inside a 100 ms turn with headroom. Prediction: confirmed at < 20 µs
  median. (This is the "only bytes bind, not time" feasibility claim, now measured
  rather than argued.)

**H4 — ladder placement of scripted baselines** (the sim→ladder calibration, same move
as series 2's registered venue predictions). Ladder structure at registration: divisions
bottom→top ≈ 1,402 / 5,665 / 1,202 / 1,385 / 419 agents (names presumed Wood 2, Wood 1,
Bronze, Silver, top league). Submissions replace each other (one bot per account), so
baselines run sequentially, each archived after rank stabilizes:

- **random**: never promotes out of Wood. Registered: finishes in a Wood league,
  bottom half of its division.
- **greedy (1-ply)**: clears plain-3×3 Wood bosses (1-ply suffices to never lose plain
  TTT against imperfect play), promotes to Bronze, lands in Bronze's **bottom third**.
- **ab-6 (alpha-beta, depth 6, simple eval)**: promotes out of Bronze; registered:
  **Silver division, middle third**, and does NOT reach the top league (the top is
  MCTS/NN territory; fixed-depth minimax with a naive eval should stall below it).

H4 is the riskiest registration (a deliberate feature: if the mapping surprises us,
that's information — it recalibrates what self-play Elo must reach before the campaign).

House rules, unchanged from series 1/2: one file per repo; hypotheses registered in the
README before results; ≥ 5–10 seeds with IQM and bootstrap CIs wherever variance exists;
ties reported as ties; honest nulls welcome; selfchecks via independently-coded second
implementations.

## Results

*(none yet — this section fills in below this line only after the hypotheses above were
committed. Chronology is the whole point.)*
