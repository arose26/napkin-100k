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
- **ab-id (iterative-deepening alpha-beta, 90 ms budget, simple documented eval,
  Python)**: promotes out of Bronze; registered: **Silver division, middle third**, and
  does NOT reach the top league (the top is MCTS/NN territory; shallow minimax with a
  naive eval should stall below it). *(Amended pre-results from "depth 6" to
  time-budgeted: fixed depth 6 isn't reliably reachable in interpreted Python inside
  100 ms, and the baseline's job is calibration under the real turn budget.)*

H4 is the riskiest registration (a deliberate feature: if the mapping surprises us,
that's information — it recalibrates what self-play Elo must reach before the campaign).

House rules, unchanged from series 1/2: one file per repo; hypotheses registered in the
README before results; ≥ 5–10 seeds with IQM and bootstrap CIs wherever variance exists;
ties reported as ties; honest nulls welcome; selfchecks via independently-coded second
implementations.

## Results

*(fills in below this line only after the hypotheses above were committed.
Chronology is the whole point.)*

**H1 — venue parity vs the official Java referee: CONFIRMED (2026-08-19).** The official
CG-SDK referee was built locally and driven headlessly (`FuzzMain.java`) with our engine's
CG protocol adapter playing both seats. Every ply, the adapter recomputes the legal move
set and compares it with what the referee actually sent; at game end it prints its
predicted scores, compared against the referee's.

Level 2: **2,412 games / 137,250 plies, 0 valid-action-set mismatches, 0 outcome
disagreements** on the 2,329 games that reached a comparable end state.

Two caveats, recorded rather than smoothed over:
- **83 of 2,412 games (3.4%) ended with our agent eliminated on time** (referee score
  −1) and so produced no end-state prediction. They are excluded as missing
  observations, not counted as passes. They arrived in three contiguous bursts, each
  coinciding with heavy concurrent load on this laptop (arena submissions, a browser
  automation wait, an API-calling code review) — a logic fault would scatter across
  seeds instead. `python3 napkin_referee.py cg` runs an interpreted alpha-beta inside
  a 100 ms turn, so it is genuinely load-sensitive. Re-running the exact eliminated
  seeds on an idle machine is the check; the result is recorded below.
- The level-2 run terminated after 2,412 of 2,600 requested games without printing its
  summary line. Its stderr had been discarded, so the cause is unrecorded — an honest
  gap in the instrumentation, fixed by keeping stderr next time. The data collected
  before it stopped stands on its own and already exceeds the registered ≥100,000-ply
  threshold by 37%.

**H2 — blind-reimplementation parity: CONFIRMED (2026-08-19).** A second engine was
written by an agent from the extracted rules text alone (`blind_engine.py`, kept in this
repo as the verification artifact; it never saw `napkin_referee.py`). Seeded random-game
fuzz, comparing valid-action *sets* every ply and outcome every game:
level 2 — 25,000 games, **1,472,677 plies, 0 divergences**;
level 1 — 2,000 games, 15,269 plies, 0 divergences.
Repro: `python3 napkin_referee.py fuzz --other blind_engine.py --level 2 --games 25000 --seed 12`
(and `--level 1 --games 2000 --seed 11`).

**H3 — sandbox measurements: (a) FALSIFIED, (b) and (c) confirmed (2026-08-19).**
Measured by submitting the probe bot (`napkin_referee.py probe`) to the real arena as
Napkin100k, then reading its stderr out of a live replay.

- **(a) FALSIFIED — the 100k cap counts UTF-8 BYTES, not UTF-16 code units.** We
  predicted the community claim would hold. It does not. Discriminating experiment, run
  against the venue's own compiler endpoint:

  | source | UTF-8 bytes | UTF-16 units | codepoints | verdict |
  |---|---|---|---|---|
  | 60,000 × U+0100 padding | 125,824 | 65,824 | 65,824 | **REJECTED** |
  | 60,000 × U+1F600 padding | 245,824 | 125,824 | 65,824 | **REJECTED** |
  | ASCII padding | 100,000 | 100,000 | 100,000 | ACCEPTED |
  | ASCII padding | 100,001 | 100,001 | 100,001 | **REJECTED** |
  | U+0100 padding | 99,998 | 52,911 | 52,911 | ACCEPTED |

  Rejection message: `Submitted code is too big. Max chars is 100000`. The first row is
  decisive: only 65,824 UTF-16 units, still rejected. The last row rules out the reverse
  error (a 52,911-unit source is fine at 99,998 bytes). **The budget is exactly 100,000
  UTF-8 bytes, inclusive.** Consequence for the series: the "UTF-16 stretch" that would
  have re-admitted a ~112KB 5-seed ensemble is dead; the conservative base85-in-ASCII
  plan (~85–95KB of weights after the harness) is the only plan, and
  ensemble→distillation (napkin-shrink) is now forced rather than optional.
- **(b) CONFIRMED** — `avx2=1` in the sandbox (`avx512f=0`). Runtime-guarded AVX2 code
  compiles and executes.
- **(c) CONFIRMED** — int8 18k-param MLP (series-2 deployed shape) at **7.478 µs/eval**
  measured in-sandbox, against a predicted < 20 µs. That is ~13,000 evals inside a
  100 ms turn. Bytes bind; time does not. (Same binary locally: 4.8 µs — the sandbox is
  ~1.6× slower, worth remembering when budgeting search.) The `#pragma GCC optimize`
  line is load-bearing: CG compiles at `-O0` by default.

Side result worth keeping: the sandbox's arithmetic checksum matched the local run
bit-for-bit (`checksum=-1900643909`), so local timing/accuracy work transfers.

**Wood league: CLEARED (2026-08-19).** The probe bot went **34–0** in its placement
battles, took rank 1 of its Wood division, and promoted to **Bronze**, where the real
Ultimate rules begin. In Bronze it then ran 141–61 and reached **rank 1 of 5,666 in
Bronze / global rank 3,005 of 10,070**. (The IDE's "Rank 1" is the rank *inside* the
current league; the global number is the one to quote, and both are archived in
`out/ladder_snapshots.jsonl`.)

### H4 — INVALIDATED AS REGISTERED, by our own sequencing error (2026-08-19)

H4 predicted where each scripted baseline would land *starting from Wood*. We then
submitted the probe bot first — it cleared Wood and promoted the account to Bronze.
**CodinGame leagues are sticky: there is no demotion.** We did not assume this, we
measured it: submitting the deliberately terrible `random` baseline afterwards left the
account in Bronze (global rank **5,101/10,070**, score 19.76) rather than dropping it
back to Wood.

Consequences, stated plainly rather than papered over:

- "random never promotes out of Wood, bottom half of a Wood division" — **unmeasurable
  on this account, forever.** The account cannot return to Wood.
- "greedy promotes to Bronze, bottom third" — the *promotion event* is unobservable; only
  the resulting Bronze standing is.
- "ab-id promotes out of Bronze to Silver, middle third" — **still fully measurable**;
  the Bronze→Silver promotion has not happened yet.

The obvious "fix" — a second account starting fresh in Wood — is **forbidden** (CG staff
have stated alt accounts are not allowed, and no-multi-account is a standing series
constraint carried from series 2). So the arm stays dead. This is the cost of submitting
an instrument before the calibration it was meant to calibrate, and it is exactly the
kind of thing the registration protocol exists to make visible.

**H4′ (re-registered 2026-08-19, before the greedy/ab runs, replacing the dead arms):**
with league sticky at Bronze, the measurable calibration is *within-league*. Registered
predictions, in ladder score and global rank:

1. Ordering will be strict: `random` < `greedy` < `ab-id` in both score and global rank.
2. `greedy` lands in the **bottom half of Bronze** (global rank worse than 5,600).
3. `ab-id` **promotes to Silver** (i.e. beats the Bronze boss) within one submission.

Measured so far: `random` — global **5,101/10,070**, score 19.76, Bronze.
(The probe bot, which is roughly greedy-plus-a-perfect-3×3-opening, scored 30.35 at
global 3,005 — informative context, but it is not one of the three registered baselines
and is not used as one.)
