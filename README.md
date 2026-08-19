# napkin-100k

Train a neural network by self-play on one laptop, squeeze it — weights and all — into a
single source file under CodinGame's hard **100,000-byte** limit, and find out how much
playing strength actually fits in 100KB of text.

The game is [Ultimate Tic-Tac-Toe](https://www.codingame.com/multiplayer/bot-programming/tic-tac-toe),
a public arena with **10,070 ranked bots** whose top ranks are held by the scene's best
bot engineers. The account is
[Napkin100k](https://www.codingame.com/profile/22639068dad6ecdf6717bb383d739a954432057),
disclosed in its profile bio. One account, arena ladders only, never timed contests.

House rules, unchanged from series 1/2: **one file**; hypotheses registered in this
README *before* results; ≥5–10 seeds with IQM and bootstrap CIs wherever variance
exists; ties reported as ties; honest nulls welcome; selfchecks written as
independently-coded second implementations.

## The constraint that shapes everything

CodinGame rejects any submission over **100,000 bytes** of source. That figure is
measured, not folklore — the widely repeated claim that the limit counts UTF-16 code
units is false, and the difference decides whether a net fits at all:

| source | UTF-8 bytes | UTF-16 units | verdict |
|---|---|---|---|
| padded with U+0100 | 125,824 | 65,824 | **rejected** |
| padded with U+1F600 | 245,824 | 125,824 | **rejected** |
| ASCII | 100,000 | 100,000 | accepted |
| ASCII | 100,001 | 100,001 | **rejected** |
| padded with U+0100 | 99,998 | 52,911 | accepted |

The first row settles it: only 65,824 UTF-16 units, still rejected. **The budget is
exactly 100,000 UTF-8 bytes**, and the whole architecture is sized backwards from it:

    100,000 bytes   hard cap
    -  ~6,000       C++ inference harness
    ----------
      ~94,000       for weights
      / 1.25        base85 chars per int8 weight
    ----------
      ~75,000       weights affordable;  this net uses 68,224

Time does not bind. An int8 net of this class evaluates in **7.478 µs** inside the
CodinGame sandbox (measured there, not extrapolated), so a 100 ms turn affords thousands
of evaluations. AVX2 is available; the sandbox compiles at `-O0` unless the source asks
otherwise, so the emitted file carries its own `#pragma GCC optimize`. **Bytes bind.**

## What lives here

`napkin_100k.py` — one file, four jobs:

1. **The world.** An exact replica of CodinGame's referee, verified bit-level against the
   venue's own Java engine. Every claim below inherits its meaning from this parity, so
   it is checked first and hardest.
2. **The net.** A self-play Q-network, 324→128→128→81 (68,224 weights), trained with the
   recipe carried from series 1/2 — replay buffer, target network, league of past selves.
3. **The opponents.** `random`, `greedy`, and a time-budgeted alpha-beta, used to measure
   the net offline. The alpha-beta is the real bar: it is a strong player in this game.
4. **The packer.** torch checkpoint → int8 → base85 → one self-contained C++ source with
   a hand-rolled forward pass. One file that writes one file.

Commands: `selfcheck`, `fuzz`, `train`, `pack`, `check-net`, `bench-net`, `match`,
`snapshot`.

## Registered hypotheses

Written before the corresponding results existed. Chronology is the point; the commit
history is the timestamp.

**H1 — the replica matches the venue.** Our engine will agree with CodinGame's official
Java referee bit-for-bit: identical legal-move sets every ply, identical winner and
scores every game, over ≥ 100,000 plies of seeded random games at both rule levels.
Target: **zero divergences**.

**H2 — the replica matches an independent implementation.** A second engine, coded from
the extracted rules alone and never shown our engine's source, will agree with it on
100% of plies over ≥ 1,000,000 plies.

**H3 — the net.** The hypothesis the project exists for.

1. **Beats every scripted opponent offline**, both seats, in this repo's verified engine:
   ≥ 90% vs `random`, ≥ 75% vs `greedy`, and **> 50% vs the alpha-beta**. The third is
   load-bearing — alpha-beta with a decent search depth is a genuinely strong UTTT
   player, and beating it is what makes a 68k-weight net interesting.
2. **Quantisation cost measured, not assumed**: int8 + base85 fits under 100,000 bytes,
   and the int8 net loses **< 3%** win rate against the fp32 net it came from. "It fits
   after quantisation" is not the same claim as "it is as strong after quantisation",
   and series 1/2 never measured the difference.
3. **The emitted C++ is faithful to torch**: same chosen move on ≥ 99.9% of random legal
   positions. A packer that silently alters the policy would invalidate everything
   downstream, so this gates every submission.
4. **Ladder**: the net earns its own standing on the public arena. **Registered risk,
   stated plainly: Legend is not predicted.** A 68k-weight MLP choosing moves greedily,
   with no search, against a field whose summit is MCTS and neural nets, may well stall
   well short. If it does, that is the result and it will be reported as the result.

**Registered null that would matter:** if the int8 net cannot beat the alpha-beta
offline, the honest headline of this project is "a hand-written search beat our net
inside the same 100KB" — and that is what will be written.

**H4 — is lookahead the missing ingredient?** (registered before implementing)

H3 diagnosed the 0–60 loss as *absence of search*, not weak weights. That diagnosis is
cheap to test directly: take **the same trained net, unchanged**, and use it as the leaf
evaluator inside a negamax search instead of picking its argmax move. Nothing is
retrained; the only variable is lookahead.

**Registered prediction:** the same net, searched to depth 3, beats `ab` at > 50%.
If lookahead is really the deficit, this closes most of a 0–60 gap. If it does not,
my diagnosis was wrong and the problem lies in the learned evaluation itself.

**H5 — the AlphaZero loop** (registered before implementing). Replace DQN-with-argmax
with the arrangement this game actually rewards: a policy+value network guiding an MCTS,
trained on **search-improved targets** — the search's visit distribution becomes the
policy target, the game outcome becomes the value target, and the improved policy feeds
the next round of self-play. The net supplies all learned knowledge; the search carries
no hand-tuned evaluation. Sized to the same measured budget.

**Registered predictions:** (1) it beats `ab` offline at ≥ 60%, both seats; (2) it packs
under 100,000 bytes *including* the MCTS harness; (3) only if (1) and (2) hold does it go
to the public arena — a bot that loses to a baseline is not worth a submission.

## Results

**H1 — CONFIRMED.** Against the official Java referee, driven headlessly with our engine
playing both seats and re-deriving the legal-move set every ply:

- Ultimate rules: **2,412 games / 137,250 plies — 0 legal-move mismatches, 0 outcome
  disagreements** across the 2,329 games reaching a comparable end state.
- Plain 3×3 rules: **700 games / 5,309 plies — 0 mismatches, 0 disagreements.**

Two caveats kept on the record rather than smoothed away. 83 of the 2,412 games (3.4%)
ended with our agent timing out and so produced no comparable end state; they arrived in
three contiguous bursts coinciding with heavy load on this laptop, and re-running those
exact seeds on an idle machine produced **0 timeouts and 0 mismatches**, which exonerates
the engine. Separately the level-2 run stopped after 2,412 of 2,600 requested games
without printing its summary line, with its stderr discarded — an instrumentation gap,
fixed by keeping stderr next time. The data collected still exceeds the registered
100,000-ply threshold by 37%.

**H2 — CONFIRMED.** A clean-room second engine (`blind_engine.py`, written from the rules
text alone) agrees with the primary engine on **1,472,677 plies over 25,000 ultimate-rules
games and 15,269 plies over 2,000 plain games — 0 divergences.**

Repro:

```bash
python3 napkin_100k.py fuzz --other blind_engine.py --level 2 --games 25000 --seed 12
```

**H4 — CONFIRMED, and it identifies the real deficit.** The *same trained net*, weights
untouched, used as the leaf evaluator inside a depth-3 negamax instead of taking its own
argmax:

| opponent | net alone (argmax) | same net + depth-3 search |
|---|---|---|
| `greedy` | 0.505 | **0.950** [0.764, 0.991] |
| `ab` (30 ms) | **0.000** (0W–60L–0D) | **0.525** [0.375, 0.671] (21W–19L) |

Nothing was retrained. The only change is lookahead, and it moves the net from losing
every single game against `ab` to roughly even with it. The H3 diagnosis was right:
**what the net was missing is search, not weights.**

Stated carefully, because the registered prediction was "> 50%": the point estimate
0.525 clears it, but the interval spans parity, so the honest reading is *the net with
search is now competitive with `ab`, not demonstrably better than it*. A larger run is
under way to tighten that interval.

**H3 — the net: 1 FALSIFIED, 2 CONFIRMED, 3 FALSIFIED (threshold was wrong), 4 not
attempted.** 500 iterations of self-play (≈128,000 games) on one laptop GPU, ~34 minutes.

**H3-1 — FALSIFIED, and decisively on the part that mattered.** Offline, both seats, in
the verified engine:

| opponent | net (fp32) | net (int8, as packed) | registered target |
|---|---|---|---|
| `random` | 0.868 [0.814, 0.908] | 0.890 [0.839, 0.926] | ≥ 0.90 — **missed, just** |
| `greedy` | 0.505 [0.436, 0.574] | 0.507 [0.439, 0.576] | ≥ 0.75 — **missed** |
| `ab` (30 ms) | **0.000** — 0W–60L–0D | — | > 0.50 — **missed utterly** |

The registered null has occurred, so it gets stated in the words it was registered in:
**a hand-written alpha-beta search beat our net, inside the same 100KB.** The net does
not lose narrowly to the search; it loses every single game.

Why, without excuses: this net picks its move by evaluating the current position only —
no lookahead. Its opponent searches roughly eight plies. Ultimate Tic-Tac-Toe is sharply
tactical, so a one-ply evaluator is playing a fundamentally different and much weaker
game, and 128k self-play games is a small fraction of what a net needs to compensate.
The parity between fp32 and int8 above shows the deficit is *not* a quantisation
artefact. It does **not** show that 68,224 weights is enough capacity — precision and
capacity are different constraints, and only the first has been measured. Whether a
larger, over-budget net would close the gap is untested, and testing it is exactly the
strength-versus-bytes curve this project set out to draw.

**H3-2 — CONFIRMED.** int8 costs nothing measurable. Against `random` the packed net
scored **+2.2 points** over fp32 and against `greedy` **+0.2** — both well inside their
confidence intervals, i.e. no detectable loss, comfortably under the registered 3%
ceiling. The packed source is **97,324 bytes, 2,676 under the cap.**

**H3-3 — FALSIFIED as written, but the packer is sound and the threshold was wrong.**
The emitted C++ chose the same move as torch on **94.59%** of positions, not the
registered ≥ 99.9%. Diagnosis rather than assumption: an independently written **numpy
int8** reference — implementing the same arithmetic the C++ does, but sharing none of
its code — agrees with torch on **94.27%**, statistically indistinguishable from the
C++'s 94.59%. So the C++ faithfully implements int8 inference; the disagreements are
inherent to quantisation, and they sit on near-ties (median fp32 Q-gap at a
disagreement **0.0017**, with 93% of disagreements under 0.01 on a Q scale of ±1).
The registered threshold of 99.9% was simply the wrong bar for an int8 net — it is the
right bar for a bit-exact fp32 emitter.

`check-net` was corrected accordingly and now reports two separate things: the packer's
correctness, gated, and quantisation drift, measured. Re-run against the trained net:

    C++ vs int8 reference   208/208 (100.00%)   <- packer correctness, gated at 99.9%
    int8 vs fp32            196/208 ( 94.23%)   <- quantisation drift, measured

**The packer is exactly correct**; the 5.8% of decisions that differ from fp32 are
quantisation, and they cost no measurable strength (H3-2).

**H3-4 — not attempted.** The net has not been submitted to the arena and will not be
while it loses 0–60 to a baseline. Putting it on the public ladder now would measure
nothing that this table has not already measured.

### Training notes (kept because they cost real time)

Two defects in the reinforcement-learning code, both of which produce a *quietly weak*
net rather than an error, and both now locked behind assertions in `selfcheck`:

1. **Double-counted return.** The final game outcome was written onto every transition
   *and* bootstrapped on top of. Fixing it dropped the training loss roughly 40×.
2. **Bootstrapping over illegal moves.** The target took `max` over all 81 actions,
   including illegal ones whose Q-values are never trained and therefore drift freely.
   The signature was unmistakable in hindsight: strength peaked early then decayed while
   the loss climbed 20×. The legal-move mask now travels with the stored next state.
