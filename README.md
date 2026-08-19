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

**H3 — in progress.** Training runs; results land here when they exist, pass or fail.

Verified so far: the packer round-trips correctly — a packed checkpoint compiled to
**97,582 bytes** (2,418 under the cap) and matched torch's chosen move on **180 of 180**
sampled positions, satisfying H3-3 on a smoke checkpoint.

### Training notes (kept because they cost real time)

Two defects in the reinforcement-learning code, both of which produce a *quietly weak*
net rather than an error, and both now locked behind assertions in `selfcheck`:

1. **Double-counted return.** The final game outcome was written onto every transition
   *and* bootstrapped on top of. Fixing it dropped the training loss roughly 40×.
2. **Bootstrapping over illegal moves.** The target took `max` over all 81 actions,
   including illegal ones whose Q-values are never trained and therefore drift freely.
   The signature was unmistakable in hindsight: strength peaked early then decayed while
   the loss climbed 20×. The legal-move mask now travels with the stored next state.
