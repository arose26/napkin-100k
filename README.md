# napkin-100k

Train a neural network by self-play on one laptop, squeeze it — weights and all — into a
single source file under CodinGame's hard **100,000-byte** limit, and find out how much
playing strength actually fits in 100KB of text.

The game is [Ultimate Tic-Tac-Toe](https://www.codingame.com/multiplayer/bot-programming/tic-tac-toe),
a public arena with **10,070 ranked bots** whose top ranks are held by the scene's best
bot engineers. The account is
[Napkin100k](https://www.codingame.com/profile/22639068dad6ecdf6717bb383d739a954432057),
disclosed in its profile bio. One account, arena ladders only, never timed contests.

## Result so far

A self-play-trained neural network, weights and all, inside a single **95,891-byte**
source file: **rank 1,017 of 10,071** on CodinGame's Ultimate Tic-Tac-Toe arena —
**Gold league, top 11%**, on a ladder whose summit is held by the scene's best bot
engineers (zasmu 35.56, Daporan 34.62, MrSubZero 33.85; this net 20.44).

66,752 int8 weights decoded from base85 at startup, evaluated by a hand-rolled forward
pass, with the value head scoring search leaves and the policy head ordering moves. No
libraries, no external data, trained by self-play on one laptop GPU.

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

**The cap is platform-wide, not per-game.** Probed directly on seven arenas via the
compile endpoint (which never touches a ladder): 100,000 bytes accepted, 100,001 rejected
with the identical message, on **Ultimate Tic-Tac-Toe, Othello, Connect 4, Spring
Challenge 2021, Yavalath, Breakthrough and Mad Pod Racing** (the flagship racing arena,
formerly Coders Strike Back). The bytes-not-characters semantics travels too: a source of
120,027 UTF-8 bytes but only 60,027 UTF-16 units was rejected on Othello exactly as on
UTTT. So the budget arithmetic in this repo transfers unchanged to any CodinGame arena.


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

**The registered prediction is NOT met.** It said "beats `ab` at > 50%". A larger
120-game run returned **exactly 60W–60L, score 0.500** [0.412, 0.588]; pooled with the
first 40 games that is **81W–79L over 160 games = 0.506** [0.430, 0.583]. This is a dead
heat, not a win. The prediction as written is unsupported and is recorded as such.

What *is* strongly supported is the diagnosis behind it: adding lookahead moved the same
weights from **0.000 to 0.506** against the same opponent. The deficit H3 exposed was
search, and search alone recovers all of it — up to parity with the alpha-beta, and no
further. That ceiling is itself informative: a net trained by DQN to pick moves is not
also a good *evaluation function*, which is precisely the gap the AlphaZero loop (H5)
attacks by training the value head on game outcomes and the policy head on search.

**H5 — the AlphaZero loop: NOT MET at this compute budget, and the search ceiling is
the more interesting finding.**

The loop is implemented and verified: a policy+value net (324→128→96 trunk, 61,632
weights) guiding a batched PUCT **tree** search with exact terminal values and no
hand-written evaluation anywhere. Correctness of the search is asserted in `selfcheck`
by a test that does not depend on training at all — with an *untrained* net, where the
value head is pure noise and only terminals carry signal, the search still found
**111 of 111 forced wins**. A flipped backup sign would make it avoid them.

Trained on one laptop GPU it does not reach competitive strength: after 60 iterations
(~22 min) it scores 0.07–0.30 against `greedy`, with the value loss *rising* (0.30 →
0.67) as the buffer fills with targets from many policy generations. AlphaZero is
compute-hungry and this is a fraction of what it needs. **Registered prediction (1)
≥ 60% vs `ab` is not met, so gate (3) forbids a submission, and none was made.**

**The search ceiling.** Putting the H3/H4/H5 numbers side by side against the same
opponent tells a cleaner story than any of them alone:

| player | vs `ab` | games |
|---|---|---|
| DQN net alone (argmax, no search) | 0.000 | 60 |
| same net + depth-3 negamax (Python, fp32) | 0.506 | 160 |
| same net + iterative-deepening negamax (C++, int8, depth 4) | 0.467 | 60 |

Deeper search did **not** help. Three separate arrangements of the same weights all land
at or just below parity with `ab`, and the deepest of them is not the best. That is the
signature of a weak *evaluation function*: extra depth propagates evaluation error rather
than averaging it away. The DQN net was trained to rank moves for argmax selection, not
to score positions, and it shows the moment it is asked to be an evaluator. The H4 ceiling
is therefore not an artefact of depth — it is the quality of the learned evaluation, which
is exactly what an AlphaZero value head is meant to fix and exactly what more compute
would buy.

The packed searching bot is nonetheless a working artefact: **97,321 bytes** including
weights and search, 2,679 under the cap, reaching depth 4 inside the turn budget and
scoring 0.917 against `greedy`.

**A robustness bug the correctness gate caught.** `check-bot` — which asserts that a
packed bot never plays an illegal move and never declines a forced win — found the
searching bot declining wins. The cause was not the search: with the machine idle it takes
**10 of 10**, and the emitted bot's own move generator reported **zero** disagreements
with the referee's legal list. It was CPU contention from a concurrent training run
exhausting the turn budget, at which point the bot fell back to whatever move happened to
be first. Two fixes: the fallback is now the net's own highest-Q move rather than an
arbitrary one, and depth 1 is never abandoned on time, so a forced win is always seen.
The lesson generalises beyond this bug — **every benchmark of a time-budgeted bot must be
run on an idle machine.** The 0.467 above is a re-measurement after that was fixed; the
first attempt, taken while training ran, read 0.417.

The packed searching bot is nonetheless a working artefact: **96,689 bytes** including
weights and search, 3,311 under the cap, reaching depth 4 inside the turn budget and
scoring 0.917 against `greedy`.

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

### Deployed bot (2026-08-19)

The net is live on the arena: **`out/net_search_bot.cpp`**, 98,626 bytes, 1,374 under the
cap — the self-play-trained weights plus the search that uses them as its evaluation
function. It compiles in the venue's sandbox, reaches depth 3 there inside the turn
budget, and its own move generator reports zero disagreements with the referee's legal
list. The account is in **Gold**.

**This submission does not meet the gate I registered for it.** H5 said a bot goes to the
arena only at ≥ 60% against `ab`; measured, this one is at **0.467**. It was deployed at
the owner's explicit direction, and the reason is worth recording because it is not about
strength: until now the account's ladder position was held by a *hand-written* alpha-beta.
Replacing it means the bot on the public ladder is finally the thing this project is
actually about. The registered gate stands as written and this is logged as an override,
not as the gate being satisfied.

**One robustness fix was needed first, and the correctness gate is what forced it.**
`check-bot` caught the searching bot declining forced wins — which a negamax with exact
terminal values must never do. It was not a search bug: the bot's internal position was
verified against the referee every turn with zero drift, and on an idle machine it took
10/10. It was the *time budget* — under CPU contention the clock expired before the search
committed, and the bot fell back to an arbitrary move. Three fixes, in increasing order of
how much they actually settle the matter: the fallback became the net's own highest-Q move
rather than the first legal one; depth 1 was made unabortable; and finally an immediately
winning move is now detected **exactly, before any search runs**, because a forced win is
certain knowledge and should never be contingent on a clock. With that, **12/12 forced
wins even while another process pinned a core.**

The generalisable lesson, which also corrected one of my own numbers: **a time-budgeted
bot must be benchmarked on an idle machine.** The first `ab` measurement read 0.417 while
training competed for CPU; re-measured fairly it is 0.467.

### H6 — GPU self-play (registered before results)

The measurement below said the GPU was idle because the *environment* was Python. So the
environment moved onto the GPU: `TensorUTTT` re-expresses the identical rules as batched
bitmask arithmetic — masked legal moves, table-lookup win detection, every game stepped in
lockstep — and candidate expansion goes with it, cloning all 81 moves of every game into
one batch so a one-ply policy improvement costs a single forward pass. Trajectory staging
and the replay buffer are preallocated GPU tensors written by scatter and harvested by
mask, so nothing crosses to the host inside the step loop.

**Verified before trusted.** A second implementation of the rules is worthless until it
agrees with the reference, and a divergence here would be silent — the training data would
simply be wrong. `gpu-parity` plays random games in lockstep against the verified engine
and compares the legal-move **set** every ply, the outcome every game, and the encoder
feature-for-feature. It passes, and it is wired into `selfcheck` so it cannot drift.

**Throughput measured:**

| loop | finished games/s |
|---|---|
| Python AlphaZero loop (previous) | ~3 |
| GPU engine, raw self-play at batch 8192 | **15,680** |
| GPU training loop (engine + 81-way expansion + learning) | **33** |

The training figure is lower than raw self-play because every step now evaluates *every
legal move of every game* — that is real work buying a better policy target, not overhead.
Removing the last per-step host transfers alone took it from 8 to 33 games/s.

**Registered prediction:** with self-play this much cheaper, the value head gets enough
data to become a usable evaluator, and the net clears **> 0.50 vs `ab`** — beating the
plateau that three arrangements of the DQN net all hit. Early signal at 8,080 games:
**0.550 vs `greedy`**, already above the DQN net's 0.505 and above anything the Python
AlphaZero loop reached.

### The ladder disagrees with the offline harness — and the ladder is the real test

The derived-feature net measured **0.483 against `ab`**, statistically level with every
earlier attempt, so the offline harness called it "no better". Submitted to the arena, on
the same account, replacing the previous bot:

| | previous bot | derived-feature net |
|---|---|---|
| global rank | 1,708 / 10,071 | **1,017 / 10,071** |
| ladder score | 15.6 | **20.44** |
| league | Gold | Gold (top 11%) |

**691 places better.** So the offline result was
misleading, and the reason is instructive: `ab` is a *single* hand-tuned opponent, and
being level with one opponent says little about a field of 10,071 that includes many real
MCTS implementations. A benchmark with one opponent has almost no resolution — the ladder
discriminates because it is diverse.

The lesson for the rest of this project: **treat `ab` as a smoke test, not a metric.** It
is useful for catching a bot that is broken, and nearly useless for ranking bots that
work. The measurements worth trusting are the ladder, and offline results against a
*population* of opponents rather than one.

### Derived features: the value head improved, the ceiling did not move

Giving the network the game's own abstractions (per-board and master threats, active
board, drawn flags, board difference) is the first change that moved the value loss at all:

| configuration | value loss | vs `greedy` | vs `ab` |
|---|---|---|---|
| raw 324 features | ~0.90 | 0.85–0.90 | 0.500 |
| raw + 5x value weight | ~0.90 | 0.90–1.00 | — |
| raw + trunk 144 (68,352 weights) | 0.871 | 1.000 | — |
| **derived 364 features** | **0.826** | **1.000** | **0.483** [0.362, 0.607] |

391,029 self-play games in 855 s. Packed to 95,891 bytes, 4,109 under the cap, 10/10
forced wins. So the representation genuinely helped the evaluator, and the bot now beats
`greedy` every single game — **but against `ab` it is still level.** That is the fourth
distinct approach to land on the same number:

| player | vs `ab` |
|---|---|
| DQN net + depth-3 negamax (Python) | 0.506 |
| DQN net + iterative-deepening negamax (C++) | 0.467 |
| GPU-AlphaZero net, raw features + C++ search | 0.500 |
| GPU-AlphaZero net, derived features + C++ search | 0.483 |

Every one of precision, search depth, loss weighting, capacity and input representation
has now been tested. Four of them changed nothing; the last improved the value head
measurably and still did not move the result. At this point the honest reading is not
"the net is broken" but that **`ab` is a genuinely strong opponent and parity with it is
where this class of net lands** — a ~60k-weight evaluator inside a shallow search is about
as good as a well-tuned hand-written alpha-beta, and no better. Beating it looks like it
needs a different order of training compute, not another knob.

### H6 result — prediction NOT met, and the same ceiling for the third time

209,850 self-play games at 96 games/s (~36 min), then packed and benchmarked on an idle
machine: **30W–30L–0D of 60 vs `ab`, score 0.500** [0.377, 0.623]. The registered
prediction was "> 0.50", so it is not met. Packed size 88,498 bytes; 10/10 forced wins.

Three unrelated arrangements now land on the same number:

| player | vs `ab` |
|---|---|
| DQN net + depth-3 negamax (Python) | 0.506 |
| DQN net + iterative-deepening negamax (C++) | 0.467 |
| **GPU-AlphaZero net + C++ search** | **0.500** |

A ceiling that survives changing the training algorithm, the search implementation and the
language is unlikely to be about any of those. **The value loss says what it is about:
0.90 against a target variance of about 1.0** — the value head explains roughly a tenth of
outcome variance, i.e. it is close to useless as an evaluator. A search whose leaf
evaluation is near-constant degenerates to "exact near the end, guessing in the middle",
which is a decent description of a bot that goes exactly even with a hand-tuned eval.

Note what the improvements did and did not move: the exploration schedule lifted
**vs-greedy 0.367 → 0.833** and the policy loss to 1.95 (uniform would be ln 81 = 4.39),
so the **policy** head is learning well. The **value** head never budged from ~0.90 in any
run. Policy and value are trained through a shared trunk with equal loss weights, and the
policy term is roughly twice the size, so the value head may simply be losing the
gradient competition — which is a testable claim, not a story.

### What the value head actually knows (measured 2026-08-20)

The ~0.90 value loss could mean two very different things: a weak evaluator, or a target
that is genuinely unpredictable in self-play between near-equal players. Bucketing squared
error by game phase separates them, and the comparison that matters is **MSE against the
target's own variance** — a constant predictor of zero scores exactly the variance.

| phase | value MSE | target variance | variance explained |
|---|---|---|---|
| plies 0–10 | 0.961 | 0.954 | ~0% |
| plies 10–20 | 0.966 | 0.954 | ~0% |
| plies 20–30 | 0.963 | 0.954 | ~0% |
| plies 30–45 | 0.929 | 0.953 | 2% |
| plies 45+ | 0.767 | 0.930 | 18% |
| **overall** | **0.916** | **0.949** | **3.5%** |

So the value head is **barely better than a constant zero**, and only becomes slightly
informative in the endgame where the position is nearly resolved anyway. That is the
ceiling: a search whose leaf evaluation carries ~3% of the signal is, in the middlegame,
searching with almost no evaluation at all — exact near terminals, guessing elsewhere.
It explains why three different arrangements all drew level with a hand-tuned eval and why
searching *deeper* did not help.

It also rules out the remaining easy explanations. Value loss stayed at ~0.90 when the
loss weight was raised 5x, and again at trunk width 144 (68,352 weights). Precision, depth,
gradient weight and capacity have now each been tested and none of them is the constraint.
What has *not* been tested is the input representation: the network is a flat MLP over raw
cell occupancy, so it has to rediscover "three in a row" independently for nine small
boards and the master board, from 324 binary inputs. That is the next experiment.

**A methodology note, because it nearly produced a wrong answer.** The first run of this
diagnostic reported MSE 1.66 — *worse* than predicting zero — which would have implied an
inverted value head. It was an artefact: the diagnostic played greedily, so all 512 games
in the batch were the same deterministic game, and it measured one trajectory replicated.
The tell was `n` values landing on exact multiples of 1024 and a target variance of exactly
0.000 in the last bucket. With sampling restored the numbers above are the real ones.

### Two self-play improvements, and what each was worth

**1. Exploration belongs in the opening, not on every ply.** The first GPU run mixed 15%
uniform noise into a temperature-sampled policy at *every* move. With play that noisy the
game's outcome is close to unpredictable from any given position, so the value head was
being asked to regress noise — which is precisely what a value loss frozen near 0.88 looks
like. Confining exploration to the first 12 plies and playing near-greedy afterwards:

| vs `greedy` | previous run | with schedule + augmentation |
|---|---|---|
| at iteration 40 | 0.367 | **0.817** |
| at iteration 120 | 0.633 | **0.833** |

**2. Eight-fold symmetry augmentation — free data.** Ultimate Tic-Tac-Toe is symmetric
under the 8 dihedral transforms, but *only* if the transform is applied to the master board
**and identically to every small board**; otherwise the "your move chooses my board" rule
breaks and the augmented samples are mislabelled. That property is asserted, not assumed.
Then a subtler trap: tensor indexing `x[:, :, perm]` *gathers*, so it applies the
permutation's inverse. That is still a valid symmetry and features and policy receive the
same map, but the way to know is to check, so `selfcheck` replays real games through each
of the 8 maps and asserts `augment`'s output equals the engine's own encoding of the
symmetric position with the policy target still aligned. All 8 pass.

**Byte headroom, measured by packing real nets rather than estimated:**

| trunk | weights | packed size |
|---|---|---|
| 324-128-96 (current) | 61,632 | 88,625 |
| 324-144-96 | 68,352 | **97,289 — fits, 2.7KB spare** |
| 324-152-96 | 71,712 | 101,609 — **over the cap** |

So 144 is the widest trunk this budget allows. Capacity is the one lever still untested:
int8 precision was shown not to cost strength, and depth was shown not to help, so
"is 61,632 weights enough?" remains genuinely open.

### Where the compute actually goes (measured 2026-08-19)

The obvious response to "the value head needs more training" is to rent a GPU. Measured,
that is the wrong lever for this code:

| machine | 5 AlphaZero iterations |
|---|---|
| this laptop (RTX 4050, 6 GB) | 105 s |
| Colab **Tesla T4**, 2 vCPU | **117 s** |

The T4 is *slower*, because nothing that matters is on the GPU. Self-play spends its time
in the Python engine and the per-node MCTS bookkeeping, and Colab supplies only 2 vCPUs
for that. The network forward passes — the only GPU work — are a rounding error beside it.

So the real unlock is not a bigger GPU but **moving self-play onto it**: a tensorised
engine where thousands of games advance as batched bitmask operations, with the tree
search batched alongside. That is the difference between a T4 idling behind two Python
threads and a T4 actually generating training data. Recording it as the identified next
step rather than guessing at hyperparameters.

### Training notes (kept because they cost real time)

Two defects in the reinforcement-learning code, both of which produce a *quietly weak*
net rather than an error, and both now locked behind assertions in `selfcheck`:

1. **Double-counted return.** The final game outcome was written onto every transition
   *and* bootstrapped on top of. Fixing it dropped the training loss roughly 40×.
2. **Bootstrapping over illegal moves.** The target took `max` over all 81 actions,
   including illegal ones whose Q-values are never trained and therefore drift freely.
   The signature was unmistakable in hindsight: strength peaked early then decayed while
   the loss climbed 20×. The legal-move mask now travels with the stored next state.
