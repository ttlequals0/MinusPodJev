# Jev ad-detection benchmark: test report

Date: 2026-09-19
Corpus: MinusPod LLM benchmark, `benchmark/` (relocated from the MinusPod app)
Subject: TypeSafe Jev (per-segment ad-ness judgments) vs 84 chat LLMs

All numbers in this report were reproduced independently from `src/benchmark/` and
`results/raw/`, not taken from the pre-existing generated report. Every analysis passed a
harness self-check that reproduced jev F1 0.9294 / F0.5 0.9572 at IoU 0.5 and the published
report's IoU-0.5 rows before any new result was accepted.

---

## 1. Verdict

Jev does not beat chat LLMs at ad detection on this corpus. It ties the best chat model
within noise, costs far less, and runs far faster. Its errors are shaped differently: it
never cuts blind, but when it is wrong it cuts a whole segment at a time.

- The published "jev 0.957 beats Haiku 0.908" holds only at IoU 0.5. The lead dies between
  IoU 0.76 and 0.78. At IoU 0.8, four chat models beat jev.
- Cross-validated over the parameters fitted to this corpus, jev's margin over Haiku is
  +0.006 F0.5, paired t(11) = 0.131. Not significant.
- Jev is 36x cheaper per episode and about 59x faster per call.
- Jev never cut audio containing no advertising (0.0 s/h spurious across 17.5 hours). No
  chat model matched that.
- Jev removes about 3x more editorial content than Haiku, because segment quantization makes
  every boundary error cost a whole ~26-second segment.

One finding matters more than the ranking. Running Haiku through jev's own per-segment
decomposition beats both jev and Haiku emitting spans directly (F0.5 0.826 vs 0.760 vs 0.722
at IoU 0.8). The decomposition is the strong part; the judge is what to replace.

---

## 2. Corpus and method

| property | value |
|---|---|
| episodes | 14 (12 ad-bearing, 2 no-ad controls) |
| ad-bearing audio | 17.484 h (episodes 16.4 min to 4.31 h) |
| control audio | 1.682 h |
| segments (VAD) | 2908; mean 21.7s, median 24.3s, max 51.4s |
| windows | 171 (600s window, 420s stride, 180s overlap) |
| canonical truth spans | 49 (from 53 raw, 4 merged by the 15s rule) |
| jev thresholds | ENTER 0.95, STAY 0.40, MAX_RUN_GAP 30s, BRIDGE 30s |
| jev passes | 1 (published); 5-pass also measured |
| chat trials | 5 at temperature 0 |

Jev scores by asking one boolean "is this line advertising?" per segment (a TypeSafe "noul"
question), then assembling contiguous runs of high-probability segments into spans with
`spans_from_probabilities`. Every boundary is a real segment edge. The chat models receive
the MinusPod production prompt and emit ad spans directly.

The investigation ran as parallel independent verifications, each re-deriving its numbers
from source. Live jev access was used where needed: the 5-pass run this session fetched 855
windows live against `api.typesafe.ai` (`cache: 0 hit, 855 fetched`).

---

## 3. Reproduction of the published numbers

Confirmed exactly. Jev 1 pass at shipped thresholds, IoU 0.5:

| metric | value |
|---|---|
| F1 | 0.9294 |
| F0.5 | 0.9572 |
| precision | 0.9792 |
| recall | 0.8931 |
| controls | 2/2 PASS |

The ranking rebuild matched `results/report-combined.md` on 86 of 86 rows. The published
figures are real; the issue is what they mean, not whether they reproduce.

---

## 4. The ranking is an artifact of IoU 0.5

The entire benchmark ranks at IoU 0.5 (`DEFAULT_IOU_THRESHOLD`). A match at 0.5 can be off
by half the span, which for a product that cuts audio is a large error scored as a hit.

| IoU | jev F0.5 | rank among chat models | chat models above jev |
|---|---|---|---|
| 0.50 | 0.9572 | 1 | 0 |
| 0.70 | 0.8749 | 1 | 0 |
| 0.76 | 0.8332 | 1 | 0 |
| 0.78 | 0.7811 | 2 | 1 |
| 0.80 | 0.7484 | 5 | 4 |
| 0.90 | 0.7156 | 4 | 3 |

Jev's drop from IoU 0.5 to 0.8 is -0.2088, the largest of any model in the top 25. It is the
steepest beneficiary of the loose threshold. At IoU 0.8 the models that pass it are
qwen3.5-plus-02-15, claude-haiku-4-5, gemini-3.6-flash and gemini-3.5-flash.

---

## 5. Statistical significance

| comparison | jev F0.5 | haiku F0.5 | diff | t(11) |
|---|---|---|---|---|
| as published (jev tuned in-sample) | 0.9572 | 0.9077 | +0.0495 | +1.887 |
| jev cross-validated on its real selection surface | 0.9139 | 0.9077 | +0.0063 | +0.131 |

One-sided critical value at 11 df is 1.796. The published version clears it by 0.09;
two-sided it never clears.

Paired bootstrap (10,000 draws, 12 episodes resampled):

| comparison | IoU | diff | P(jev higher) | 95% CI |
|---|---|---|---|---|
| jev vs haiku[segment_ids] trial 1 | 0.5 | +0.0056 | 0.601 | [-0.0283, +0.0438] |
| jev vs haiku[timestamps], published | 0.5 | +0.0495 | 0.979 | [+0.0021, +0.1000] |
| jev vs haiku[timestamps] | 0.8 | -0.0119 | 0.454 | [-0.1475, +0.1098] |

The published interval excludes zero by two thousandths of a point on 12 episodes; per-episode
record is 6-2-4. Haiku's own trial-to-trial F0.5 range in segment_ids mode is 0.106, more than
double jev's entire 0.0495 headline margin.

### The tuning asymmetry

Jev has 8 configuration constants fitted against these 12 episodes; the published
cross-validation covers 2 of them (enter/stay). Expanding it:

| selection surface | held-out F0.5 | optimism |
|---|---|---|
| enter/stay (published) | 0.957 | +0.000 |
| enter/stay/max_gap | 0.957 | +0.000 |
| enter/stay/bridge | 0.914 | +0.043 |
| all four | 0.914 | +0.043 |

The +0.000 claim is true and reproduces, but it is nearly vacuous: the probability signal is
sharply bimodal (3839 of 4228 answers below 0.40, 210 at or above 0.95), so thresholds do not
overfit because the signal is bimodal, not because the pipeline was validated. The chat models
received zero per-corpus tuning: one prompt commit, never edited, no content retries, no
model-specific special-casing.

---

## 6. Error decomposition: granularity vs assembly vs judgment

A four-level ladder isolates where jev's error lives. L0 truth -> L1 truth snapped to segment
edges -> L2 oracle labels through the real assembler -> L3 live jev. At IoU 0.8, jev's 0.2781
F1 shortfall from perfect splits:

| component | F1 delta | share |
|---|---|---|
| granularity (L0->L1) | 0.0583 | 21.0% |
| assembly (L1->L2) | 0.0296 | 10.7% |
| judgment (L2->L3) | 0.1901 | 68.4% |

Judgment dominates, 6.4x the cost of assembly. The entire assembly delta is one episode;
span lists are byte-identical on 11 of 12. The assembly residual is irreducible by tuning:
the corpus contains a 28.93s silence gap inside one ad (needs max_gap >= 28.93) and a 27.72s
gap between two ads (needs < 27.72). The granularity share is a floor: this corpus's truth is
authored on segment edges (53 of 53 raw starts land on a segment start), so against
independently annotated truth it would be larger.

Judgment failure anatomy: 0 spurious spans (every one of jev's 45 spans lands on a real ad),
4 of 49 truth spans missed entirely (3 are episode-opening pre-rolls), and boundary error is
bimodal (36 of 45 starts exact within 1s, the other 9 off by 5-72s). The failure mode is one
shoulder segment mislabeled, not drift.

---

## 7. The outward leading edge is the model, not the decomposition

Jev's spans open 3.96s early on average (outward, eating editorial). The 2x2 (model x
decomposition), same windows, same assembler, same thresholds:

| row | model | decomposition | F0.5 @0.5 | F0.5 @0.8 | start bias @0.5 |
|---|---|---|---|---|---|
| jev | jev | per-segment + assembler | 0.9572 | 0.7603 | -3.96s |
| haiku-per-segment | haiku | per-segment + assembler | 0.9160 | **0.8261** | -0.07s |
| haiku-segment_ids | haiku | span emission | 0.9050 | 0.7221 | +0.87s |
| jev-ceiling | oracle | per-segment + assembler | 0.9960 | 0.9147 | -1.19s |

Swapping the model under a fixed decomposition moves the start edge 3.89s; swapping the
decomposition under a fixed model moves it 0.94s. The decomposition's own floor is one
episode (tosh-show, truth start mid-segment); excluding it the oracle start bias is exactly
0.00s and jev is still -3.02s.

Mechanism: jev puts shoulder segments in the 0.40-0.95 STAY band, which lets an open run
extend backward over them. Of 179 window-instances jev scores in that band, Haiku pushes 70
above 0.95 and 75 below 0.40, keeping 34. The two models agree on the ad body to three
decimals and diverge only on the shoulder.

Consequence: haiku-per-segment is the strongest non-oracle row measured at IoU 0.8. The
decomposition is not the weak link. The judge is.

---

## 8. Audio damage, seconds per hour (mean)

A threshold-free metric measuring the product's actual harm. Chat rows are per-trial means
over 5 trials (one-pass-equivalent).

| row | editorial removed /h | spurious /h | ad retained /h | >30s events /h | % of damage in >30s |
|---|---|---|---|---|---|
| jev 1 pass | 14.6s | 0.0s | 21.3s | 0.23 | 62.9% |
| jev-ceiling (oracle) | 3.6s | 0.0s | 0.0s | 0.00 | 0% |
| haiku-per-segment | 7.2s | 2.8s | 22.9s | 0.00 | 0% |
| claude-haiku-4-5 [timestamps] | 4.7s | 0.9s | 22.6s | 0.00 | 0% |
| qwen3.5-plus-02-15 | 19.2s | 5.0s | 17.5s | 0.15 | 39.5% |
| google/gemini-3.5-flash | 14.0s | 4.1s | 19.5s | 0.05 | 13.9% |

Jev never cut blind: 0.0 s/h spurious across 17.5 hours, every wrongly removed second
attached to a real ad break. But jev removes ~3x the editorial of Haiku, and 62.9% of that is
in cuts longer than 30 seconds. Jev's median editorial event is 27.3s; the chat models' median
audible event is 0-2s. All 12 rows predicted zero spans on both controls in every trial.

Stated in units per episode: jev costs $0.075 less and finishes ~285s sooner than Haiku,
removes 14.5 more seconds of the listener's show (21.3s vs 6.8s), and leaves about the same
ad in. A cost-and-latency-for-quality trade, not a free win.

---

## 9. Audio damage, tail risk

Per (episode, trial) draw, 60 draws per chat row.

| row | p90 s/h | worst draw s/h | largest single cut s | draws with >30s cut | with >60s cut |
|---|---|---|---|---|---|
| jev (5 passes as draws) | 65.0 | 104.5 | 46.5 | 18/60 | 0/60 |
| claude-haiku-4-5 [timestamps] | 18.0 | 41.1 | 26.5 | 0/60 | 0/60 |
| qwen3.5-plus | 63.4 | 72.9 | 70.1 | 9/60 | 5/60 |
| grok-4.3 | 60.7 | 146.7 | 69.9 | 17/60 | 5/60 |
| gemini-3.5-flash-lite | 18.0 | 100.2 | 133.7 | 2/60 | 1/60 |
| sonnet-4-6 | 58.8 | 60.7 | 64.4 | 9/60 | 2/60 |

Chat models do produce catastrophic cuts: 65 cuts over 30s across 7 of 8 rows, the worst a
133.7s removal by gemini-3.5-flash-lite (126.0s of audible speech, 2.9x jev's worst). Five of
eight chat rows produced a cut over 60s; jev produced none in 60 draws.

Distribution shape (largest cut / median cut): jev 1.7, haiku[timestamps] 883, grok 2331,
gemini-3.5-flash-lite 4457. Jev has no tail; the chat rows are concentrated near zero with an
open tail that occasionally fires, so a low mean can hide a two-minute deletion.

Jev's damage is quantized, not bounded: the assembler chains segments without a cap, and jev's
46.5s cut spans two segments. But against the specific comparator, claude-haiku-4-5[timestamps]
produced zero >30s cuts in 60 draws and wins every risk framing (mean, median, p90, CVaR90
31.7 vs 91.3 s/h, adversarial-episode, smallest worst cut). A structural zero has no sampling
variance, so the episode bootstrap backs that contrast at P = 0.0000.

Identifiability caveat: 12 episodes cannot identify a tail. Every p90 CI is wider than its
point estimate; gemini-3.5-flash-lite's 133.7s cut has a bootstrap CI of [16.4, 133.7], resting
on one episode. The only tail claims that carry weight are the structural zeros and the
>30s-cut-rate contrast.

---

## 10. Cost, latency, determinism

| | jev 1 pass | claude-haiku-4-5 | ratio |
|---|---|---|---|
| cost / episode | $0.0021 | $0.0773 | 36.6x cheaper |
| p50 call latency | 410ms | 24.2s | ~59x faster |
| trial F1 stdev (5-pass) | 0.0242 | 0.0375 | jev more deterministic |

The speed gap is architectural: jev returns typed per-segment probabilities and never
generates prose. Haiku asked the same per-segment questions still took p50 39.2s because it
emits the answers as tokens. Jev's 0.0242 trial variance is the lowest measured; it had never
been measured before, only defaulted to zero.

Five passes make jev worse at the shipped thresholds (F0.5 0.9572 -> 0.9193): averaging pulls
the bimodal probabilities toward the middle so fewer segments clear the 0.95 gate. The 1-pass
and 5-pass results differ in exactly one span corpus-wide. Single pass is strictly better.

---

## 11. Correctness bug found and fixed

`spans_from_probabilities` set a span's end to `members[-1]["end"]` (the last member) rather
than the maximum end. Because segments are sorted by start, a segment strictly containing the
next one truncated the span. Live case: ep-security-now segments 108/109, span ended 2664.88
instead of 2687.39, truncating a real ad by 22.51s.

Fixed in three places (span end, the max_gap running frontier, the bridge wall). Effect:
+0.0119 F1 and F0.5 at IoU 0.8 and 0.9, neutral at 0.5-0.7, end MAE -0.268s at IoU 0.5, both
controls still PASS. Four tests added; reverting any one fix fails exactly one. Root cause is
upstream duplicate segments from `transcriber.py` chunk-seam merging, not the benchmark.

---

## 12. Reporting defects fixed

Three figures previously believed fixed had not landed:

- Cost normalization was split across two denominators (`jev.py` per-episode, `aggregate.py`
  corpus-total), so a regeneration would print jev vs Haiku at 515x. Unified to per episode;
  true ratio 36.6x. README "1% of the cost" corrected to 3%.
- Latency rendered a 0.0 default as a measurement, publishing jev as the fastest system. Now
  renders `n/a` when no timings exist and sorts untimed rows last.
- Determinism published 0.0000 from a single trial (n<2). Now renders `n/a`; a real figure is
  produced when multiple passes exist.

Also added an `F0.5 @0.8` column to every ranked table so the strict-threshold collapse is
visible beside the rank, and corrected the window-length description (85s -> 600s/420s).
Test suite 285 -> 298 passing, no existing assertion changed. Both reports regenerated.

---

## 13. The external review, point by point

The investigation was prompted by an external first-pass review. Scorecard:

| claim | verdict |
|---|---|
| Baseline reproduces (F1 0.929, F0.5 0.957) | confirmed exactly |
| IoU sensitivity table | numbers confirmed |
| "Falsifies the ceiling framing" | no; README already scopes it to IoU >= 0.5 |
| "Decomposition fine, assembly is not" | no; measured split is 10.7% assembly, 68.4% judgment |
| Overlap oracle not inflating the ceiling | confirmed at 0.5; at 0.8 it slightly deflates it |
| Start bias -3.96s | confirmed, but 5 spans across 4 episodes |
| lead=0.50 "strictly free" | no; +29.1s ad left in, a 3.4:1 trade |
| lead=0.90 is the right value | no; break-even past 0.50, and it inverts the bias |
| Leave-2-out CV on lead | confirmed and honest; the tie breaks against their thesis |
| "Detection untouched" | holds 0.40-0.90, breaks at 0.95; not structural |
| TAIL_THRESHOLD = 0.40 | correct value, dead code (can never trim) |
| Duplicate segments exist | confirmed, 63 pairs, upstream |
| "Sorting changes which duplicate lands in window" | no; member sets identical, 22 reorder, 24 cache keys |
| "Duplicate straddling an ad boundary" | no; zero corpus-wide |
| The patch applies | no; bare @@ headers, no tool accepts it |
| Open question: does Haiku show the same outward edge | no; +0.87s, inward |

The review's most valuable contribution was indirect: gesturing at the duplicate-segment
problem led to the confirmed truncation bug in section 11. Its proposed `lead`-threshold patch
should not land: `lead > enter` is unguarded (corrupts 10 of 35 grid points), `TAIL_THRESHOLD`
is dead, trimming defeats canonicalization, cross-window dedup undoes it, and the CV numbers in
the patch comment cannot be produced by the patched code.

---

## 14. What would settle it

The corpus cannot resolve a difference of this size (12 episodes, paired SE 0.048).

1. A held-out corpus: 10+ ad-bearing episodes captured after the constants were frozen,
   scored at frozen 0.95/0.40/30/30 with no re-tuning. At $0.0021/episode of jev spend this is
   cheap and is the only thing that would settle the ranking.
2. BRIDGE_SECONDS and the state-variant dimension inside the CV.
3. Truth annotated independently of the segment grid, to measure the real granularity floor.

The engineering direction the data points to: keep the decomposition, replace or improve the
judge, and get boundaries below segment granularity (which would remove both the 3.6 s/h
oracle floor and the 27s median error).

---

## 15. Corrections made during the investigation

Recorded for honesty; each was caught and fixed against the data.

- Claimed jev makes "many small errors"; the tail analysis showed it is few-large-errors.
- Claimed jev's damage is "bounded by construction"; the assembler chains segments uncapped,
  so it is quantized, not bounded.
- Claimed live jev access was an open prerequisite; the 5-pass run had already fetched 855
  windows live this session.
- Initially reported the determinism defect as differently shaped than it was; the live bug is
  `metrics.trial_stdev` returning 0.0 for n<2.

---

## 16. Artifacts preserved

Results (`benchmark/results/`):
- `report.md`, `report-combined.md` (regenerated, post-fix)
- `audit-2026-09-19.md` (the working investigation log, ~600 lines)
- `raw/jev_cache.json` (1026 entries: 171 single-pass + 855 five-pass, live)
- `raw/haiku_persegment_cache.json` (171 windows, 4228 segment answers, the ablation)

Tools: the analysis harness from each parallel verification (baseline, IoU sensitivity,
lead sweep, corpus duplicates, patch review, decomposition ablation, strict-IoU ranking,
tuning/cost audit, error-decomposition ladder, damage metric, tail risk). Each is a
self-contained script that re-derives its numbers from the caches and corpus.

All are captured in the accompanying archive zip.
