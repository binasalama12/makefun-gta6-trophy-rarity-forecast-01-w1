# gta6-trophy-rarity-forecast-01 — wallet w1

Forecast each of the 55 GTA VI PS5 trophies' Exophase earn rate (%) at the
case's target time, as five quantiles (5/25/50/75/95%) per trophy.

## Approach

Per trophy, a **logit-linear trend**: fit a least-squares line of
`logit(rate)` against `ln(1 + hours since 2026-11-19T00:00:00Z)` over the
case's last 24 hours of reads, and read the line at the target for the
median. The spread combines a random walk with the fit's residual scatter:

```
spread = sqrt((0.5 * sqrt(horizon_days))^2 + residual_std^2)
```

with the residual std floored at 0.15 logit units. Quantiles are the
median's logit shifted by the standard-normal z-scores
(-1.645, -0.674, 0, 0.674, 1.645), mapped back through the logistic
function.

**Flat history branch.** On launch night Exophase had not refreshed yet,
so every read is identical. A flat history means the next refresh is
still ahead of us, and it dilutes every rate as the tracked-player count
jumps (638 → 3,868 on the first refresh). So when a trophy has fewer
than 3 distinct rate values in the window, the median is the last rate
scaled by 0.30 in rate space and the spread is a wide 1.4 logit units.
The one exception is a trophy already at 100%, which the refreshed page
kept at 100%: its median stays at 99.99 with a narrow 0.1 spread.
With fewer than 3 usable reads at all, the fallback is the last rate
with the same wide 1.4 spread.

All quantiles are clamped to [0.01, 99.99], rounded to 4 decimals
(half up), and forced non-decreasing.

## How to run

```sh
MAKEFUN_INPUT_DIR=<input dir> MAKEFUN_OUTPUT_DIR=<output dir> python3 main.py
```

Reads every `*.in` in the input directory and writes `<id>.out` per case
to the output directory. Python 3.13, standard library only, no network,
deterministic (two runs produce identical bytes).

## Files

- `main.py` — the whole program (single file, no dependencies).
- `README.md` — this file.
- `LICENSE` — MIT.
