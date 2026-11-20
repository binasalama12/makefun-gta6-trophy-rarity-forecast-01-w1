"""GTA VI trophy earn-rate forecaster (make.fun gta6-trophy-rarity-forecast-01, wallet w1).

Model: per-trophy logit-linear trend, with a dilution branch for flat history.

For each trophy, take the case's reads from the last 24 hours before the
checkpoint and fit, by least squares, a straight line of logit(rate) against
ln(1 + hours since 2026-11-19T00:00:00Z).  The median forecast is the line
read at the target instant.  The spread combines a random walk with the
fit's residual scatter:

    spread = sqrt((0.5 * sqrt(horizon_days)) ** 2 + residual_std ** 2)

with residual_std floored at 0.15 logit units.  Quantiles are the median's
logit shifted by the standard-normal z-scores (-1.645, -0.674, 0, 0.674,
1.645), mapped back through the logistic function.

Flat history (fewer than 3 distinct rate values in the window, which is
what launch night looks like: Exophase had not refreshed yet, so every
read is identical): the next refresh dilutes every rate as the tracked
player count jumps, so the median is the last rate scaled by 0.30 in rate
space and the spread is a wide 1.4 logit units.  The one exception is a
trophy already at 100%: the refreshed page kept it at 100%, so its median
stays at 99.99 with a narrow spread.

Degenerate history (fewer than 3 usable reads at all): fall back to the
last rate as the median with the same wide 1.4 logit spread.

All quantiles are clamped to [0.01, 99.99], rounded to 4 decimals
(half up), and forced non-decreasing.  The program is deterministic:
no randomness, no clock reads, no network; inputs are processed in
file-name order.

I/O contract (from the kit's RULES.txt): read every *.in in
$MAKEFUN_INPUT_DIR (else /input), write <id>.out to $MAKEFUN_OUTPUT_DIR
(else /output), one {"trophies": [{"id": n, "q": [q05, q25, q50, q75,
q95]}]} object per case.  Python 3.13, standard library only.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
WINDOW_SECONDS = 24 * 3600

RW_SIGMA = 0.5          # random-walk sigma, logit units per sqrt(day)
RESID_FLOOR = 0.15      # floor on the fit-residual std, logit units
FLAT_SPREAD = 1.4       # spread used on the flat-history / fallback branch
DILUTION = 0.30         # flat-history median = last rate * this, in rate space
FULL_SPREAD = 0.1       # spread for a trophy already at 100%


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def sigmoid_logit(lg):
    # Numerically stable logistic: a strongly negative extrapolated
    # logit would make math.exp(-lg) overflow, so use the exp(lg)
    # form on that side instead.  Same value, no OverflowError.
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as int."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_epoch(moment):
    return (moment - EPOCH).total_seconds() / 3600.0


def quantiles_from(median_logit, spread):
    out = []
    prev = None
    for z in Z:
        q = round4(clamp_rate(sigmoid_logit(median_logit + z * spread)))
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def fit_line(xs, ys):
    """Least-squares y = a + b*x; returns (a, b, residual_std) or None."""
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid_var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
    else:
        resid_var = 0.0
    return a, b, math.sqrt(max(resid_var, 0.0))


def forecast_trophy(rates, times, target, horizon_days):
    """rates/times: parallel lists of one trophy's reads in the window."""
    points = [(t, r) for t, r in zip(times, rates)
              if r is not None and isinstance(r, (int, float))
              and math.isfinite(r)]
    if not points:
        return quantiles_from(logit(0.0), FLAT_SPREAD)
    last_rate = points[-1][1]
    if len(points) < 3:
        # Degenerate history: no trend to fit; last rate, wide spread.
        return quantiles_from(logit(last_rate), FLAT_SPREAD)
    distinct = {r for _, r in points}
    if len(distinct) < 3:
        # Flat history: Exophase has not refreshed yet; the next refresh
        # dilutes the rates, so shift the median down and widen a lot.
        if last_rate >= 99.99:
            return quantiles_from(logit(99.99), FULL_SPREAD)
        return quantiles_from(logit(last_rate * DILUTION), FLAT_SPREAD)
    xs = [math.log1p(max(hours_since_epoch(t), 0.0)) for t, _ in points]
    ys = [logit(r) for _, r in points]
    fit = fit_line(xs, ys)
    if fit is None:
        return quantiles_from(logit(last_rate), FLAT_SPREAD)
    a, b, resid = fit
    x_target = math.log1p(max(hours_since_epoch(target), 0.0))
    median_logit = a + b * x_target
    resid = max(resid, RESID_FLOOR)
    spread = math.sqrt((RW_SIGMA * math.sqrt(horizon_days)) ** 2 + resid ** 2)
    return quantiles_from(median_logit, spread)


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("unexpected levels %r" % (case["levels"],))
    if not case["history"]:
        raise ValueError("case has no history")
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    cutoff = checkpoint.timestamp() - WINDOW_SECONDS
    window = [r for r in case["history"]
              if parse_time(r["at"]).timestamp() >= cutoff]
    if not window:
        window = case["history"][-1:]
    times = [parse_time(r["at"]) for r in window]
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rates = [r["rates"][i] if i < len(r["rates"]) else None for r in window]
        rows.append({"id": trophy["id"],
                     "q": forecast_trophy(rates, times, target, horizon_days)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
