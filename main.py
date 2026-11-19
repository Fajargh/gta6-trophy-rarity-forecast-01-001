"""GTA VI trophy-rarity forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Forecasts, for every trophy of Grand Theft Auto VI on PS5, its earn rate on
Exophase at the case's target, as five quantiles, scored by pinball loss on
the logit scale. Python 3.13, standard library only, no network, fully
deterministic (no randomness, no wall-clock or dict-order dependence).

The model, in brief
-------------------
An Exophase earn rate is a ratio: earners / players_tracked. The two sides
move for different reasons, so the model forecasts them separately and
divides, instead of extrapolating the ratio itself:

* players_tracked N(t): new players sync their profiles over launch week.
  log N is fitted against log time since release, log(1 + hours).
* earners E_i(t) for trophy i: players who have earned it. log E_i is
  fitted the same way, per trophy, with the slope shrunk toward a prior
  that depends on the trophy's class (story chapters rise fastest as the
  player base plays through, grind/collectible trophies slowest, the
  platinum slowest of all).

Fits are weighted least squares over the distinct states of the history
(consecutive identical reads are one state, so a long flat stretch cannot
outweigh a refresh). The forecast reads both fits at the target time.

Two regimes, because the history can be flat:

* Rich history (several distinct states): the trend fit above, with a
  spread that combines the fit residual, the slope's standard error times
  the extrapolation distance, and a floor that grows with the horizon.
* Flat history (one distinct state, as on launch night): no trend exists
  to fit, and the page is stale — Exophase refreshes on its own schedule,
  hours apart. The kit's rules record the first refresh: tracked players
  638 -> 3,868 (x6.06) while trophies earned went 943 -> 4,276 (x4.53),
  and the 100% trophy alone accounts for +3,230 of the +3,333 new
  earners, so every other trophy gained almost no earners while the
  denominator grew sixfold. A flat page therefore means the next refresh
  dilutes most rates hard. The median is the last rate times a
  class-dependent dilution factor (story ~0.3-0.4, grind/activity ~0.2),
  with a wide spread (1.0 logit) because how much each trophy's earners
  grow in the refresh is very uncertain.

Logical constraints are enforced on the medians: the chapter chain is
monotone (Chapter 1 >= Chapter 2 >= Chapter 3 >= Chapter 4 >= final
mission, since each chapter requires the one before), and the platinum
cannot exceed any other trophy (it requires all of them). The one trophy
at 100% (every tracked player has it) is forecast to stay at ~100% with a
narrow lower tail.

Quantiles are normal quantiles on the logit scale around the median:
q_tau = inv_logit(logit(median) + z_tau * sigma), z = (-1.645, -0.674, 0,
0.674, 1.645). The spread is deliberately honest: on the logit scale a
band that is too narrow is punished far harder than one a little wide,
and a 4.5-day horizon over a launch week is genuinely uncertain.

How the grader runs it: `python3 -P -s /bot/main.py`, reading every
<id>.in in $MAKEFUN_INPUT_DIR (else /input) and writing <id>.out to
$MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release-day origin for the log-time axis: hours since 2026-11-19T00:00Z.
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """Round to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(when: datetime) -> float:
    return max(0.0, (when - RELEASE).total_seconds() / 3600.0)


def log_time(hours: float) -> float:
    """Log-time axis: ln(1 + hours since release)."""
    return math.log1p(hours)


# ---------------------------------------------------------------------------
# Trophy classification from metadata only (tier + description).
# ---------------------------------------------------------------------------

_GRIND_WORDS = (
    "stunt", "location", "card set", "challenge", "race", "vehicle",
    "every spot", "all of leonida", "meditate", "laundering", "garage",
    "register", "explore all", "mission strand", "gold medal",
    "drone race", "shooting range", "100 vehicles", "50 challenges",
    "250 stunts", "20 different races",
)


def classify(trophy: dict) -> str:
    """One of: platinum, completion, chapter1..4, final, mission, grind,
    activity. Derived from tier and description wording only."""
    tier = trophy.get("tier", "")
    desc = (trophy.get("description") or "").strip()
    low = desc.lower()
    if tier == "platinum":
        return "platinum"
    if "100% completion" in low:
        return "completion"
    m = re.match(r"complete chapter (\d+)\.", low)
    if m:
        return "chapter" + m.group(1)
    if low.startswith("complete the final mission"):
        return "final"
    if low.startswith("complete "):
        return "mission"
    if any(w in low for w in _GRIND_WORDS):
        return "grind"
    return "activity"


# Prior slope of log(earners) against log(1 + hours since release), by
# class. Story trophies accumulate earners fastest as players play
# through; grind/collectible and platinum trophies accumulate slowest.
# The overall level (~0.9) matches the launch-night refresh the kit's
# rules record (total earners x4.53 over the first ~4.5 hours).
PRIOR_SLOPE = {
    "platinum": 0.30,
    "completion": 0.35,
    "chapter1": 1.15,
    "chapter2": 1.05,
    "chapter3": 1.00,
    "chapter4": 0.95,
    "final": 0.90,
    "mission": 0.90,
    "grind": 0.60,
    "activity": 0.80,
}

# Prior slope of log(players_tracked) against log(1 + hours).
PRIOR_SLOPE_N = 1.0

# Flat-history dilution factors: the factor the median rate is multiplied
# by when the history is a single stale page and a refresh is due. From
# the recorded first refresh (tracked x6.06, total earners x4.53, and the
# 100% trophy taking +3,230 of the +3,333 new earners): every other
# trophy's earners barely grew while the denominator grew sixfold, so
# typical rates fall to roughly a fifth; story trophies, which new
# players reach first, fall less.
DILUTION = {
    "platinum": 0.30,
    "completion": 0.30,
    "chapter1": 0.40,
    "chapter2": 0.35,
    "chapter3": 0.33,
    "chapter4": 0.30,
    "final": 0.30,
    "mission": 0.30,
    "grind": 0.18,
    "activity": 0.20,
}

# Weight of the class prior when shrinking a fitted slope: the prior
# counts as this many observations.
PRIOR_WEIGHT = 3.0


# ---------------------------------------------------------------------------
# Weighted least squares of y on x, with slope shrinkage toward a prior.
# Returns (intercept, slope, residual_std, slope_se, x_mean, n).
# ---------------------------------------------------------------------------

def fit_log_trend(xs, ys, prior_slope):
    n = len(xs)
    if n == 0:
        return None
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    sxx = sum((x - x_mean) ** 2 for x in xs)
    if n < 2 or sxx <= 1e-12:
        return (y_mean, prior_slope, 0.5, 0.5, x_mean, n)
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = sxy / sxx
    # Shrink the fitted slope toward the class prior.
    slope = (n * slope + PRIOR_WEIGHT * prior_slope) / (n + PRIOR_WEIGHT)
    intercept = y_mean - slope * x_mean
    if n >= 3:
        resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        var = sum(r * r for r in resid) / (n - 2)
        resid_std = math.sqrt(var)
        slope_se = math.sqrt(var / sxx) if sxx > 0 else 0.5
    else:
        resid_std = 0.5
        slope_se = 0.5
    return (intercept, slope, resid_std, slope_se, x_mean, n)


def distinct_states(case):
    """Collapse the history into distinct states: consecutive reads with
    identical (players_tracked, rates) are one state, kept at the time it
    first appeared. Returns a list of (x, at_hours, n_tracked, rates)."""
    states = []
    prev_key = None
    for read in case["history"]:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        when = parse_time(read["at"])
        hours = hours_since_release(when)
        states.append((log_time(hours), hours, read.get("players_tracked"), rates))
    return states


def forecast_case(case: dict, name: str) -> dict:
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    target_hours = hours_since_release(parse_time(case["target"]))
    x_target = log_time(target_hours)
    classes = [classify(t) for t in trophies]
    n_tr = len(trophies)

    # Fallback if there is no usable history at all: a flat 1% median
    # with a very wide band is still a valid forecast.
    if not case.get("history"):
        rows = []
        for i, t in enumerate(trophies):
            lg = logit(1.0)
            rows.append({"id": t["id"], "q": [round4(inv_logit(lg + z * 1.5)) for z in Z]})
        return {"trophies": rows}

    states = distinct_states(case)
    last_rates = states[-1][3]
    last_n = states[-1][2] or 1

    # ---- players_tracked trend -----------------------------------------
    xs_n = [s[0] for s in states if s[2] and s[2] > 0]
    ys_n = [math.log(s[2]) for s in states if s[2] and s[2] > 0]
    if len(xs_n) >= 2:
        fit_n = fit_log_trend(xs_n, ys_n, PRIOR_SLOPE_N)
    else:
        fit_n = (math.log(max(1, last_n)), PRIOR_SLOPE_N, 0.5, 0.5, xs_n[0] if xs_n else 0.0, len(xs_n))
    a_n, b_n, s_n, se_n, xbar_n, _ = fit_n
    log_n_target = a_n + b_n * x_target
    # Sanity guards on the extrapolation: the tracked count cannot fall
    # (players only join), and a fit blowing up to more than 100x growth
    # over the horizon would be a numerical artefact, not a forecast.
    log_n_target = min(max(log_n_target, math.log(max(1, last_n))),
                       math.log(max(1, last_n)) + math.log(100.0))

    flat = len(states) <= 1

    # ---- per-trophy forecasts ------------------------------------------
    medians = [0.0] * n_tr
    sigmas = [0.0] * n_tr
    for i in range(n_tr):
        cls = classes[i]
        last_rate = last_rates[i] if i < len(last_rates) else None
        if last_rate is None:
            last_rate = 1.0
        # The trophy every tracked player holds stays at ~100%.
        if last_rate >= 99.99:
            medians[i] = 99.99
            sigmas[i] = 0.5
            continue
        if flat:
            medians[i] = clamp(last_rate * DILUTION[cls])
            sigmas[i] = 1.0
            continue
        # Earners series over the distinct states.
        xs, ys = [], []
        for (x, _h, n_trk, rates) in states:
            if i >= len(rates) or rates[i] is None or not n_trk or n_trk <= 0:
                continue
            earners = rates[i] / 100.0 * n_trk
            xs.append(x)
            ys.append(math.log(max(earners, 0.5)))
        if len(xs) >= 2:
            a_i, b_i, s_i, se_i, xbar_i, _ = fit_log_trend(xs, ys, PRIOR_SLOPE[cls])
        else:
            a_i = math.log(max(last_rate / 100.0 * last_n, 0.5))
            b_i = PRIOR_SLOPE[cls]
            s_i, se_i, xbar_i = 0.5, 0.5, xs[0] if xs else 0.0
        log_e_target = a_i + b_i * x_target
        rate = 100.0 * math.exp(log_e_target - log_n_target)
        medians[i] = clamp(rate)
        # Spread: fit residual + slope uncertainty over the extrapolation
        # distance + tracked-count uncertainty + a horizon floor.
        dx = max(0.0, x_target - xbar_i)
        floor = 0.30 + 0.25 * math.sqrt(horizon_days)
        var = s_i * s_i + (se_i * dx) ** 2 + s_n * s_n + (se_n * dx) ** 2 + floor * floor
        sigmas[i] = min(2.0, max(0.25, math.sqrt(var)))

    # ---- logical constraints on the medians -----------------------------
    # Chapter chain: id order follows the story, so Chapter 1 >= Chapter 2
    # >= Chapter 3 >= Chapter 4 >= final mission (each requires the one
    # before). Enforce by cumulative minimum along the chain.
    by_id = {t["id"]: idx for idx, t in enumerate(trophies)}
    chain = [by_id[k] for k in (3, 4, 5, 6, 7) if k in by_id]
    running = None
    for idx in chain:
        if running is None:
            running = medians[idx]
        else:
            running = min(running, medians[idx])
            medians[idx] = running
    # Platinum requires every other trophy: it cannot exceed the rarest.
    if 1 in by_id:
        p = by_id[1]
        medians[p] = min(medians[p], min(medians))

    # ---- quantiles -------------------------------------------------------
    rows = []
    for i, trophy in enumerate(trophies):
        lg = logit(medians[i])
        sigma = sigmas[i]
        qs = [round4(clamp(inv_logit(lg + z * sigma))) for z in Z]
        # Rounding can break monotonicity at the extremes; restore it.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
