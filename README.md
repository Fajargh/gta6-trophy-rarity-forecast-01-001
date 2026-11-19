# gta6-trophy-rarity-forecast-01-001
[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Fajargh/gta6-trophy-rarity-forecast-01-001/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Fajargh/gta6-trophy-rarity-forecast-01-001/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for every Grand Theft Auto VI trophy's earn rate on Exophase at 2026-11-26T00:00:00Z — the percent of the players Exophase tracks who will have earned each trophy one week after launch — as five quantiles per trophy, scored by pinball loss on the logit scale.

## How it works

An Exophase earn rate is a ratio, `earners / players_tracked`, and the two sides move for different reasons, so `main.py` forecasts them separately and divides:

- **Players tracked N(t):** new players keep syncing profiles over launch week. `log N` is fitted against log time since release, `ln(1 + hours)`.
- **Earners E_i(t) per trophy:** players who have earned it. `log E_i` is fitted the same way, per trophy, with the slope shrunk toward a class prior from the trophy's metadata — story chapters rise fastest as the player base plays through, grind/collectible trophies slowest, the platinum slowest of all.
- Fits are weighted least squares over the **distinct states** of the history (consecutive identical reads are one state, so a long flat stretch cannot outweigh a refresh).

Two regimes, because the history can be flat:

- **Rich history** (several distinct states): the trend fit above, with a spread combining the fit residual, the slope's standard error times the extrapolation distance, and a floor that grows with the horizon.
- **Flat history** (one distinct state, as on launch night): no trend exists to fit and the page is stale — Exophase refreshes on its own schedule, hours apart. The kit's rules record the first refresh: tracked players 638 → 3,868 (×6.06) while trophies earned went 943 → 4,276 (×4.53), and the 100% trophy alone accounts for +3,230 of the +3,333 new earners, so every other trophy gained almost no earners while the denominator grew sixfold. The median is therefore the last rate times a class-dependent dilution factor (story ~0.3–0.4, grind/activity ~0.2), with a wide spread (1.0 logit).

Logical constraints are enforced on the medians: the chapter chain is monotone (Chapter 1 ≥ Chapter 2 ≥ Chapter 3 ≥ Chapter 4 ≥ final mission), and the platinum cannot exceed any other trophy, since it requires all of them. The one trophy at 100% is forecast to stay at ~100% with a narrow lower tail.

Quantiles are normal quantiles on the logit scale around the median. The spread is deliberately honest: on the logit scale a band that is too narrow is punished far harder than one a little wide, and a 4.5-day horizon over a launch week is genuinely uncertain.

## How to run it

Python 3.13, standard library only, no network, fully deterministic:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `<id>.in` in the input directory and writes `<id>.out` to the output directory: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`, one row per trophy.

## Validation

On the kit's practice cases (p00/p01, launch night, flat history), the program writes valid outputs for all 55 trophies (non-decreasing quantiles in 0–100, ≤ 65,536 bytes), byte-identical across two runs, in ~0.15 s per run — well under the 8 s cap. Unlike the house baseline, which holds the last rate with a narrow band (practice loss ~1.51), its medians sit at ~0.2× the last rate with a much wider band, because the next refresh dilutes most rates hard while the denominator grows sixfold.

## Files

- `main.py` — the forecaster (single entrypoint, stdlib only)
- `LICENSE` — MIT

<sub>Entered on [make.fun](https://make.fun/?ref=8cvj6bwy&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
