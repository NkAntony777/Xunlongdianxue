# Xunlong Engine Architecture

## Layering (dependency direction)

```
api → pipeline (future) → scoring / field / core.locate
                       ↘ domain rules (scale, water channels)
io  ← dem/rivers types only
rendering → dem bounds + arrays (no api)
```

Forbidden: `domain/field` importing `api` or `rendering`.

## Package map (PR-1+)

| Package | Responsibility |
|---------|----------------|
| `engine.core.scale` | Site scale L, fractional beast windows |
| `engine.core.field.water_raster` | Water EDT, same-bank, cross-water |
| `engine.core.field.qi` | Qi / score grid, peaks |
| `engine.core.scoring.gate` | Candidate four-beasts / shaozu gate |
| `engine.core.four_beasts_detect` | Locate beasts + re-exports (compat) |
| `engine.core.fengshui_score` | score_candidate + rank + re-exports |
| `engine.core.dragon_vein` | Dragon (to split later) |

## Compatibility

Public imports keep working:

```python
from engine.core.four_beasts_detect import compute_score_grid, beast_distance_windows
from engine.core.fengshui_score import find_and_rank_candidates, _gate_beasts_for_hole
```

Prefer new paths for new code:

```python
from engine.core.scale import beast_distance_windows
from engine.core.field.qi import compute_score_grid
from engine.core.scoring.gate import _gate_beasts_for_hole
```

## Next (PR-2+)

- `scoring/candidate.py`, `scoring/rank.py`, `scoring/serialize.py`
- `pipeline/analyze_aoi.py` thin layers router
- Split `dragon_vein.py` into hydro / ridge / primary
