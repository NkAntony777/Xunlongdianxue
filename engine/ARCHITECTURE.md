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

## PR-2 scoring split

| Module | Role |
|--------|------|
| `scoring.weights` | DEFAULT_WEIGHTS |
| `scoring.candidate` | score_candidate, FusedScore |
| `scoring.gate` | beasts gate |
| `scoring.rank` | find_and_rank_candidates |
| `scoring.serialize` | to_geojson / to_json / _sanitize |
| `pipeline.analyze_aoi` | AOI use-case without HTTP |

`engine.core.fengshui_score` remains a thin re-export facade.

## PR-3 dragon split

| Module | Role |
|--------|------|
| `core.dragon.types` | RidgeLine, PrimaryDragon, … |
| `core.dragon.hydro` | flats / D8 / accumulation |
| `core.dragon.ridge` | extract / vectorize / light mask |
| `core.dragon.entrance` | entrance refine |
| `core.dragon.yaoxia` | 过峡 |
| `core.dragon.primary` | select / reorient / alignment |
| `core.dragon.incoming` | shaozu/xuanwu on ridge |
| `core.dragon.analyze` | analyze_dragon_vein |
| `core.dragon.viewshed` | viewshed + dual anchor |
| `core.dragon_vein` | thin re-export facade |

`pipeline.analyze_aoi` + `structured_from_aoi` power `/api/layers` structured block.
