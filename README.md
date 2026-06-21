# Tanker Synthesis Model — v0.1
### INDOPACOM Distributed Logistics Product Tanker Design Family
**ABS SVR / Jones Act | 100–200 m LOA | Autonomous Operations**

---

## Architecture

```
tanker_synthesis/
├── synthesis.py               ← Main runner / CLI
├── modules/
│   ├── design_point.py        ← Shared parameter vector (DesignPoint dataclass)
│   ├── resistance_holtrop.py  ← Holtrop-Mennen 1984 + propulsive efficiency
│   ├── weights_watson.py      ← Watson/Barrass parametric weight estimate
│   ├── stability_prelim.py    ← KB, BM, GM, freeboard preliminary check
│   └── cost_parametric.py     ← Jones Act 2nd-tier yard acquisition + OPEX
├── outputs/                   ← CSV exports, plots
└── tests/                     ← (next: validation against reference vessels)
```

## Design philosophy

This model follows the HOLISHIP / OpenMDAO approach:
- **Single control vector** (DesignPoint) flows through all discipline modules
- Each module is callable independently (no hidden coupling)
- OpenMDAO ExplicitComponent wrappers provided for gradient-based MDO
- Surrogate-ready: replace any module with a trained RSM without changing interfaces

## Usage

```bash
# Run baseline three-point family (T120 / T150 / T185)
python synthesis.py

# Single design point
python synthesis.py --loa 150

# Full 100–200m LOA sweep (21 points)
python synthesis.py --sweep

# Speed-power survey at 165m
python synthesis.py --speed-survey 165
```

## Python API

```python
from modules.design_point import DesignPoint, HullParams, MissionParams
from modules import compute_resistance, compute_weights, compute_stability, compute_cost

dp = DesignPoint(name="my_tanker")
dp.hull    = HullParams(LOA=155, LBP=150, B=24, D=13, T=9, Cb=0.78)
dp.mission = MissionParams(Vs_kn=14.0, range_nm=8000, crew_target=8)

compute_resistance(dp)
compute_weights(dp)
compute_stability(dp)
compute_cost(dp)
print(dp.summary())
```

## Module status

| Module | Method | Validation status |
|---|---|---|
| Resistance | Holtrop-Mennen 1984 | Pending: validate vs Deltamarin B.Delta model test |
| Propulsive eff. | Harvald wake fraction | Preliminary |
| Weights | Watson/Barrass regression | Pending: validate vs T150 NASSCO reference |
| Stability | Morrish KB + BM from Cwp | Screening only; full NAPA book TBD |
| Cost | Parametric Jones Act 2nd-tier | Calibrated to NASSCO/Philly yard public data |

## Roadmap (next modules)

- `structural_abs.py` — ABS SVR scantling rule checker (automated rule compliance)
- `arrangements.py` — cargo tank arrangement, house/bridge sizing, UNREP station
- `openmdao_problem.py` — full MDO problem with NSGA-II multi-objective optimizer
- `surrogate_rsm.py` — response surface model trainer (replaces expensive CFD calls)
- `hullform_caeses.py` — CAESES API interface for parametric hull geometry

## Key references

- Holtrop & Mennen (1984), ISP 29(335)
- Watson (1998) *Practical Ship Design*, Elsevier
- Papanikolaou (2010) "Holistic Ship Design Optimization", CAD 42(11)
- HOLISHIP H2020 project (2016–2020), Springer
- ABS Steel Vessel Rules (2024), Part 5C Tankers
- MARPOL Annex I, SOLAS II-1, Load Line Convention
