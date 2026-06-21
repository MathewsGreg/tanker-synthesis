"""
mission_envelope.py
===================
Mission envelope and operational requirements for the INDOPACOM
distributed logistics tanker family.

Establishes:
  1. Design range requirement (Pearl Harbor → Guam = 3,500 nm, Mahan's yardstick)
  2. Minimum speed requirement (15 knots — convoy compatible, surge capable)
  3. Froude Number feasibility envelope per ship size
  4. Speed-size constraints from Fn physics
  5. Family speed assignments (which sizes can meet which speeds economically)

Key findings embedded in this module:
  - Ships below ~130m LOA cannot meet 15 kn in displacement regime (Fn > 0.25)
  - T100/T110/T120 class: design speed 13.0–13.5 kn (Fn ≤ 0.22)
  - T130+ class: 15.0 kn feasible in displacement regime
  - T150+ class: 15.0 kn in comfortable displacement regime (Fn ≤ 0.21)
  - All sizes achieve 3,500 nm range — fuel load scales accordingly

Greg Tanker Synthesis Model — v0.1
"""

from dataclasses import dataclass
from typing import Dict

# ---------------------------------------------------------------------------
# Strategic distances (nautical miles)
# ---------------------------------------------------------------------------
DISTANCES_NM = {
    "pearl_harbor_guam":         3_500,   # Mahan's standard distance — primary design range
    "pearl_harbor_philippines":  5_100,   # via Guam route
    "guam_philippines":          1_600,
    "guam_japan_yokosuka":       1_400,
    "guam_diego_garcia":         3_900,
    "san_diego_pearl_harbor":    2_230,
    "pearl_harbor_wake_island":    950,
    "wake_island_guam":          1_510,
}

PRIMARY_RANGE_NM    = 3_500   # Pearl Harbor → Guam
DESIGN_RANGE_NM     = 4_500   # nm — Pearl Harbor to Guam + 30% margin   # With 30% weather/diversion margin added
RANGE_RESERVE_FRAC  = 0.10    # 10% fuel reserve (regulatory + operational)
WEATHER_MARGIN_FRAC = 0.15    # 15% additional fuel for adverse conditions

# ---------------------------------------------------------------------------
# Froude Number thresholds
# ---------------------------------------------------------------------------
FN_THRESHOLDS = {
    "displacement_comfortable":  0.20,   # well below hump — excellent efficiency
    "displacement_upper":        0.22,   # Holtrop-Mennen reliability limit
    "hump_entry":                0.25,   # wave-making resistance rises steeply
    "hump_peak":                 0.27,   # maximum resistance peak
    "semi_displacement":         0.30,   # hull begins to rise — not a tanker anymore
}

# CR (residuary resistance coefficient) lookup for Cb=0.78 tankers
# Interpolated from SNAME/DTMB systematic series data (Carlton, Watson)
_FN_PTS = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30]
_CR_PTS = [0.001,0.002,0.003,0.005,0.007,0.012,0.025,0.045,0.070,0.060,0.050]

def cr_residuary(Fn: float, Cb: float = 0.78) -> float:
    """Residuary resistance coefficient for tanker hull forms."""
    import numpy as np
    fn = max(0.10, min(0.30, Fn))
    cr_base = float(np.interp(fn, _FN_PTS, _CR_PTS))
    cb_factor = 1.0 + 2.0 * (Cb - 0.72)
    return cr_base * cb_factor


# ---------------------------------------------------------------------------
# Family speed assignments
# ---------------------------------------------------------------------------
@dataclass
class SpeedSpec:
    """Speed requirements for a ship of given LOA."""
    loa: float
    vs_design: float       # design (service) speed [kn]
    vs_max_fn: float       # max speed in displacement regime [kn]
    fn_at_design: float    # Froude number at design speed
    fn_assessment: str     # qualitative assessment
    range_nm: float        # design range requirement [nm]
    meets_15kn: bool       # True if 15 kn in displacement regime
    meets_hawaii_guam: bool # True if range reaches 3500nm at design speed


def fn_at_speed(lbp: float, vs_kn: float) -> float:
    import math
    return vs_kn * 0.5144 / math.sqrt(9.81 * lbp)


def max_displacement_speed(lbp: float, fn_limit: float = 0.22) -> float:
    """Maximum speed in displacement regime for given LBP."""
    import math
    return fn_limit * math.sqrt(9.81 * lbp) / 0.5144


def lbp_for_fn_at_speed(vs_kn: float, fn_target: float) -> float:
    """Minimum LBP to achieve vs_kn at fn_target."""
    return (vs_kn * 0.5144 / fn_target) ** 2 / 9.81


# Family standard speed assignments
# -----------------------------------------------------------------------
# REVISED FAMILY: 120m – 200m LOA
# All ships: 15.0 kn design speed, 4,500 nm range, MAN 32/44CR engine
#
# Decision rationale:
#   - Ships below 120m at 15 kn approach hump region (Fn > 0.25) — fuel
#     penalty and engine oversizing make them uneconomical for INDOPACOM
#   - T120 at Fn=0.229 is the marginal lower bound — still in displacement
#     regime, Holtrop-Mennen valid, wave resistance manageable (~2.5% Rf)
#   - Single design speed across the family: simpler crew qualification,
#     convoy-compatible, consistent with Navy escort requirements
#   - Single engine type (MAN 32/44CR) across all 7 hull sizes:
#     one spare parts inventory, one PrimeServ contract, one MCR rating
# -----------------------------------------------------------------------
FAMILY_LOA_MIN    = 120.0    # m — lower bound of design family
FAMILY_LOA_MAX    = 200.0    # m — upper bound of design family
FAMILY_VS_KN      = 15.0     # kn — standard design speed (all sizes)
FAMILY_MAIN_ENGINE = "L32/44CR"   # MAN engine model for all main engines

FAMILY_SPEED_ASSIGNMENTS = {
    # Single tier — all ships in family at 15 kn
    (120, 200):  (15.0, "Full displacement regime — Fn 0.177–0.229 across family"),
}


def get_design_speed(loa: float) -> float:
    for (lo, hi), (spd, _) in FAMILY_SPEED_ASSIGNMENTS.items():
        if lo <= loa <= hi:
            return spd
    return 15.0


def assess_fn(fn: float) -> str:
    if fn > FN_THRESHOLDS["hump_peak"]:
        return "CRITICAL — past hump peak, semi-displacement"
    elif fn > FN_THRESHOLDS["hump_entry"]:
        return "HUMP — severe resistance rise, avoid"
    elif fn > FN_THRESHOLDS["displacement_upper"]:
        return "HIGH — above reliable Holtrop-Mennen range"
    elif fn > FN_THRESHOLDS["displacement_comfortable"]:
        return "ACCEPTABLE — upper displacement"
    else:
        return "GOOD — full displacement regime"


def build_speed_spec(loa: float, vs_design: float = None) -> SpeedSpec:
    import math
    lbp = loa * 0.967
    if vs_design is None:
        vs_design = get_design_speed(loa)
    fn_design = fn_at_speed(lbp, vs_design)
    vs_fn22   = max_displacement_speed(lbp, 0.22)
    days_3500 = PRIMARY_RANGE_NM / (vs_design * 24)
    return SpeedSpec(
        loa=loa,
        vs_design=vs_design,
        vs_max_fn=round(vs_fn22, 1),
        fn_at_design=round(fn_design, 4),
        fn_assessment=assess_fn(fn_design),
        range_nm=DESIGN_RANGE_NM,
        meets_15kn=vs_design >= 15.0,
        meets_hawaii_guam=(vs_design > 0)  # all sizes reach 3500nm given enough fuel
    )


# ---------------------------------------------------------------------------
# Engine family standardization
# ---------------------------------------------------------------------------
"""
BUSINESS DECISION: Standardize on MAN Energy Solutions (Everllence) portfolio.

Rationale:
  1. MAN 32/44CR is the current reference engine for 25–50k DWT product tankers
     (GTS 41k DWT tankers, 4 × MAN 10L32/44CR, delivered 2025)
  2. MAN L27/38 covers the lower power range (T100–T130 class, 2,100–3,690 kW)
     explicitly marketed for 'small to medium-sized tankers' — exact match
  3. MAN L23/30H covers auxiliary gensets across entire family — one service network
  4. Single PrimeServ global service network — critical for INDOPACOM logistics
     (Guam, Japan, Philippines, Singapore, Pearl Harbor all have MAN coverage)
  5. Common spare parts, training, and maintenance procedures across fleet
  6. MAN tools (CEAS configurator) directly output scantling-compatible data

MAN Marine Four-Stroke Family — full portfolio for this design:
  L23/30H Mk3  — 200 kW/cyl, 900 rpm, 6–9 cyl → 1,200–1,800 kW [AUXILIARY only]
  L27/38       — 350 kW/cyl, 750 rpm, 6–9 cyl → 2,100–3,150 kW [T100–T130 main]
  L32/44CR     — 600 kW/cyl, 750 rpm, 6–16 cyl→ 3,600–9,600 kW [T140–T200 main]

Architecture options:
  Option A — Single main engine (T100–T150): simplest, lowest weight/cost
  Option B — Twin main engines (T165+): redundancy, N-1 capability at reduced speed
  All ships: 3 × L23/30H aux gensets (N+1 redundancy, one hot standby)

Note on rebranding: MAN Energy Solutions renamed to 'Everllence' in June 2025,
but the engine portfolio, part numbers, and service network are unchanged.
The synthesis model uses 'MAN' as the historical/common reference.
"""

@dataclass
class MANEngine:
    model: str
    kw_per_cyl: float
    rpm: int
    cyl_options: list      # valid cylinder counts
    sfc_100: float         # g/kWh at 100% MCR
    sfc_85:  float         # g/kWh at 85% MCR (service)
    sfc_75:  float         # g/kWh at 75%
    sfc_50:  float         # g/kWh at 50%
    mass_per_cyl_t: float  # tonnes per cylinder
    tier3_native: bool
    role: str              # main | aux | both

MAN_ENGINE_FAMILY = {
    "L23/30H": MANEngine(
        model="L23/30H Mk3", kw_per_cyl=200, rpm=900,
        cyl_options=[6, 7, 8, 9],
        sfc_100=185, sfc_85=178, sfc_75=180, sfc_50=196,
        mass_per_cyl_t=0.90, tier3_native=False,
        role="aux",
    ),
    "L27/38": MANEngine(
        model="L27/38", kw_per_cyl=350, rpm=750,
        cyl_options=[6, 7, 8, 9],
        sfc_100=172, sfc_85=162, sfc_75=165, sfc_50=182,
        mass_per_cyl_t=1.55, tier3_native=False,
        role="main",
    ),
    "L32/44CR": MANEngine(
        model="32/44CR", kw_per_cyl=600, rpm=750,
        cyl_options=[6, 7, 8, 9, 10, 12, 16],
        sfc_100=163, sfc_85=153, sfc_75=158, sfc_50=173,
        mass_per_cyl_t=2.06, tier3_native=True,
        role="main",
    ),
}

AUX_GENSET = MAN_ENGINE_FAMILY["L23/30H"]


def select_man_engine(pb_required_kw: float,
                      min_margin_pct: float = 15.0) -> Dict:
    """
    Select the best MAN engine configuration for required brake power.
    Returns dict with engine model, cylinder count, n_engines, MCR, margin.

    Strategy:
      1. Try L32/44CR single engine (preferred for T140+)
      2. Try L27/38 single engine (preferred for T100–T130)
      3. Try twin engines if single exceeds 16 cylinders
      4. Always maintain minimum MCR margin
    """
    mcr_needed = pb_required_kw / (1.0 - min_margin_pct / 100.0)

    results = []
    for key, eng in MAN_ENGINE_FAMILY.items():
        if eng.role == "aux":
            continue
        for n_eng in [1, 2]:
            mcr_per_eng = mcr_needed / n_eng
            kw_per_eng  = mcr_per_eng
            n_cyl = kw_per_eng / eng.kw_per_cyl
            # Round to nearest valid cylinder count
            valid = [c for c in eng.cyl_options if c * eng.kw_per_cyl >= kw_per_eng * 0.85]
            if not valid:
                continue
            n_cyl_selected = min(valid)
            mcr_each = n_cyl_selected * eng.kw_per_cyl
            mcr_total = mcr_each * n_eng
            margin_pct = (mcr_total - pb_required_kw) / pb_required_kw * 100
            if margin_pct < min_margin_pct - 2:
                continue
            mass_t = n_cyl_selected * eng.mass_per_cyl_t * n_eng
            results.append({
                "model":       eng.model,
                "key":         key,
                "n_cyl":       n_cyl_selected,
                "n_engines":   n_eng,
                "mcr_each_kw": mcr_each,
                "mcr_total_kw":mcr_total,
                "margin_pct":  round(margin_pct, 1),
                "sfc_85":      eng.sfc_85,
                "rpm":         eng.rpm,
                "mass_t":      round(mass_t, 1),
                "tier3":       eng.tier3_native,
                "label":       f"{n_eng}×MAN {n_cyl_selected}L{eng.model}" if n_eng > 1
                               else f"MAN {n_cyl_selected}L{eng.model}",
            })

    if not results:
        return {}

    # Prefer: single engine L32/44CR > single L27/38 > twin engines
    # Sort by: n_engines ASC, then prefer L32/44CR, then smallest margin
    def sort_key(r):
        model_pref = 0 if "32/44CR" in r["model"] else 1
        return (r["n_engines"], model_pref, r["margin_pct"])

    results.sort(key=sort_key)
    return results[0]


def select_aux_gensets(elec_load_kw: float, n_sets: int = 3) -> Dict:
    """Select auxiliary gensets. Standard: 3 × L23/30H (N+1 arrangement)."""
    eng = AUX_GENSET
    # 2 running at full load, 1 standby → each running set at ~50% load max
    load_per_running = elec_load_kw / (n_sets - 1)
    target_kw = load_per_running / 0.75   # run at 75% MCR
    n_cyl = max(6, min(9, round(target_kw / eng.kw_per_cyl + 0.5)))
    kw_each = n_cyl * eng.kw_per_cyl
    return {
        "model":   eng.model,
        "n_cyl":   n_cyl,
        "n_sets":  n_sets,
        "kw_each": kw_each,
        "kw_total":kw_each * n_sets,
        "label":   f"3 × MAN {n_cyl}L{eng.model}",
    }
