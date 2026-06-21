"""
structural_abs.py
=================
ABS structural scantlings module for double-hull product tankers.

Rule basis:
  ABS Rules for Building and Classing Marine Vessels 2024
  Part 5C Chapter 1  — Vessels Carrying Oil in Bulk (≥150m)
  Part 5C Chapter 2  — Vessels Carrying Oil in Bulk (<150m)
  Part 3             — Hull Construction and Equipment (general)
  Common Structural Rules (CSR) for Double Hull Oil Tankers (LR/ABS/DNV 2004)
                      — incorporated into ABS 5C-1 from July 2005

Design philosophy:
  - All dimensions on NET scantling basis (ABS 5C-1-2/3)
  - Nominal Design Corrosion Values (NDCV) added to give built-up (gross) dimensions
  - Material: ABS Grade AH32 (yield σ_y = 315 N/mm²) for high-stress locations,
              ABS Grade A (σ_y = 235 N/mm²) as default
  - Framing: longitudinal (Isherwood system) throughout — standard for tankers
  - Stiffener preference: bulb flats (HP sections) — specified by owner
    Buckling coefficient C = 37 for web depth/thickness (CSR Table 10.2.1)
  - Frame spacing: 800 mm longitudinal (s), 3,200 mm transverse web spacing (ℓ)
  - Double hull: MARPOL Reg 19 double hull width Wh, double bottom height h_db

Key formulas implemented:
  Plating:
    t_net = Cs × s × √(p / σ_y)       [ABS 5C-1-4-7.3, 9.1, 9.3]
    where Cs = 0.73 (bottom/deck), 0.7 (side), 0.66 (bulkhead)
    p = design pressure at midpoint of panel

  Longitudinal stiffeners (bulb flat HP section):
    SM_req = Cs × h × s × ℓ²          [ABS 5C-1-4-7.5, 9.5]
    where SM is section modulus [cm³], h = design head [m], s = spacing [m],
    ℓ = unsupported span [m], Cs = 7.8 cm³ per unit

  Bulb flat web proportions:
    d_w / t_w ≤ C × √(235/σ_y)        [CSR Table 10.2.1]
    C = 37 for bulb profiles
    → d_w / t_w ≤ 37 × √(235/σ_y)

  Hull girder section modulus:
    SM_required = C1 × C2 × L² × B × (Cb + 0.7) / Q  [ABS 3-2-1]
    where Q = material factor (1.0 for Grade A, 0.78 for AH32)

NDCV (Nominal Design Corrosion Values) per ABS 5C-1-2, Table 1:
  Outer bottom:           2.0 mm
  Inner bottom (cargo):   2.0 mm
  Deck plating:           1.5 mm
  Side shell:             2.0 mm
  Double hull voids:      1.5 mm
  Cargo tank boundary:    2.0 mm
  Ballast tank:           2.5 mm

Greg Tanker Synthesis Model — v0.1
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Material constants
# ---------------------------------------------------------------------------

GRADE_A   = {"name": "ABS Grade A",   "fy": 235.0, "fu": 400.0, "Q": 1.00, "E": 206000.0}
GRADE_AH32 = {"name": "ABS Grade AH32","fy": 315.0, "fu": 440.0, "Q": 0.78, "E": 206000.0}
GRADE_AH36 = {"name": "ABS Grade AH36","fy": 355.0, "fu": 490.0, "Q": 0.72, "E": 206000.0}

# Default material assignments for tanker zones
ZONE_MATERIALS = {
    "bottom_outer":       GRADE_AH32,   # keel + bottom — high girder stress
    "inner_bottom":       GRADE_A,      # inner bottom — lower stress
    "side_outer":         GRADE_A,      # side shell
    "deck":               GRADE_AH32,   # strength deck — high girder stress
    "lng_bhd":            GRADE_A,      # longitudinal bulkhead
    "trans_bhd":          GRADE_A,      # transverse bulkhead
    "double_hull_void":   GRADE_A,      # double hull voids
    "girder":             GRADE_AH32,   # web frames and girders
}

# Nominal Design Corrosion Values (mm) — ABS 5C-1-2/3, Table 1
NDCV = {
    "bottom_outer":       2.0,
    "inner_bottom_cargo": 2.0,
    "deck":               1.5,
    "side_outer":         2.0,
    "double_hull":        1.5,
    "cargo_boundary":     2.0,
    "ballast_tank":       2.5,
    "lng_bhd":            2.0,
}


# ---------------------------------------------------------------------------
# Bulb flat (HP profile) standard section library
# European bulb flat HP sections (EN 10067) — standard for shipbuilding
# Columns: (designation, height h_mm, tw_mm, A_cm2, I_cm4, SM_cm3, mass_kg_m)
# ---------------------------------------------------------------------------

HP_SECTIONS = [
    # (name,   h,   tw,   A,     I,      SM,    mass)
    ("HP120×7", 120,  7.0,  10.16,  218.0,  28.3,  7.97),
    ("HP140×7", 140,  7.0,  11.93,  361.0,  40.2,  9.37),
    ("HP160×8", 160,  8.0,  15.30,  621.0,  59.0, 12.01),
    ("HP180×8", 180,  8.0,  17.00, 902.0,   77.4, 13.35),
    ("HP200×9", 200,  9.0,  21.24, 1376.0, 104.0, 16.67),
    ("HP220×10",220, 10.0,  25.87, 2003.0, 138.0, 20.31),
    ("HP240×11",240, 11.0,  30.80, 2827.0, 178.0, 24.18),
    ("HP260×12",260, 12.0,  36.45, 3870.0, 224.0, 28.62),
    ("HP300×13",300, 13.0,  46.67, 7006.0, 349.0, 36.64),
    ("HP320×13",320, 13.0,  48.30, 8520.0, 398.0, 37.93),
    ("HP340×14",340, 14.0,  56.00,11100.0, 480.0, 43.96),
    ("HP370×14",370, 14.0,  59.60,14500.0, 583.0, 46.79),
    ("HP400×16",400, 16.0,  74.40,20700.0, 773.0, 58.40),
    ("HP430×17",430, 17.0,  85.00,27200.0, 952.0, 66.70),
]


def select_hp_section(sm_req_cm3: float) -> dict:
    """
    Select the lightest HP (bulb flat) section that meets the required section modulus.
    Returns dict with section properties.
    """
    for name, h, tw, A, I, SM, mass in HP_SECTIONS:
        if SM >= sm_req_cm3:
            return {
                "designation": name,
                "h_mm": h,
                "tw_mm": tw,
                "A_cm2": A,
                "I_cm4": I,
                "SM_cm3": SM,
                "mass_kg_m": mass,
            }
    # Larger than catalogue — use heaviest
    name, h, tw, A, I, SM, mass = HP_SECTIONS[-1]
    return {"designation": name+"*OVR", "h_mm": h, "tw_mm": tw,
            "A_cm2": A, "I_cm4": I, "SM_cm3": SM, "mass_kg_m": mass}


def check_hp_proportions(h_mm: float, tw_mm: float, fy: float = 235.0) -> Tuple[float, bool]:
    """
    Check bulb flat web depth/thickness ratio per CSR Table 10.2.1:
    d_w / t_w ≤ 37 × √(235 / σ_y)
    Returns (ratio, passes).
    """
    C = 37.0
    limit = C * math.sqrt(235.0 / fy)
    ratio = h_mm / tw_mm
    return ratio, ratio <= limit


# ---------------------------------------------------------------------------
# Design pressure functions  (ABS 5C-1-3, Section 7)
# ---------------------------------------------------------------------------

def pressure_sea_external(z_m: float, T: float) -> float:
    """
    External sea pressure on outer shell below waterline.
    p = ρ_sw × g × (T - z) where z = height above keel [m]
    Returns pressure in kN/m².
    """
    p = 1.025 * 9.81 * max(T - z_m, 0)
    return max(p, 0.0)


def pressure_cargo_internal(h_cargo_m: float, sg: float = 0.9) -> float:
    """
    Cargo hydrostatic pressure at base of full cargo tank.
    p = ρ_cargo × g × h  where h = tank height from the point to the top [m]
    sg = specific gravity of cargo (0.9 for typical product)
    Returns pressure in kN/m².
    """
    return sg * 9.81 * h_cargo_m


def pressure_ballast_internal(h_ballast_m: float) -> float:
    """
    Ballast water pressure in double hull / double bottom tanks.
    p = 1.025 × g × h_ballast  [kN/m²]
    """
    return 1.025 * 9.81 * h_ballast_m


def design_pressure_bottom(T: float, h_tank: float, sg_cargo: float = 0.9) -> float:
    """
    Design pressure for bottom shell and inner bottom plating.
    Governing: max of external sea pressure at keel or internal cargo head.
    ABS 5C-1-3/7.7
    """
    p_ext = pressure_sea_external(0.0, T)        # at keel
    p_int = pressure_cargo_internal(h_tank, sg_cargo)
    return max(p_ext, p_int)


def design_pressure_side(z: float, T: float, h_tank: float,
                          sg_cargo: float = 0.9) -> float:
    """
    Design pressure for side shell at height z above keel.
    ABS 5C-1-3/7.7 — governing of external and internal.
    """
    p_ext = pressure_sea_external(z, T)
    p_int = pressure_cargo_internal(max(h_tank - z, 0), sg_cargo)
    return max(p_ext, p_int)


# ---------------------------------------------------------------------------
# Plate thickness  (ABS 5C-1-4, Sections 7.3 and 9.1)
# ---------------------------------------------------------------------------

# Cs coefficients for plate thickness formula t = Cs × s × √(p / fy)
CS_PLATE = {
    "bottom":        0.73,   # 5C-1-4/7.3
    "inner_bottom":  0.73,
    "deck":          0.73,
    "side":          0.70,
    "lng_bhd":       0.66,
    "trans_bhd":     0.66,
    "double_hull":   0.66,
}

# ABS absolute minimums (mm) for each zone
T_MIN = {
    "bottom":        6.0,
    "inner_bottom":  6.0,
    "deck":          6.0,
    "side":          5.5,
    "lng_bhd":       5.0,
    "trans_bhd":     5.0,
    "double_hull":   5.0,
    "keel":          8.0,
}


def net_plate_thickness(zone: str, p_kNm2: float, s_mm: float,
                         mat: dict = None) -> float:
    """
    Net plate thickness from ABS 5C-1 formula:
    t_net = Cs × s × √(p / σ_y)   [mm]
    where s in mm, p in N/mm² (= kN/m²/1000 * 1000 = kN/m² × 1e-3 → N/mm²)
    Actually: 1 kN/m² = 0.001 N/mm²
    """
    if mat is None:
        mat = GRADE_A
    Cs = CS_PLATE.get(zone, 0.70)
    fy = mat["fy"]  # N/mm²
    p_Nmm2 = p_kNm2 * 1e-3   # convert kN/m² → N/mm²
    if p_Nmm2 <= 0:
        return T_MIN.get(zone, 5.0)
    t_net = Cs * s_mm * math.sqrt(p_Nmm2 / fy)
    return max(t_net, T_MIN.get(zone, 5.0))


def gross_plate_thickness(t_net: float, zone: str) -> float:
    """Add NDCV corrosion margin to get built-up (gross) thickness."""
    ndcv_map = {
        "bottom":        NDCV["bottom_outer"],
        "inner_bottom":  NDCV["inner_bottom_cargo"],
        "deck":          NDCV["deck"],
        "side":          NDCV["side_outer"],
        "lng_bhd":       NDCV["lng_bhd"],
        "trans_bhd":     NDCV["cargo_boundary"],
        "double_hull":   NDCV["double_hull"],
        "keel":          NDCV["bottom_outer"],
    }
    ca = ndcv_map.get(zone, 2.0)
    # Round up to nearest 0.5 mm (standard mill increment)
    t_gross = t_net + ca
    return math.ceil(t_gross * 2) / 2.0   # round up to nearest 0.5 mm


# ---------------------------------------------------------------------------
# Longitudinal stiffener (bulb flat) section modulus  (ABS 5C-1-4/7.5, 9.5)
# ---------------------------------------------------------------------------

CS_LONG = {
    "bottom":       7.8,   # ABS 5C-1-4/7.5 coefficient
    "inner_bottom": 7.8,
    "deck":         7.8,
    "side":         7.8,
    "lng_bhd":      6.5,
    "trans_bhd":    6.5,
}


def req_section_modulus_longitudinal(zone: str, p_kNm2: float,
                                     s_m: float, l_m: float,
                                     mat: dict = None) -> float:
    """
    Required section modulus of longitudinal stiffener (including attached plate).
    SM_req = Cs × h × s × ℓ²   [cm³]

    where:
      Cs  = rule coefficient (7.8 for plating / 6.5 for bulkheads)
      h   = design pressure head [m] = p [kN/m²] / (ρ_sw × g) for sea,
            or p [kN/m²] / (ρ_cargo × g) for cargo.
            More directly: SM = Cs × p_kNm2/9.81 × s × l²   (simplified)
      s   = stiffener spacing [m]
      ℓ   = unsupported span [m]

    ABS 5C-1-4-7.5(a):
      SM = Cs × p × s × ℓ² / σ_y  (general form)
    Here: using standard Cs which already encodes σ_y for Grade A.
    For other grades, apply material factor Q.
    """
    if mat is None:
        mat = GRADE_A
    Cs = CS_LONG.get(zone, 7.8)
    Q  = mat.get("Q", 1.0)
    # h = effective head in metres = p / (rho_sw * g) ≈ p / 10.05
    h = p_kNm2 / (1.025 * 9.81)
    sm_req = Cs * h * s_m * l_m ** 2 / Q
    return max(sm_req, 5.0)   # minimum 5 cm³


# ---------------------------------------------------------------------------
# Hull girder section modulus  (ABS Part 3-2-1)
# ---------------------------------------------------------------------------

def req_hull_section_modulus(L: float, B: float, Cb: float,
                              mat_deck: dict = None,
                              C1_table: float = None) -> Dict:
    """
    Required midship hull girder section modulus (sagging + hogging).
    ABS Part 3-2-1/3.1:

    SM_req = C1 × C2 × L² × B × (Cb + 0.7) / Q   [cm² × m]

    where:
      C1 = wave coefficient = 0.0792L for L < 90m,
           7.179(1 + L/232)^0.5 for 90 ≤ L ≤ 300m
      C2 = 0.01 (for tankers with no deck opening amidships)
      Q  = material factor for deck plate

    Also apply the minimum:
      SM ≥ 0.95 C1 × C2 × L² × B × Cb
    """
    if mat_deck is None:
        mat_deck = GRADE_AH32
    Q = mat_deck.get("Q", 1.0)

    # C1 wave coefficient
    if L < 90:
        C1 = 0.0792 * L
    elif L <= 300:
        C1 = 7.179 * math.sqrt(1 + L / 232)
    else:
        C1 = 10.75 - ((300 - L) / 100) ** 1.5  # rarely used

    if C1_table is not None:
        C1 = C1_table

    C2 = 0.01

    SM_sag = C1 * C2 * L**2 * B * (Cb + 0.7) / Q
    SM_hog = 0.78 * SM_sag   # hogging = 0.78 × sagging (ABS 3-2-1/3.3)

    return {
        "C1":     round(C1, 4),
        "C2":     C2,
        "SM_sag_cm2m": round(SM_sag, 0),
        "SM_hog_cm2m": round(SM_hog, 0),
    }


# ---------------------------------------------------------------------------
# MARPOL double hull dimensions  (MARPOL Annex I Reg 19, for L < 200m)
# ---------------------------------------------------------------------------

def marpol_double_hull(L: float, B: float, T: float) -> Dict:
    """
    MARPOL Reg 19 minimum double hull and double bottom dimensions.
    For vessels 120m ≤ L < 200m:
      W_h = max(0.76m, B/15)    double hull width [m]
      h_db = max(0.76m, B/15)   double bottom height [m]  (same formula for <200m)
    Minimum widths capped at 2.0m per ABS guidance.
    """
    Wh  = max(0.76, B / 15.0)
    Wh  = min(Wh, 2.0)
    hdb = max(T / 15.0, 0.76)
    hdb = min(hdb, T * 0.20)

    # Minimum clear height for maintenance access in double bottom
    hdb_access = max(hdb, 0.80)   # ABS 5C-1-1/5.21 min 800mm

    return {
        "Wh_m":        round(Wh, 3),
        "hdb_m":       round(hdb, 3),
        "hdb_access_m":round(hdb_access, 3),
        "note":        f"MARPOL Reg 19 · Wh = max(0.76, B/15) = {Wh:.3f}m, "
                       f"hdb = max(T/15, 0.76) = {hdb:.3f}m",
    }


# ---------------------------------------------------------------------------
# Full midship section scantling calculation
# ---------------------------------------------------------------------------

@dataclass
class PanelScantling:
    """Scantling result for one structural panel zone."""
    zone:           str
    location:       str       # descriptive label
    material:       str       # grade name
    p_kNm2:         float     # design pressure
    t_net_mm:       float     # required net plate thickness
    t_gross_mm:     float     # with NDCV corrosion margin
    t_adopted_mm:   float     # rounded up to nearest 0.5mm
    stiffener:      str       # HP section designation
    sm_req_cm3:     float     # required SM
    sm_provided_cm3:float     # provided SM
    hp_check_ok:    bool      # proportions check passes
    hp_dw_tw_ratio: float     # actual d/t ratio
    hp_dw_tw_limit: float     # limit d/t ratio


@dataclass
class MidshipSection:
    """Complete midship section scantling result."""
    # Principal dimensions used
    L: float; B: float; D: float; T: float; Cb: float
    s_mm: float      # longitudinal frame spacing
    l_web_m: float   # web frame spacing (unsupported span)

    # Double hull geometry
    Wh_m: float      # double hull width
    hdb_m: float     # double bottom height
    y_inner: float   # inner hull half-breadth from CL

    # Panel scantlings
    panels: List[PanelScantling] = field(default_factory=list)

    # Hull girder
    SM_sag_cm2m: float = 0.0
    SM_hog_cm2m: float = 0.0
    SM_actual_cm2m: float = 0.0   # computed from section

    # Steel weight (parametric from scantlings)
    W_steel_t: float = 0.0

    # Structural VCG (from mass-weighted centroids)
    VCG_struct_m: float = 0.0

    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main scantling solver
# ---------------------------------------------------------------------------

class MidshipScantlingSolver:

    def __init__(self, s_mm: float = 800.0, l_web_m: float = 3.2):
        self.s_mm    = s_mm
        self.s_m     = s_mm / 1000
        self.l_web_m = l_web_m

    def solve(self, L: float, B: float, D: float, T: float,
              Cb: float, sg_cargo: float = 0.9,
              n_cargo_tiers: int = 1) -> MidshipSection:
        """
        Compute all midship section scantlings from first principles.

        Parameters
        ----------
        sg_cargo  : specific gravity of design cargo (0.9 = product tanker)
        n_cargo_tiers : number of stacked cargo tank levels (1 = single deck)
        """
        dh  = marpol_double_hull(L, B, T)
        Wh  = dh["Wh_m"]
        hdb = dh["hdb_m"]
        y_inner = B/2 - Wh   # inner hull half-breadth

        # Cargo tank height (inner bottom to deck)
        h_cargo = D - hdb - 0.05   # subtract deck plate ~50mm

        ms = MidshipSection(
            L=L, B=B, D=D, T=T, Cb=Cb,
            s_mm=self.s_mm, l_web_m=self.l_web_m,
            Wh_m=Wh, hdb_m=hdb, y_inner=y_inner,
        )

        panels = []

        # ---- ZONE 1: Outer bottom shell ----
        p1 = design_pressure_bottom(T, h_cargo, sg_cargo)
        mat1 = ZONE_MATERIALS["bottom_outer"]
        t1n = net_plate_thickness("bottom", p1, self.s_mm, mat1)
        t1g = gross_plate_thickness(t1n, "bottom")
        sm1 = req_section_modulus_longitudinal("bottom", p1, self.s_m, self.l_web_m, mat1)
        hp1 = select_hp_section(sm1)
        r1, ok1 = check_hp_proportions(hp1["h_mm"], hp1["tw_mm"], mat1["fy"])
        lim1 = 37 * math.sqrt(235/mat1["fy"])
        panels.append(PanelScantling(
            zone="bottom", location="Outer bottom shell (keel to bilge)",
            material=mat1["name"], p_kNm2=round(p1,2),
            t_net_mm=round(t1n,2), t_gross_mm=round(t1g,2),
            t_adopted_mm=round(math.ceil(t1g*2)/2, 1),
            stiffener=hp1["designation"], sm_req_cm3=round(sm1,1),
            sm_provided_cm3=hp1["SM_cm3"], hp_check_ok=ok1,
            hp_dw_tw_ratio=round(r1,1), hp_dw_tw_limit=round(lim1,1)
        ))

        # ---- ZONE 2: Inner bottom plating ----
        p2 = pressure_cargo_internal(h_cargo, sg_cargo)   # full cargo head from below
        p2_ballast = pressure_ballast_internal(hdb)        # ballast from DB side
        p2_gov = max(p2, p2_ballast)
        mat2 = ZONE_MATERIALS["inner_bottom"]
        t2n = net_plate_thickness("inner_bottom", p2_gov, self.s_mm, mat2)
        t2g = gross_plate_thickness(t2n, "inner_bottom")
        sm2 = req_section_modulus_longitudinal("inner_bottom", p2_gov, self.s_m, self.l_web_m, mat2)
        hp2 = select_hp_section(sm2)
        r2, ok2 = check_hp_proportions(hp2["h_mm"], hp2["tw_mm"], mat2["fy"])
        lim2 = 37 * math.sqrt(235/mat2["fy"])
        panels.append(PanelScantling(
            zone="inner_bottom", location="Inner bottom plating",
            material=mat2["name"], p_kNm2=round(p2_gov,2),
            t_net_mm=round(t2n,2), t_gross_mm=round(t2g,2),
            t_adopted_mm=round(math.ceil(t2g*2)/2, 1),
            stiffener=hp2["designation"], sm_req_cm3=round(sm2,1),
            sm_provided_cm3=hp2["SM_cm3"], hp_check_ok=ok2,
            hp_dw_tw_ratio=round(r2,1), hp_dw_tw_limit=round(lim2,1)
        ))

        # ---- ZONE 3: Double bottom inner structures ----
        # DB longitudinal girder plating — governed by ballast head
        p3 = pressure_ballast_internal(T)   # full T head on outer bottom
        t3n = max(net_plate_thickness("double_hull", p3, self.s_mm, GRADE_A), 7.0)
        t3g = gross_plate_thickness(t3n, "double_hull")
        panels.append(PanelScantling(
            zone="double_hull", location="Double bottom void / ballast structure",
            material=GRADE_A["name"], p_kNm2=round(p3,2),
            t_net_mm=round(t3n,2), t_gross_mm=round(t3g,2),
            t_adopted_mm=round(math.ceil(t3g*2)/2, 1),
            stiffener="(floor plate — no long. stiffener)",
            sm_req_cm3=0.0, sm_provided_cm3=0.0, hp_check_ok=True,
            hp_dw_tw_ratio=0.0, hp_dw_tw_limit=0.0
        ))

        # ---- ZONE 4: Outer side shell ----
        # Design at 1/3 height (most critical combination of ext + int)
        z_side = D / 3.0
        p4 = design_pressure_side(z_side, T, h_cargo, sg_cargo)
        mat4 = ZONE_MATERIALS["side_outer"]
        t4n = net_plate_thickness("side", p4, self.s_mm, mat4)
        t4g = gross_plate_thickness(t4n, "side")
        sm4 = req_section_modulus_longitudinal("side", p4, self.s_m, self.l_web_m, mat4)
        hp4 = select_hp_section(sm4)
        r4, ok4 = check_hp_proportions(hp4["h_mm"], hp4["tw_mm"], mat4["fy"])
        lim4 = 37 * math.sqrt(235/mat4["fy"])
        panels.append(PanelScantling(
            zone="side", location=f"Outer side shell (at z={z_side:.1f}m)",
            material=mat4["name"], p_kNm2=round(p4,2),
            t_net_mm=round(t4n,2), t_gross_mm=round(t4g,2),
            t_adopted_mm=round(math.ceil(t4g*2)/2, 1),
            stiffener=hp4["designation"], sm_req_cm3=round(sm4,1),
            sm_provided_cm3=hp4["SM_cm3"], hp_check_ok=ok4,
            hp_dw_tw_ratio=round(r4,1), hp_dw_tw_limit=round(lim4,1)
        ))

        # ---- ZONE 5: Inner side shell (double hull void outboard face) ----
        # Governed by cargo pressure from inside
        p5 = pressure_cargo_internal(h_cargo * 0.8, sg_cargo)   # 80% tank height as avg
        mat5 = ZONE_MATERIALS["double_hull_void"]
        t5n = net_plate_thickness("double_hull", p5, self.s_mm, mat5)
        t5g = gross_plate_thickness(t5n, "double_hull")
        sm5 = req_section_modulus_longitudinal("lng_bhd", p5, self.s_m, self.l_web_m, mat5)
        hp5 = select_hp_section(sm5)
        r5, ok5 = check_hp_proportions(hp5["h_mm"], hp5["tw_mm"], mat5["fy"])
        lim5 = 37 * math.sqrt(235/mat5["fy"])
        panels.append(PanelScantling(
            zone="lng_bhd", location="Inner hull / lng. bulkhead (cargo face)",
            material=mat5["name"], p_kNm2=round(p5,2),
            t_net_mm=round(t5n,2), t_gross_mm=round(t5g,2),
            t_adopted_mm=round(math.ceil(t5g*2)/2, 1),
            stiffener=hp5["designation"], sm_req_cm3=round(sm5,1),
            sm_provided_cm3=hp5["SM_cm3"], hp_check_ok=ok5,
            hp_dw_tw_ratio=round(r5,1), hp_dw_tw_limit=round(lim5,1)
        ))

        # ---- ZONE 6: Strength deck plating ----
        p6 = 10.0   # minimum 10 kN/m² deck load (ABS Part 3)
        mat6 = ZONE_MATERIALS["deck"]
        t6n = net_plate_thickness("deck", p6, self.s_mm, mat6)
        t6g = gross_plate_thickness(t6n, "deck")
        sm6 = req_section_modulus_longitudinal("deck", p6, self.s_m, self.l_web_m, mat6)
        hp6 = select_hp_section(sm6)
        r6, ok6 = check_hp_proportions(hp6["h_mm"], hp6["tw_mm"], mat6["fy"])
        lim6 = 37 * math.sqrt(235/mat6["fy"])
        panels.append(PanelScantling(
            zone="deck", location="Strength deck plating",
            material=mat6["name"], p_kNm2=round(p6,2),
            t_net_mm=round(t6n,2), t_gross_mm=round(t6g,2),
            t_adopted_mm=round(math.ceil(t6g*2)/2, 1),
            stiffener=hp6["designation"], sm_req_cm3=round(sm6,1),
            sm_provided_cm3=hp6["SM_cm3"], hp_check_ok=ok6,
            hp_dw_tw_ratio=round(r6,1), hp_dw_tw_limit=round(lim6,1)
        ))

        # ---- ZONE 7: Longitudinal bulkhead (centerline) ----
        p7 = pressure_cargo_internal(h_cargo, sg_cargo)   # full cargo head
        mat7 = GRADE_A
        t7n = net_plate_thickness("lng_bhd", p7, self.s_mm, mat7)
        t7g = gross_plate_thickness(t7n, "lng_bhd")
        sm7 = req_section_modulus_longitudinal("lng_bhd", p7, self.s_m, self.l_web_m, mat7)
        hp7 = select_hp_section(sm7)
        r7, ok7 = check_hp_proportions(hp7["h_mm"], hp7["tw_mm"], mat7["fy"])
        lim7 = 37 * math.sqrt(235/mat7["fy"])
        panels.append(PanelScantling(
            zone="lng_bhd", location="Centerline longitudinal bulkhead",
            material=mat7["name"], p_kNm2=round(p7,2),
            t_net_mm=round(t7n,2), t_gross_mm=round(t7g,2),
            t_adopted_mm=round(math.ceil(t7g*2)/2, 1),
            stiffener=hp7["designation"], sm_req_cm3=round(sm7,1),
            sm_provided_cm3=hp7["SM_cm3"], hp_check_ok=ok7,
            hp_dw_tw_ratio=round(r7,1), hp_dw_tw_limit=round(lim7,1)
        ))

        ms.panels = panels

        # ---- Hull girder section modulus requirement ----
        girder = req_hull_section_modulus(L, B, Cb, mat_deck=mat6)
        ms.SM_sag_cm2m = girder["SM_sag_cm2m"]
        ms.SM_hog_cm2m = girder["SM_hog_cm2m"]

        # ---- Simplified structural steel weight from scantlings ----
        # Sum plate areas × thickness × density for key panels
        rho = 7850 / 1e9   # kg/mm³  → convert via lengths in mm
        # Approximate perimeter contributions per unit length
        t_bot = panels[0].t_adopted_mm; t_ib = panels[1].t_adopted_mm
        t_side = panels[3].t_adopted_mm; t_deck = panels[5].t_adopted_mm
        t_lng = panels[6].t_adopted_mm

        B_mm = B * 1000; D_mm = D * 1000

        # Plate weight per m length (kg/m):
        w_bot   = (B_mm) * t_bot   * rho * 1000   # outer bottom
        w_ib    = (B_mm * 0.90) * t_ib  * rho * 1000   # inner bottom (excl void flanks)
        w_side  = 2 * (D_mm) * t_side * rho * 1000     # both sides
        w_deck  = (B_mm) * t_deck  * rho * 1000         # deck
        w_lng   = D_mm * t_lng   * rho * 1000           # CL lng bhd

        plate_w_per_m = w_bot + w_ib + w_side + w_deck + w_lng   # kg/m

        # Stiffener weight per m length:
        # Longitudinals (one per frame space)
        n_long_bottom = int(B / (self.s_m))
        n_long_side   = int(D / (self.s_m)) * 2   # both sides
        n_long_deck   = int(B / (self.s_m))
        n_long_lng    = int(D / (self.s_m))

        stiff_w_per_m = (
            n_long_bottom * hp1["mass_kg_m"]
          + n_long_side   * hp4["mass_kg_m"]
          + n_long_deck   * hp6["mass_kg_m"]
          + n_long_lng    * hp7["mass_kg_m"]
        )

        total_w_per_m = plate_w_per_m + stiff_w_per_m   # kg/m
        W_steel_total = total_w_per_m * L / 1000   # tonnes (approx, midship section extended)

        ms.W_steel_t = round(W_steel_total * 1.25, 0)   # ×1.25 for ends and brackets

        # ---- Structural VCG ----
        # Weighted centroid of major structural elements
        moments = [
            w_bot   * 0.0,            # bottom at keel
            w_ib    * hdb * 1000,     # inner bottom at DB height
            w_side  * D_mm / 2.0,    # sides at mid-depth
            w_deck  * D_mm,          # deck at D
        ]
        total_mass = w_bot + w_ib + w_side + w_deck
        VCG_mm = sum(moments) / max(total_mass, 1)
        ms.VCG_struct_m = round(VCG_mm / 1000, 3)

        # Warnings
        for p in panels:
            if not p.hp_check_ok:
                ms.warnings.append(
                    f"{p.location}: {p.stiffener} d/t={p.hp_dw_tw_ratio:.1f} "
                    f"> limit {p.hp_dw_tw_limit:.1f} — select heavier section")

        return ms


# ---------------------------------------------------------------------------
# Integration with DesignPoint
# ---------------------------------------------------------------------------

def compute_structural(dp) -> MidshipSection:
    """
    Compute structural scantlings and populate dp.structure.
    Mutates dp in place.
    """
    solver = MidshipScantlingSolver(s_mm=800.0, l_web_m=3.2)
    ms = solver.solve(
        L=dp.hull.LBP,
        B=dp.hull.B,
        D=dp.hull.D,
        T=dp.hull.T,
        Cb=dp.hull.Cb,
    )
    dp.structure = ms
    # Update steel weight in weights module
    dp.weights.W_steel_t = ms.W_steel_t
    # Update VCG
    dp.weights.VCG_m = ms.VCG_struct_m * 1.4   # ~1.4x structural VCG for outfit+mach
    return ms


def structure_report(ms: MidshipSection) -> str:
    dh = f"  Double hull:  Wh={ms.Wh_m:.3f}m · hdb={ms.hdb_m:.3f}m · inner B/2={ms.y_inner:.2f}m"
    girder = (f"  Hull girder:  SM_sag={ms.SM_sag_cm2m:.0f} cm²·m · "
              f"SM_hog={ms.SM_hog_cm2m:.0f} cm²·m   [ABS 3-2-1, Grade AH32 deck]")
    weight = f"  Steel weight: {ms.W_steel_t:.0f} t · struct VCG={ms.VCG_struct_m:.2f}m"

    lines = [
        f"\n{'='*80}",
        f"  Midship Section Scantlings — {ms.L:.0f}m LBP · B={ms.B:.1f}m · D={ms.D:.1f}m · T={ms.T:.1f}m",
        f"  Frame grid: {ms.s_mm:.0f}mm longitudinal · {ms.l_web_m*1000:.0f}mm web frames",
        dh, girder, weight,
        f"{'='*80}",
        f"  {'Zone':<38}  {'Mat':<10}  {'p kNm²':>7}  {'t_net':>6}  {'t_gross':>7}  "
        f"{'t_adopt':>7}  {'Stiffener':>16}  {'SM_req':>7}  {'SM_prov':>7}  {'HP d/t':>7}  {'Limit':>6}",
        "  " + "-"*128,
    ]
    for p in ms.panels:
        ok = "✓" if p.hp_check_ok else "✗"
        lines.append(
            f"  {p.location:<38}  {p.material[:10]:<10}  {p.p_kNm2:>7.1f}  "
            f"{p.t_net_mm:>6.2f}  {p.t_gross_mm:>7.2f}  {p.t_adopted_mm:>7.1f}  "
            f"{p.stiffener:>16}  {p.sm_req_cm3:>7.1f}  {p.sm_provided_cm3:>7.1f}  "
            f"{p.hp_dw_tw_ratio:>7.1f}  {p.hp_dw_tw_limit:>6.1f}  {ok}"
        )
    if ms.warnings:
        lines.append(f"\n  Warnings ({len(ms.warnings)}):")
        for w in ms.warnings:
            lines.append(f"    ⚠  {w}")
    lines.append("="*80)
    return "\n".join(lines)
