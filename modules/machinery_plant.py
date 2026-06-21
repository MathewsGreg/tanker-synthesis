"""
machinery_plant.py
==================
Speed-power-machinery plant module for the tanker design family.

Covers the full propulsion chain:
  Resistance (Holtrop-Mennen) → Effective Power (PE)
  → Propeller sizing (Wageningen B-series regression)
  → Delivered Power (PD) via propulsive efficiency
  → Brake Power (PB) at service condition
  → Engine selection from catalogue
  → Generator / electric plant sizing
  → Fuel consumption curves (off-design via admiralty coefficient)
  → Range envelope

Engine catalogue (2024 data):
  Wärtsilä W32      — 580 kW/cyl, 750 rpm, 6–16 cyl, 3,480–9,280 kW
  MAN 32/44CR       — 600 kW/cyl, 750 rpm, 6–16 cyl, 3,600–9,600 kW  (GTS tanker ref.)
  Wärtsilä 34DF     — 500 kW/cyl, 750 rpm, 6–16 cyl, 3,000–8,000 kW  (dual-fuel)
  Wärtsilä 25 (DF)  — 375 kW/cyl, 750 rpm, 6–12 cyl, 2,250–4,500 kW  (small ships)

Auxiliary genset catalogue:
  MAN 23/30H        — 900 kW, 900 rpm (8L), 680 kW (6L) — standard tanker aux

Propeller: Wageningen B-series regression (van Lammeren 1969, Oosterveld 1975)
  Valid: Z=3-7 blades, Ae/Ao=0.3-1.05, P/D=0.5-1.4, J=0.1-1.0

Greg Tanker Synthesis Model — v0.1
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Engine catalogue
# ---------------------------------------------------------------------------

@dataclass
class EngineSpec:
    """A specific engine configuration (model + cylinder count)."""
    make: str
    model: str
    n_cyl: int
    kw_per_cyl: float
    rpm: int
    sfc_100: float     # g/kWh at 100% MCR, MDO
    sfc_85: float      # g/kWh at 85% MCR (service point) — typically best efficiency
    sfc_75: float      # g/kWh at 75% MCR
    sfc_50: float      # g/kWh at 50% MCR
    mass_t: float      # engine dry weight [t]
    length_m: float    # installed length [m]
    fuel_type: str     # MDO | DF | LNG
    tier3: bool        # IMO Tier III compliant
    notes: str = ""

    @property
    def mcr_kw(self) -> float:
        return self.n_cyl * self.kw_per_cyl

    @property
    def id(self) -> str:
        return f"{self.make} {self.n_cyl}L{self.model}"


# Build catalogue
def _make_catalogue() -> List[EngineSpec]:
    cat = []
    # Wärtsilä W32 (diesel, 580 kW/cyl, 750 rpm)
    # SFC: ~165 g/kWh at 100%, 155 at 85%, 160 at 75%, 175 at 50%
    for nc, mass, length in [(6,17.5,6.1),(7,19.8,6.9),(8,22.0,7.6),(9,24.2,8.4),
                              (12,28.0,8.8),(16,36.0,11.2)]:
        cat.append(EngineSpec("Wärtsilä","W32",nc,580,750,
            165,155,160,178,mass,length,"MDO",False,
            "IMO Tier II standard; Tier III with SCR"))

    # MAN 32/44CR (600 kW/cyl, 750 rpm) — GTS tanker reference
    # SFC: 163 at 100%, 153 at 85%, 158 at 75%, 173 at 50%
    for nc, mass, length in [(6,18.5,6.3),(7,21.0,7.1),(8,23.5,7.9),(9,26.0,8.7),
                              (10,28.5,9.4),(12,30.5,9.2),(16,40.0,11.8)]:
        cat.append(EngineSpec("MAN","32/44CR",nc,600,750,
            163,153,158,173,mass,length,"MDO",True,
            "Common-rail; Tier III native; GTS 41k DWT tanker ref."))

    # Wärtsilä 34DF (dual-fuel, 500 kW/cyl, 750 rpm)
    # SFC diesel: 168 at 100%, 158 at 85%, 163 at 75%, 182 at 50%
    for nc, mass, length in [(6,19.0,6.5),(8,24.5,8.0),(9,27.0,8.9),
                              (12,34.0,9.4),(16,44.0,12.0)]:
        cat.append(EngineSpec("Wärtsilä","34DF",nc,500,750,
            168,158,163,182,mass,length,"DF",True,
            "LNG/diesel dual-fuel; 25% GHG reduction on LNG"))

    # Wärtsilä W25 (smaller ships, 375 kW/cyl, 750 rpm)
    for nc, mass, length in [(6,11.0,5.2),(8,14.0,6.5),(9,15.5,7.1),(12,19.0,7.8)]:
        cat.append(EngineSpec("Wärtsilä","W25",nc,375,750,
            170,160,165,183,mass,length,"DF",True,
            "Compact DF; Erik Thun coastal tankers; IMO Tier III"))

    return cat

ENGINE_CATALOGUE = _make_catalogue()


# Auxiliary genset options
@dataclass
class AuxGenSpec:
    make: str
    model: str
    kw_e: float       # electrical output [kW]
    rpm: int
    sfc_g_kwh: float
    mass_t: float

AUX_GENSET_CATALOGUE = [
    AuxGenSpec("MAN",  "8L23/30H", 900, 900, 178, 8.5),
    AuxGenSpec("MAN",  "6L23/30H", 680, 900, 180, 6.8),
    AuxGenSpec("Wärtsilä","9L20",  900, 900, 176, 8.2),
    AuxGenSpec("Wärtsilä","6L20",  630, 900, 178, 5.9),
]


# ---------------------------------------------------------------------------
# Wageningen B-series propeller model
# ---------------------------------------------------------------------------

class WageningenBSeries:
    """
    Open-water efficiency of the Wageningen B-series propeller.
    Regression coefficients from Oosterveld & van Oossanen (1975).

    Computes KT, KQ, eta_O as functions of J (advance coefficient),
    P/D (pitch ratio), Ae/Ao (blade area ratio), Z (blade count).

    Valid range: J 0.1–1.0, P/D 0.5–1.4, Ae/Ao 0.3–1.05, Z 3–7
    """

    # KT regression coefficients (Oosterveld 1975, Table I)
    _CT = [
        (0.00880496,  0, 0, 0, 0),
        (-0.204554,   1, 0, 0, 0),
        (0.166351,    0, 1, 0, 0),
        (0.158114,    0, 2, 0, 0),
        (-0.147581,   2, 0, 1, 0),
        (-0.481497,   1, 1, 1, 0),
        (0.415437,    0, 2, 1, 0),
        (0.0144043,   0, 0, 0, 1),
        (-0.0530054,  2, 0, 0, 1),
        (0.0143481,   0, 1, 0, 1),
        (0.0606826,   1, 1, 0, 1),
        (-0.0125894,  0, 0, 1, 1),
        (0.0109689,   1, 0, 1, 1),
        (-0.133698,   0, 3, 0, 0),
        (0.00638407,  0, 6, 0, 0),
        (-0.00132718, 2, 6, 0, 0),
        (0.168496,    3, 0, 1, 0),
        (-0.0507214,  0, 0, 2, 0),
        (0.0854559,   2, 0, 2, 0),
        (-0.0504475,  3, 0, 2, 0),
        (0.010465,    1, 6, 2, 0),
        (-0.00648272, 2, 6, 2, 0),
        (-0.00841728, 0, 3, 0, 1),
        (0.0168424,   1, 3, 0, 1),
        (-0.00102296, 3, 3, 0, 1),
        (-0.0317791,  0, 3, 1, 1),
        (0.018604,    1, 0, 2, 1),
        (-0.00410798, 0, 2, 2, 1),
    ]

    # KQ regression coefficients
    _CQ = [
        (0.00379368,  0, 0, 0, 0),
        (0.00886523,  2, 0, 0, 0),
        (-0.032241,   1, 1, 0, 0),
        (0.00344778,  0, 2, 0, 0),
        (-0.0408811,  0, 1, 1, 0),
        (-0.108009,   1, 1, 1, 0),
        (-0.0885381,  2, 1, 1, 0),
        (0.188561,    0, 2, 1, 0),
        (-0.00370871, 1, 0, 0, 1),
        (0.00513696,  0, 1, 0, 1),
        (0.0209449,   1, 1, 0, 1),
        (0.00474319,  2, 1, 0, 1),
        (-0.00723408, 2, 0, 1, 1),
        (0.00438388,  1, 1, 1, 1),
        (-0.0269403,  0, 2, 1, 1),
        (0.0558082,   3, 0, 1, 0),
        (0.0161886,   0, 3, 1, 0),
        (0.00471729,  1, 3, 1, 0),
        (0.0196283,   3, 0, 2, 0),
        (-0.0502782,  0, 6, 2, 0),
        (-0.030055,   3, 6, 2, 0),
        (0.0417122,   2, 6, 0, 1),
        (-0.0397722,  0, 3, 0, 1),
        (-0.00350024, 0, 6, 0, 1),
        (-0.0106854,  3, 0, 1, 1),
        (0.00110903,  3, 3, 1, 1),
        (-0.000313912,0, 6, 1, 1),
        (0.0035985,   3, 0, 2, 1),
        (-0.00142121, 0, 6, 2, 1),
        (-0.00383637, 1, 0, 2, 1),
        (0.0126803,   0, 2, 2, 1),
        (-0.00318278, 2, 3, 2, 1),
        (0.00334268,  0, 6, 2, 1),
        (-0.00183491, 1, 1, 0, 2),
        (0.000112451, 3, 2, 0, 2),
        (-0.0000297228,3,6,0,2),
        (0.000269551, 1, 0, 1, 2),
        (0.00083265,  2, 0, 1, 2),
        (0.00155334,  0, 2, 1, 2),
        (0.000302683, 0, 6, 1, 2),
        (-0.0001843,  0, 6, 2, 2),
        (-0.000425399,3,6,2,2),
        (0.0000869243,3,6,0,3),
        (-0.0004659,  0, 3, 0, 3),
        (0.0000554194,0,6,0,3),
    ]

    def compute(self, J: float, PD: float, AeAo: float, Z: int,
                n_iter: int = 1) -> dict:
        """
        Compute open-water propeller coefficients.

        Parameters
        ----------
        J    : advance coefficient  Va / (n×D)
        PD   : pitch ratio P/D
        AeAo : expanded blade area ratio
        Z    : number of blades
        """
        J = max(0.05, min(J, 1.2))

        KT = sum(c * J**s * PD**t * AeAo**u * Z**v
                 for c,s,t,u,v in self._CT)
        KQ_raw = sum(c * J**s * PD**t * AeAo**u * Z**v
                     for c,s,t,u,v in self._CQ)

        KT = max(KT, 0.0)
        KQ = max(KQ_raw, 1e-6)

        eta_O = (J / (2 * np.pi)) * (KT / KQ) if KQ > 0 else 0.0

        return {"KT": KT, "KQ": KQ, "eta_O": eta_O}

    def design_point(self, Vs_kn: float, LBP: float, B: float, T: float,
                     Cb: float, PE_kW: float,
                     Z: int = 4, AeAo: float = 0.55,
                     wake_fraction: float = None,
                     thrust_deduction: float = None) -> dict:
        """
        Solve for optimal propeller diameter, P/D, and n (rpm) at design point.
        Uses a simplified parametric approach matching Bp-delta charts.
        """
        Vs = Vs_kn * 0.5144  # m/s
        Cp = Cb / 0.985

        # Wake and thrust deduction (if not supplied)
        if wake_fraction is None:
            w = max(0.0, 0.7*Cp - 0.18)
        else:
            w = wake_fraction
        if thrust_deduction is None:
            t = max(0.10, min(0.25, 0.325*Cb - 0.19))
        else:
            t = thrust_deduction

        Va = Vs * (1 - w)          # advance speed [m/s]
        eta_H = (1 - t) / (1 - w)  # hull efficiency
        eta_R = 1.035               # relative rotative efficiency (single screw)

        # Thrust required: T_req = RT / (1 - t)
        # RT = PE / Vs
        RT_kN = PE_kW / Vs
        T_req_kN = RT_kN / (1 - t)

        # Constraint: propeller diameter ≤ 0.72 × T (single screw tanker clearance)
        D_max = 0.72 * T
        # Typical range: 0.55×T to 0.72×T
        D_nominal = min(D_max, 0.014 * LBP**0.5 * T**0.5 * 3.5)

        # Iterate to find optimal P/D and n
        best_eta_O = 0.0
        best_result = {}
        wb = WageningenBSeries()

        for D_frac in np.linspace(0.55, 0.72, 8):
            D = D_frac * T
            for PD in np.linspace(0.65, 1.10, 10):
                # Find n that satisfies KT requirement at this D and PD
                # KT = T_req / (rho * n² * D⁴)
                # J = Va / (n * D)
                # Solve iteratively
                rho = 1025.0
                # Initial n estimate from J≈0.7
                J_est = 0.70
                n_est = Va / (J_est * D)
                for _ in range(12):
                    J = Va / (n_est * D) if n_est > 0 else 0.1
                    res = wb.compute(J, PD, AeAo, Z)
                    n2 = max(n_est**2, 1e-9)
                    KT_req = T_req_kN * 1000 / (rho * n2 * D**4)
                    if abs(res["KT"] - KT_req) < 0.001:
                        break
                    # Newton step: adjust n
                    if res["KT"] > 0:
                        n_est = max(0.1, n_est * np.sqrt(res["KT"] / max(KT_req, 1e-6)))
                    else:
                        break

                n_rpm = n_est * 60
                if n_rpm < 60 or n_rpm > 300:
                    continue

                J = Va / (n_est * D)
                res = wb.compute(J, PD, AeAo, Z)
                if res["eta_O"] > best_eta_O:
                    best_eta_O = res["eta_O"]
                    best_result = {
                        "D_m":    round(D, 3),
                        "PD":     round(PD, 3),
                        "n_rpm":  round(n_rpm, 1),
                        "J":      round(J, 4),
                        "KT":     round(res["KT"], 4),
                        "KQ":     round(res["KQ"], 5),
                        "eta_O":  round(res["eta_O"], 4),
                        "eta_H":  round(eta_H, 4),
                        "eta_R":  eta_R,
                        "eta_D":  round(res["eta_O"] * eta_H * eta_R, 4),
                        "w":      round(w, 4),
                        "t":      round(t, 4),
                        "Va_ms":  round(Va, 3),
                        "T_req_kN": round(T_req_kN, 1),
                        "Z":      Z,
                        "AeAo":   AeAo,
                    }

        return best_result


# ---------------------------------------------------------------------------
# Engine selector
# ---------------------------------------------------------------------------

class EngineSelector:

    def select(self, PB_required_kW: float,
               prefer_df: bool = False,
               max_engines: int = 2) -> List[EngineSpec]:
        """
        Select best engine configuration for required brake power.

        Strategy:
          1. Single engine if MCR/0.85 ≥ PB_required (service at 85% MCR)
          2. Twin engines if single too large
          3. Prefer MAN 32/44CR (best real-world tanker reference)
          4. Fall back to Wärtsilä W32
          5. DF option: Wärtsilä 34DF if prefer_df
        """
        # Service PB → MCR installed: PB_req = MCR × 0.85 → MCR = PB/0.85
        mcr_needed = PB_required_kW / 0.85

        candidates = []

        for n_eng in range(1, max_engines + 1):
            mcr_per_eng = mcr_needed / n_eng

            for eng in ENGINE_CATALOGUE:
                # Skip non-DF if prefer_df and vice versa
                if prefer_df and eng.fuel_type == "MDO":
                    continue

                if abs(eng.mcr_kw - mcr_per_eng) / mcr_per_eng < 0.25:
                    # MCR within 25% of requirement
                    margin = (eng.mcr_kw * n_eng - mcr_needed) / mcr_needed
                    if 0.0 <= margin <= 0.35:
                        candidates.append((margin, n_eng, eng))

        if not candidates:
            # Relax to 40% margin
            for n_eng in range(1, max_engines + 1):
                mcr_per_eng = mcr_needed / n_eng
                for eng in ENGINE_CATALOGUE:
                    if prefer_df and eng.fuel_type == "MDO":
                        continue
                    margin = (eng.mcr_kw * n_eng - mcr_needed) / mcr_needed
                    if -0.05 <= margin <= 0.45:
                        candidates.append((abs(margin), n_eng, eng))

        if not candidates:
            return []

        # Sort: prefer MAN 32/44CR first, then by margin
        def sort_key(x):
            margin, n_eng, eng = x
            brand_pref = 0 if "32/44CR" in eng.model else (1 if "W32" in eng.model else 2)
            return (brand_pref, n_eng, margin)

        candidates.sort(key=sort_key)
        _, n_eng_best, eng_best = candidates[0]
        return [eng_best] * n_eng_best

    def select_aux_gensets(self, total_elec_kW: float,
                           n_sets: int = 3) -> List[AuxGenSpec]:
        """
        Select auxiliary gensets.
        Standard: 3 × gensets at 50% load each (N+1 redundancy, one standby)
        """
        load_per_running = total_elec_kW / (n_sets - 1)   # N-1 running
        target_kw = load_per_running / 0.75               # run at 75% MCR

        candidates = [(abs(g.kw_e - target_kw), i, g)
                      for i,g in enumerate(AUX_GENSET_CATALOGUE) if g.kw_e >= load_per_running]
        if not candidates:
            candidates = [(abs(g.kw_e - target_kw), i, g) for i,g in enumerate(AUX_GENSET_CATALOGUE)]

        candidates.sort()
        best = candidates[0][2]
        return [best] * n_sets


# ---------------------------------------------------------------------------
# Off-design performance (Admiralty method + SFC curve)
# ---------------------------------------------------------------------------

def speed_power_curve(LBP: float, B: float, T: float, Cb: float,
                      Vs_design: float, PB_design: float,
                      speed_range: np.ndarray = None,
                      sfc_85: float = 155.0) -> List[dict]:
    """
    Generate off-design speed-power-fuel curve using Admiralty coefficient.

    Admiralty coefficient C_adm = Δ^(2/3) × V³ / PB  (constant for similar form)
    PB(V) = Δ^(2/3) × V³ / C_adm

    SFC curve: quadratic fit through published points (100%, 85%, 75%, 50% MCR)
    """
    from modules.resistance_holtrop import HoltropMennen, PropulsiveCoefficients

    if speed_range is None:
        speed_range = np.arange(8.0, Vs_design + 2.5, 0.5)

    disp_t = Cb * LBP * B * T * 1.025
    hm = HoltropMennen()
    pc = PropulsiveCoefficients()

    # SFC curve coefficients (quadratic in % MCR)
    # Points: (50%, sfc_50), (75%, sfc_75), (85%, sfc_85), (100%, sfc_100)
    sfc_100 = sfc_85 * 1.065
    sfc_75  = sfc_85 * 1.032
    sfc_50  = sfc_85 * 1.148

    rows = []
    for Vs in speed_range:
        r = hm.compute(LBP=LBP, B=B, T=T, Cb=Cb, Vs_kn=Vs)
        p = pc.compute(LBP=LBP, B=B, T=T, Cb=Cb, Vs_kn=Vs, PE_kW=r["PE_kW"])
        PB_calm = p["PB_kW"] * 1.15   # 15% sea margin

        # MCR fraction
        mcr_frac = PB_calm / (PB_design / 0.85) if PB_design > 0 else 0.85

        # SFC at this MCR fraction (quadratic)
        # Fit: sfc = a*x² + b*x + c where x = mcr_frac
        # Through (0.5, sfc_50), (0.85, sfc_85), (1.0, sfc_100)
        sfc_at_mcr = _sfc_at_fraction(mcr_frac, sfc_50, sfc_75, sfc_85, sfc_100)

        fuel_t_day = PB_calm * sfc_at_mcr / 1e6 * 24

        rows.append({
            "Vs_kn":        round(Vs, 1),
            "Fn":           round(r["Fn"], 4),
            "Rt_kN":        round(r["Rt_kN"], 1),
            "PE_kW":        round(r["PE_kW"], 0),
            "PB_kW":        round(PB_calm, 0),
            "mcr_frac":     round(mcr_frac, 3),
            "sfc_g_kwh":    round(sfc_at_mcr, 1),
            "fuel_t_day":   round(fuel_t_day, 2),
        })
    return rows


def _sfc_at_fraction(x: float,
                     sfc_50: float, sfc_75: float,
                     sfc_85: float, sfc_100: float) -> float:
    """Interpolate SFC at given MCR fraction via piecewise linear."""
    x = max(0.20, min(x, 1.10))
    if x <= 0.50:
        return sfc_50 * (1 + (0.50 - x) * 0.8)
    elif x <= 0.75:
        t = (x - 0.50) / 0.25
        return sfc_50 + t * (sfc_75 - sfc_50)
    elif x <= 0.85:
        t = (x - 0.75) / 0.10
        return sfc_75 + t * (sfc_85 - sfc_75)
    elif x <= 1.0:
        t = (x - 0.85) / 0.15
        return sfc_85 + t * (sfc_100 - sfc_85)
    else:
        return sfc_100 * (1 + (x - 1.0) * 0.05)


def range_envelope(speed_curve: List[dict],
                   fuel_capacity_t: float,
                   reserve_frac: float = 0.10) -> List[dict]:
    """Compute achievable range at each speed given fuel capacity."""
    usable_fuel = fuel_capacity_t * (1 - reserve_frac)
    result = []
    for row in speed_curve:
        if row["fuel_t_day"] > 0:
            days = usable_fuel / row["fuel_t_day"]
            rng = days * 24 * row["Vs_kn"]
        else:
            days, rng = 0, 0
        result.append({**row,
                        "range_nm": round(rng, 0),
                        "endurance_days": round(days, 1)})
    return result


# ---------------------------------------------------------------------------
# Top-level integration
# ---------------------------------------------------------------------------

@dataclass
class MachineryPlant:
    """Complete machinery plant selection and performance."""
    # Main engine
    main_engines:    List[EngineSpec] = field(default_factory=list)
    n_main_engines:  int = 0
    mcr_total_kw:    float = 0.0
    service_pb_kw:   float = 0.0
    mcr_margin_pct:  float = 0.0

    # Propeller
    propeller: dict = field(default_factory=dict)

    # Aux gensets
    aux_gensets:     List[AuxGenSpec] = field(default_factory=list)
    n_aux_gensets:   int = 0
    total_gen_kw:    float = 0.0
    elec_load_kw:    float = 0.0

    # Performance curves
    speed_curve:     List[dict] = field(default_factory=list)
    range_curve:     List[dict] = field(default_factory=list)

    # Key single-point results
    fuel_t_day:      float = 0.0
    range_nm:        float = 0.0
    sfc_service:     float = 0.0
    eta_propulsive:  float = 0.0

    # Machinery weight
    W_machinery_t:   float = 0.0
    W_gensets_t:     float = 0.0

    warnings: List[str] = field(default_factory=list)


def compute_machinery(dp) -> MachineryPlant:
    """Full machinery plant solve. Attaches to dp.machinery. Mutates dp."""
    from modules.resistance_holtrop import HoltropMennen, PropulsiveCoefficients, compute_resistance

    h  = dp.hull
    m  = dp.mission
    pr = dp.propulsion

    # Ensure resistance is computed
    if dp.resistance.PE_kW == 0:
        compute_resistance(dp)

    # ---- Propeller design ----
    wb = WageningenBSeries()
    prop = wb.design_point(
        Vs_kn=m.Vs_kn, LBP=h.LBP, B=h.B, T=h.T, Cb=h.Cb,
        PE_kW=dp.resistance.PE_kW, Z=4, AeAo=0.55)

    # Use Wageningen eta_D to compute PB
    if prop and prop.get("eta_D", 0) > 0:
        eta_D   = prop["eta_D"]
        eta_S   = 0.97
        PD_kW   = dp.resistance.PE_kW / eta_D
        PB_kW   = PD_kW / eta_S
    else:
        # Fall back to Harvald
        pc = PropulsiveCoefficients()
        p  = pc.compute(h.LBP, h.B, h.T, h.Cb, m.Vs_kn, dp.resistance.PE_kW)
        PB_kW   = p["PB_kW"]
        eta_D   = p.get("eta_D", 0.67)
        eta_S   = 0.97
        if prop:
            prop["eta_D"] = eta_D

    PB_service  = PB_kW * (1 + pr.sea_margin)

    # ---- Engine selection ----
    prefer_df = pr.propulsion_type in ("diesel_electric", "hybrid")
    sel = EngineSelector()
    engines = sel.select(PB_service, prefer_df=prefer_df)

    if not engines:
        mp = MachineryPlant()
        mp.warnings.append("No suitable engine found in catalogue — check PB requirement")
        return mp

    mcr_total = sum(e.mcr_kw for e in engines)
    mcr_margin = (mcr_total - PB_service / pr.engine_margin) / (PB_service / pr.engine_margin)
    sfc_service = engines[0].sfc_85

    fuel_rate_t_day = PB_service * sfc_service / 1e6 * 24
    days_at_sea     = m.range_nm / (m.Vs_kn * 24)
    fuel_cap        = fuel_rate_t_day * days_at_sea * 1.10

    # ---- Aux gensets ----
    elec_load = pr.hotel_load_kW + pr.autonomy_load_kW
    aux_sets = sel.select_aux_gensets(elec_load, n_sets=3)
    total_gen_kw = sum(g.kw_e for g in aux_sets)

    # ---- Curves ----
    speeds = np.arange(8.0, m.Vs_kn + 2.5, 0.5)
    curve  = speed_power_curve(h.LBP, h.B, h.T, h.Cb,
                                m.Vs_kn, PB_service, speeds, sfc_service)
    rcurve = range_envelope(curve, fuel_cap)

    # ---- Weights ----
    W_eng  = sum(e.mass_t for e in engines)
    W_gen  = sum(g.mass_t for g in aux_sets)
    W_gear = 0.15 * W_eng    # gearbox estimate
    W_total_mach = W_eng + W_gen + W_gear + 5.0  # 5t misc (shafts, seals, foundation)

    # ---- Update dp ----
    dp.power.PB_kW              = round(PB_service, 0)
    dp.power.PD_kW              = round(PB_service * eta_S, 0)
    dp.power.PE_kW              = dp.resistance.PE_kW
    dp.power.fuel_rate_t_per_day = round(fuel_rate_t_day, 2)
    dp.power.fuel_capacity_t     = round(fuel_cap, 0)
    dp.power.total_elec_load_kW  = elec_load
    dp.power.generator_capacity_kW = total_gen_kw
    dp.power.propulsive_efficiency = round(eta_D * eta_S, 4)
    dp.weights.W_machinery_t    = round(W_total_mach, 1)

    mp = MachineryPlant(
        main_engines=engines, n_main_engines=len(engines),
        mcr_total_kw=mcr_total, service_pb_kw=PB_service,
        mcr_margin_pct=round(mcr_margin * 100, 1),
        propeller=prop or {},
        aux_gensets=aux_sets, n_aux_gensets=len(aux_sets),
        total_gen_kw=total_gen_kw, elec_load_kw=elec_load,
        speed_curve=curve, range_curve=rcurve,
        fuel_t_day=round(fuel_rate_t_day, 2),
        range_nm=round(rcurve[-3]["range_nm"] if len(rcurve)>3 else 0, 0),
        sfc_service=sfc_service,
        eta_propulsive=round(eta_D * eta_S, 4),
        W_machinery_t=round(W_total_mach, 1),
        W_gensets_t=round(W_gen, 1),
    )
    dp.machinery = mp
    return mp


def machinery_report(mp: MachineryPlant) -> str:
    if not mp.main_engines:
        return "  No engine selected."
    e = mp.main_engines[0]
    lines = [
        f"\n{'='*68}",
        f"  Machinery Plant",
        f"{'='*68}",
        f"  Main engine:   {mp.n_main_engines} × {e.id}",
        f"                 {e.mcr_kw:.0f} kW/unit  ×  {mp.n_main_engines} = {mp.mcr_total_kw:.0f} kW MCR",
        f"                 {e.rpm} rpm  |  SFC 85%: {e.sfc_85} g/kWh  |  {e.fuel_type}",
        f"                 {'IMO Tier III' if e.tier3 else 'Tier II (SCR for Tier III)'}",
        f"  Service PB:    {mp.service_pb_kw:.0f} kW  (MCR margin {mp.mcr_margin_pct:.1f}%)",
        f"  Propeller:     Ø{mp.propeller.get('D_m',0):.2f}m  P/D={mp.propeller.get('PD',0):.3f}  "
        f"Z={mp.propeller.get('Z',0)}  η_O={mp.propeller.get('eta_O',0):.3f}",
        f"                 η_D={mp.propeller.get('eta_D',0):.3f}  n={mp.propeller.get('n_rpm',0):.0f} rpm",
        f"  Fuel:          {mp.fuel_t_day:.1f} t/day  |  SFC {mp.sfc_service} g/kWh",
        f"  Aux gensets:   {mp.n_aux_gensets} × {mp.aux_gensets[0].make if mp.aux_gensets else '-'} "
        f"{mp.aux_gensets[0].model if mp.aux_gensets else '-'}  "
        f"({mp.aux_gensets[0].kw_e if mp.aux_gensets else 0} kW each)  "
        f"= {mp.total_gen_kw:.0f} kW total",
        f"  Elec load:     {mp.elec_load_kw:.0f} kW  (hotel + autonomy)",
        f"  Range:         {mp.range_nm:.0f} nm at service speed",
        f"  Mach weight:   {mp.W_machinery_t:.0f} t",
        f"",
        f"  {'Vs (kn)':>8} {'Fn':>7} {'Rt (kN)':>9} {'PE (kW)':>9} {'PB (kW)':>9} {'%MCR':>6} {'SFC':>6} {'Fuel t/d':>9} {'Range nm':>10}",
        f"  " + "-"*78,
    ]
    for row in mp.range_curve:
        lines.append(
            f"  {row['Vs_kn']:>8.1f} {row['Fn']:>7.4f} {row['Rt_kN']:>9.1f} "
            f"{row['PE_kW']:>9.0f} {row['PB_kW']:>9.0f} {row['mcr_frac']*100:>6.1f} "
            f"{row['sfc_g_kwh']:>6.1f} {row['fuel_t_day']:>9.2f} {row['range_nm']:>10.0f}")
    lines.append("="*68)
    return "\n".join(lines)
