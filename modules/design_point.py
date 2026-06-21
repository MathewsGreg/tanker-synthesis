"""
design_point.py
===============
Shared parameter vector (design point) for the tanker synthesis model.

All discipline modules read from and write to DesignPoint objects.
This is the single source of truth for the design — one dict-like object
that travels through the entire OpenMDAO problem.

Design family: Jones Act / ABS SVR product tankers, 100–200 m LOA
Mission context: INDOPACOM distributed logistics, autonomous ops

Greg Tanker Synthesis Model — v0.1
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Primary design variables (inputs to the synthesis)
# ---------------------------------------------------------------------------

@dataclass
class HullParams:
    """Principal dimensions and form coefficients."""
    LOA: float = 150.0        # Length overall [m]
    LBP: float = 145.0        # Length between perpendiculars [m]
    B: float = 23.0           # Moulded breadth [m]
    D: float = 12.5           # Moulded depth [m]
    T: float = 8.5            # Design draught [m]
    Cb: float = 0.78          # Block coefficient [-]
    Cm: float = 0.985         # Midship section coefficient [-]
    Cwp: float = 0.87         # Waterplane area coefficient [-]
    lcb_fwd_midship: float = 1.5  # LCB forward of midship [% LBP, positive fwd]

    @property
    def Cp(self) -> float:
        """Prismatic coefficient."""
        return self.Cb / self.Cm

    @property
    def displacement_vol(self) -> float:
        """Displacement volume [m³]."""
        return self.Cb * self.LBP * self.B * self.T

    @property
    def displacement_t(self) -> float:
        """Displacement [tonnes], salt water rho=1.025."""
        return self.displacement_vol * 1.025

    @property
    def LB_ratio(self) -> float:
        return self.LBP / self.B

    @property
    def BD_ratio(self) -> float:
        return self.B / self.D

    @property
    def TD_ratio(self) -> float:
        return self.T / self.D


@dataclass
class MissionParams:
    """Operational requirements."""
    Vs_kn: float = 15.0          # Service speed [knots] — INDOPACOM standard
    range_nm: float = 4500.0     # Design range [nm] — Pearl Harbor → Guam + margin
    endurance_days: float = 25.0 # Endurance without replenishment [days]
    DWT_target: float = 25000.0  # Target deadweight [tonnes]
    cargo_vol_target: float = 0.0  # Target cargo volume [m³]; 0 = derive from DWT
    crew_target: int = 10         # Target crew count (autonomy driver)
    unrep_stations: int = 1       # Number of UNREP fuel stations


@dataclass
class PropulsionParams:
    """Machinery and propulsion configuration."""
    propulsion_type: str = "diesel_mechanical"  # diesel_mechanical | diesel_electric | hybrid
    num_shafts: int = 1
    num_thrusters_bow: int = 1
    sea_margin: float = 0.15      # Sea margin on calm-water resistance [-]
    engine_margin: float = 0.85   # MCR fraction used at service speed [-]
    sfc_g_per_kWh: float = 175.0  # Specific fuel consumption at service [g/kWh]
    hotel_load_kW: float = 600.0  # Baseline hotel/auxiliary load [kW]
    autonomy_load_kW: float = 150.0  # Additional load for autonomous systems [kW]


# ---------------------------------------------------------------------------
# Computed results (outputs from discipline modules)
# ---------------------------------------------------------------------------

@dataclass
class ResistanceResults:
    """Outputs from Holtrop-Mennen resistance calculation."""
    Rt_kN: float = 0.0           # Total resistance [kN]
    Rw_kN: float = 0.0           # Wave-making resistance [kN]
    Rf_kN: float = 0.0           # Frictional resistance [kN]
    Rapp_kN: float = 0.0         # Appendage resistance [kN]
    Rb_kN: float = 0.0           # Bulb resistance [kN]
    Rtr_kN: float = 0.0          # Transom resistance [kN]
    Ra_kN: float = 0.0           # Model-ship correlation resistance [kN]
    PE_kW: float = 0.0           # Effective power [kW]
    Fn: float = 0.0              # Froude number [-]
    Ct: float = 0.0              # Total resistance coefficient [-]


@dataclass
class PowerResults:
    """Outputs from power and propulsion module."""
    PB_kW: float = 0.0           # Brake power at MCR [kW]
    PD_kW: float = 0.0           # Delivered power [kW]
    fuel_rate_t_per_day: float = 0.0  # Fuel consumption at service [t/day]
    fuel_capacity_t: float = 0.0     # Required fuel capacity [t]
    total_elec_load_kW: float = 0.0  # Total electrical load [kW]
    generator_capacity_kW: float = 0.0  # Required generator capacity [kW]
    propulsive_efficiency: float = 0.0   # Overall propulsive efficiency [-]


@dataclass
class WeightResults:
    """Outputs from parametric weight estimate (Watson/Barrass)."""
    W_steel_t: float = 0.0       # Steel weight — hull structure [t]
    W_outfit_t: float = 0.0      # Outfit and equipment weight [t]
    W_machinery_t: float = 0.0   # Machinery weight [t]
    W_lightship_t: float = 0.0   # Lightship displacement [t]
    W_deadweight_t: float = 0.0  # Achieved deadweight [t]
    W_cargo_t: float = 0.0       # Net cargo capacity [t]
    LCG_m: float = 0.0           # Lightship LCG from AP [m]
    VCG_m: float = 0.0           # Lightship VCG from keel [m]


@dataclass
class StabilityResults:
    """Preliminary stability indicators."""
    KB_m: float = 0.0            # Centre of buoyancy above keel [m]
    BM_t: float = 0.0            # Transverse metacentric radius [m]
    GM_t: float = 0.0            # Transverse metacentric height, lightship [m]
    GM_loaded_m: float = 0.0     # GM at full load [m]
    freeboard_m: float = 0.0     # Freeboard at midship [m]
    intact_stable: bool = False  # Passes basic intact stability check


@dataclass
class CostResults:
    """Parametric acquisition and lifecycle cost."""
    steel_cost_M: float = 0.0       # Steel material cost [M$]
    outfit_cost_M: float = 0.0      # Outfit and equipment cost [M$]
    machinery_cost_M: float = 0.0   # Machinery cost [M$]
    build_cost_M: float = 0.0       # Total build cost [M$]
    annual_opex_M: float = 0.0      # Annual operating cost [M$/year]
    fuel_cost_annual_M: float = 0.0 # Annual fuel cost [M$/year]
    crew_cost_annual_M: float = 0.0 # Annual crew cost [M$/year]


# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------

@dataclass
class DesignPoint:
    """
    Complete design point: inputs + all discipline outputs.
    Pass this object between modules; modules mutate their result fields.
    """
    # --- identity ---
    name: str = "baseline"
    notes: str = ""

    # --- inputs ---
    hull: HullParams = field(default_factory=HullParams)
    mission: MissionParams = field(default_factory=MissionParams)
    propulsion: PropulsionParams = field(default_factory=PropulsionParams)

    # --- discipline results (populated by modules) ---
    resistance: ResistanceResults = field(default_factory=ResistanceResults)
    power: PowerResults = field(default_factory=PowerResults)
    weights: WeightResults = field(default_factory=WeightResults)
    stability: StabilityResults = field(default_factory=StabilityResults)
    cost: CostResults = field(default_factory=CostResults)

    def summary(self) -> str:
        h, m, r, p, w = self.hull, self.mission, self.resistance, self.power, self.weights
        lines = [
            f"\n{'='*60}",
            f"  Design Point: {self.name}",
            f"{'='*60}",
            f"  Hull:       LOA={h.LOA:.1f}m  B={h.B:.1f}m  T={h.T:.1f}m  Cb={h.Cb:.3f}",
            f"              L/B={h.LB_ratio:.2f}  B/D={h.BD_ratio:.2f}  disp={h.displacement_t:.0f}t",
            f"  Mission:    Vs={m.Vs_kn:.1f}kn  range={m.range_nm:.0f}nm  DWT={m.DWT_target:.0f}t  crew={m.crew_target}",
            f"  Resistance: Rt={r.Rt_kN:.1f}kN  PE={r.PE_kW:.0f}kW  Fn={r.Fn:.4f}",
            f"  Power:      PB={p.PB_kW:.0f}kW  fuel={p.fuel_rate_t_per_day:.1f}t/day  cap={p.fuel_capacity_t:.0f}t",
            f"  Weights:    LWT={w.W_lightship_t:.0f}t  DWT={w.W_deadweight_t:.0f}t  cargo={w.W_cargo_t:.0f}t",
            f"  Stability:  GM={self.stability.GM_loaded_m:.2f}m  freeboard={self.stability.freeboard_m:.2f}m",
            f"  Cost:       build=${self.cost.build_cost_M:.1f}M  opex=${self.cost.annual_opex_M:.1f}M/yr",
            f"{'='*60}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Design family helper: generate a parametric sweep
# ---------------------------------------------------------------------------

def make_design_family(
    loa_values: list,
    base: Optional[DesignPoint] = None,
    name_prefix: str = "T"
) -> list:
    """
    Generate a list of DesignPoints spanning a range of LOA values.
    All other parameters scale from the base using standard tanker ratios.

    Scaling approach:
    - B, D, T scale from LOA using fixed L/B, B/D, T/D ratios from base
    - DWT scales as ~LOA^2.8 (empirical tanker relationship)
    - Speed and Cb held constant (family assumption)
    """
    import copy
    if base is None:
        base = DesignPoint(name="base")

    family = []
    loa_base = base.hull.LOA

    for loa in loa_values:
        dp = copy.deepcopy(base)
        dp.name = f"{name_prefix}{int(loa)}"
        scale = loa / loa_base

        # Scale principal dimensions
        dp.hull.LOA = loa
        dp.hull.LBP = loa * (base.hull.LBP / base.hull.LOA)
        dp.hull.B   = base.hull.B * scale
        dp.hull.D   = base.hull.D * scale
        dp.hull.T   = base.hull.T * scale

        # Scale DWT and cargo (volume scales as ~L²·B)
        dp.mission.DWT_target = base.mission.DWT_target * (scale ** 2.8)

        # Scale hotel/aux loads modestly (not linearly)
        dp.propulsion.hotel_load_kW = base.propulsion.hotel_load_kW * (scale ** 0.6)

        family.append(dp)

    return family
