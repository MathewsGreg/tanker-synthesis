"""
synthesis.py
============
Main synthesis runner for the Greg Tanker Design Family.

Orchestrates all discipline modules across the 100–200m LOA design family.
Produces:
  1. Per-design-point detailed results
  2. Family trade curves (speed-power, range, cost vs LOA)
  3. CSV export for further analysis
  4. Console summary tables

Usage:
    python synthesis.py
    python synthesis.py --loa 150          # single point
    python synthesis.py --sweep            # full 100-200m sweep
    python synthesis.py --speed-survey 150 # speed-power curve at 150m

Greg Tanker Synthesis Model — v0.1
"""

import sys
import os
import argparse
import csv
import json
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.design_point import DesignPoint, HullParams, MissionParams, PropulsionParams, make_design_family
from modules.resistance_holtrop import compute_resistance
from modules.weights_watson import compute_weights
from modules.stability_prelim import compute_stability
from modules.cost_parametric import compute_cost


# ---------------------------------------------------------------------------
# Full synthesis pass: run all modules on a DesignPoint
# ---------------------------------------------------------------------------

def run_synthesis(dp: DesignPoint, verbose: bool = False) -> DesignPoint:
    """Run all discipline modules sequentially on a DesignPoint."""
    compute_resistance(dp)
    compute_weights(dp)
    compute_stability(dp)
    compute_cost(dp)
    if verbose:
        print(dp.summary())
    return dp


# ---------------------------------------------------------------------------
# Baseline design points for the three-point family
# ---------------------------------------------------------------------------

def make_baseline_family() -> list:
    """
    REVISED three-point anchor family: 120m – 200m LOA
    All ships: 15.0 kn design speed, 4,500 nm range, MAN 32/44CR engine.

      T120 — 120m class, ~8,000 DWT — minimum family size, INDOPACOM coastal
      T150 — 150m class, ~22,000 DWT — primary INDOPACOM logistics tanker
      T180 — 180m class, ~38,000 DWT — forward area bulk replenishment

    Mission: Pearl Harbor → Guam (3,500 nm) + 30% margin = 4,500 nm at 15 kn.
    Engine: MAN 32/44CR throughout. Aux: 3 × MAN 9L23/30H.
    Jones Act / ABS SVR / MARPOL compliant.
    """

    # --- T120: Minimum-size INDOPACOM logistics tanker ---
    t120 = DesignPoint(name="T120")
    t120.hull    = HullParams(LOA=120, LBP=116.0, B=18.2, D=9.6, T=6.6, Cb=0.78, Cm=0.985, Cwp=0.87)
    t120.mission = MissionParams(Vs_kn=15.0, range_nm=4500, DWT_target=8000, crew_target=10, unrep_stations=1)
    t120.propulsion = PropulsionParams(hotel_load_kW=500, autonomy_load_kW=150, sea_margin=0.15, engine_margin=0.85)

    # --- T150: Primary logistics tanker ---
    t150 = DesignPoint(name="T150")
    t150.hull    = HullParams(LOA=150, LBP=145.1, B=22.7, D=12.0, T=8.3, Cb=0.78, Cm=0.985, Cwp=0.87)
    t150.mission = MissionParams(Vs_kn=15.0, range_nm=4500, DWT_target=22000, crew_target=10, unrep_stations=1)
    t150.propulsion = PropulsionParams(hotel_load_kW=600, autonomy_load_kW=150, sea_margin=0.15, engine_margin=0.85)

    # --- T180: Large forward replenishment tanker ---
    t180 = DesignPoint(name="T180")
    t180.hull    = HullParams(LOA=180, LBP=174.1, B=27.3, D=14.4, T=9.9, Cb=0.78, Cm=0.985, Cwp=0.87)
    t180.mission = MissionParams(Vs_kn=15.0, range_nm=4500, DWT_target=38000, crew_target=12, unrep_stations=1)
    t180.propulsion = PropulsionParams(hotel_load_kW=750, autonomy_load_kW=180, sea_margin=0.15, engine_margin=0.85)

    return [t120, t150, t180]


# ---------------------------------------------------------------------------
# Speed-power survey at a fixed design point
# ---------------------------------------------------------------------------

def speed_survey(base_dp: DesignPoint, speed_range=None) -> list:
    """Generate speed-power data from 8 to max+2 knots."""
    import copy
    if speed_range is None:
        vmax = base_dp.mission.Vs_kn + 2.0
        speed_range = [round(v * 0.5) * 0.5 for v in range(int(8.0/0.5), int(vmax/0.5) + 1)]
        speed_range = [v for v in [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
                       if v <= vmax]

    results = []
    for v in speed_range:
        dp = copy.deepcopy(base_dp)
        dp.mission.Vs_kn = v
        dp.name = f"{base_dp.name}_V{int(v)}"
        compute_resistance(dp)
        results.append({
            "Vs_kn": v,
            "Fn":    dp.resistance.Fn,
            "Rt_kN": dp.resistance.Rt_kN,
            "PE_kW": dp.resistance.PE_kW,
            "PB_kW": dp.power.PB_kW,
            "fuel_t_day": dp.power.fuel_rate_t_per_day,
        })
    return results


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def results_to_dict(dp: DesignPoint) -> dict:
    h, m, r, p, w, s, c = (dp.hull, dp.mission, dp.resistance,
                             dp.power, dp.weights, dp.stability, dp.cost)
    return {
        "name":          dp.name,
        "LOA_m":         h.LOA,
        "LBP_m":         h.LBP,
        "B_m":           h.B,
        "D_m":           h.D,
        "T_m":           h.T,
        "Cb":            h.Cb,
        "L_B":           round(h.LB_ratio, 3),
        "disp_t":        round(h.displacement_t, 0),
        "Vs_kn":         m.Vs_kn,
        "range_nm":      m.range_nm,
        "DWT_target_t":  m.DWT_target,
        "crew":          m.crew_target,
        "Fn":            round(r.Fn, 4),
        "Rt_kN":         round(r.Rt_kN, 1),
        "PE_kW":         round(r.PE_kW, 0),
        "PB_kW":         round(p.PB_kW, 0),
        "fuel_t_day":    round(p.fuel_rate_t_per_day, 2),
        "fuel_cap_t":    round(p.fuel_capacity_t, 0),
        "eta_propulsive":round(p.propulsive_efficiency, 3),
        "W_steel_t":     round(w.W_steel_t, 0),
        "W_lightship_t": round(w.W_lightship_t, 0),
        "W_DWT_t":       round(w.W_deadweight_t, 0),
        "W_cargo_t":     round(w.W_cargo_t, 0),
        "GM_loaded_m":   round(s.GM_loaded_m, 3),
        "freeboard_m":   round(s.freeboard_m, 2),
        "stable":        s.intact_stable,
        "build_cost_M":  round(c.build_cost_M, 1),
        "opex_M_yr":     round(c.annual_opex_M, 1),
    }


def print_family_table(family_results: list) -> None:
    """Print a formatted summary table to console."""
    keys = ["name", "LOA_m", "Vs_kn", "Fn", "PB_kW", "fuel_t_day",
            "W_DWT_t", "W_cargo_t", "GM_loaded_m", "stable", "build_cost_M"]
    header = f"{'Design':<8} {'LOA':>6} {'Vs':>5} {'Fn':>7} {'PB':>7} {'Fuel':>8} "
    header += f"{'DWT':>8} {'Cargo':>8} {'GM':>6} {'Stbl':>5} {'$Build':>8}"
    print("\n" + "="*80)
    print(header)
    print("-"*80)
    for r in family_results:
        print(f"{r['name']:<8} {r['LOA_m']:>6.0f} {r['Vs_kn']:>5.1f} "
              f"{r['Fn']:>7.4f} {r['PB_kW']:>7.0f} {r['fuel_t_day']:>8.2f} "
              f"{r['W_DWT_t']:>8.0f} {r['W_cargo_t']:>8.0f} "
              f"{r['GM_loaded_m']:>6.2f} {'✓' if r['stable'] else '✗':>5} "
              f"${r['build_cost_M']:>6.1f}M")
    print("="*80 + "\n")


def export_csv(family_results: list, path: str) -> None:
    if not family_results:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=family_results[0].keys())
        writer.writeheader()
        writer.writerows(family_results)
    print(f"  Exported: {path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Greg Tanker Synthesis Model v0.1")
    parser.add_argument("--loa",   type=float, default=None, help="Single LOA [m]")
    parser.add_argument("--sweep", action="store_true",      help="100-200m LOA sweep")
    parser.add_argument("--speed-survey", type=float, default=None,
                        metavar="LOA", help="Speed-power survey at given LOA")
    parser.add_argument("--output", default="outputs", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("\n  Greg Tanker Synthesis Model — v0.1")
    print("  INDOPACOM Distributed Logistics Product Tanker Family")
    print("  ABS SVR / Jones Act  |  100–200 m LOA\n")

    # ---- Default: run three-point baseline family ----
    if args.loa is None and not args.sweep and args.speed_survey is None:
        print("  Running three-point baseline family (T120 / T150 / T185)...\n")
        family = make_baseline_family()
        family_results = []
        for dp in family:
            run_synthesis(dp, verbose=True)
            family_results.append(results_to_dict(dp))

        print_family_table(family_results)
        export_csv(family_results, f"{args.output}/family_baseline.csv")

        # Speed surveys for T150
        print("  Speed-power survey — T150 (primary logistics tanker):")
        sv = speed_survey(family[1])
        print(f"\n  {'Speed (kn)':>10} {'Fn':>8} {'Rt (kN)':>10} "
              f"{'PE (kW)':>10} {'PB (kW)':>10} {'Fuel (t/d)':>12}")
        print("  " + "-"*65)
        for row in sv:
            print(f"  {row['Vs_kn']:>10.1f} {row['Fn']:>8.4f} {row['Rt_kN']:>10.1f} "
                  f"  {row['PE_kW']:>8.0f} {row['PB_kW']:>10.0f} {row['fuel_t_day']:>12.2f}")
        print()

    # ---- Single LOA point ----
    elif args.loa is not None:
        dp = DesignPoint(name=f"T{int(args.loa)}")
        dp.hull.LOA = args.loa
        dp.hull.LBP = args.loa * 0.967
        run_synthesis(dp, verbose=True)

    # ---- Full LOA sweep ----
    elif args.sweep:
        print("  Running full 100–200 m LOA sweep (5m steps)...\n")
        base = make_baseline_family()[1]   # T150 as base
        loa_range = list(range(120, 201, 5))
        sweep_family = make_design_family(loa_range, base=base, name_prefix="T")
        sweep_results = []
        for dp in sweep_family:
            run_synthesis(dp)
            sweep_results.append(results_to_dict(dp))
        print_family_table(sweep_results)
        export_csv(sweep_results, f"{args.output}/family_sweep.csv")

    # ---- Speed survey ----
    elif args.speed_survey is not None:
        base = make_baseline_family()[1]
        base.hull.LOA = args.speed_survey
        base.hull.LBP = args.speed_survey * 0.967
        sv = speed_survey(base)
        print(f"  Speed-power survey for LOA={args.speed_survey:.0f}m:\n")
        for row in sv:
            print(f"  Vs={row['Vs_kn']:.1f}kn  Fn={row['Fn']:.4f}  "
                  f"Rt={row['Rt_kN']:.1f}kN  PE={row['PE_kW']:.0f}kW  "
                  f"PB={row['PB_kW']:.0f}kW  fuel={row['fuel_t_day']:.2f}t/day")


if __name__ == "__main__":
    main()
