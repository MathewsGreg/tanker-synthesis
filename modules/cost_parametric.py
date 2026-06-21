"""
cost_parametric.py
==================
Parametric acquisition and lifecycle cost model for product tankers
built in second-tier US Jones Act yards.

Reference basis:
  - NASSCO ECO tanker (~$175M, 49,900 DWT, 204m) — cost ceiling
  - Philly Shipyard MR tankers (~$165M, ~46,000 DWT)
  - Eastern Shipbuilding / VT Halter product tankers (~$55–80M, 12–25k DWT)
  - Watson (1998) parametric cost regression, updated for US yard premium

Jones Act Premium: US yards typically cost 3.5–4.5× Asian yard prices
for the same vessel. This is the dominant cost driver.

Cost breakdown (approximate):
  Steel material:  12–15% of build cost
  Outfit/equip:    25–30%
  Machinery:       15–20%
  Labor:           30–35%
  Margin/overhead: 10–12%

Greg Tanker Synthesis Model — v0.1
"""

import numpy as np


class ParametricCost:

    # Commodity prices and labor rates (2024 baseline, adjust via inflation_factor)
    STEEL_PRICE_PER_T = 1200.0    # USD/tonne fabricated structural steel
    OUTFIT_UNIT       = 850.0     # USD/tonne outfit weight
    MACH_UNIT         = 2200.0    # USD/kW installed power
    LABOR_RATE_USD_HR = 95.0      # USD/man-hour (US Gulf Coast Jones Act)
    MAN_HRS_PER_T_STEEL = 110.0   # Man-hours per tonne of steel (Jones Act 2nd tier)
    JONES_ACT_PREMIUM  = 1.0      # Already embedded in labor rate above

    # Fuel cost
    MARINE_DIESEL_PER_T = 700.0   # USD/tonne (2024 MGO basis)

    # Crew cost
    CREW_COST_PER_PERSON_ANNUAL = 140_000   # USD/person/year (Jones Act deck/eng)
    AUTONOMY_SYSTEM_ANNUAL_MAINT = 0.05     # 5% of autonomy hardware per year

    def compute(self, W_steel_t, W_outfit_t, W_machinery_t, W_autonomy_t,
                PB_kW, LBP, crew, fuel_rate_t_per_day,
                annual_operating_days=300, inflation_factor=1.0) -> dict:
        """
        Parameters
        ----------
        annual_operating_days : int  Days at sea per year
        inflation_factor : float     Apply to all costs (1.0 = 2024 basis)
        """
        f = inflation_factor

        # ----------------------------------------------------------------
        # Acquisition cost
        # ----------------------------------------------------------------
        # Steel material + fabrication
        steel_fab = W_steel_t * self.STEEL_PRICE_PER_T * f
        steel_labor = W_steel_t * self.MAN_HRS_PER_T_STEEL * self.LABOR_RATE_USD_HR * f

        # Outfit
        outfit_cost = W_outfit_t * self.OUTFIT_UNIT * f

        # Autonomy hardware premium (sensors, compute, comm systems)
        autonomy_hw  = W_autonomy_t * 3500.0 * f  # higher unit cost than normal outfit

        # Machinery
        machinery_cost = PB_kW * self.MACH_UNIT * f

        # UNREP station (one station)
        unrep_cost = 2.5e6 * f   # $2.5M per UNREP station

        # ABS survey and classification premium
        abs_premium = 0.015   # 1.5% of build cost

        # Jones Act documentation / compliance cost (flat)
        jones_compliance = 0.5e6 * f

        subtotal = (steel_fab + steel_labor + outfit_cost
                    + autonomy_hw + machinery_cost + unrep_cost
                    + jones_compliance)

        # ABS and owner's margin
        build_cost = subtotal * (1 + abs_premium) * 1.10   # 10% overhead/profit

        # Sanity check against known data points
        # Scale factor: if very different from reference, apply mild correction
        dwt_approx = (1.025 * LBP * 23 * 8.5 * 0.78) - W_steel_t - W_outfit_t - W_machinery_t
        # (rough; actual DWT computed elsewhere)

        # ----------------------------------------------------------------
        # Annual operating cost (OPEX)
        # ----------------------------------------------------------------
        # Fuel
        fuel_annual = (annual_operating_days * fuel_rate_t_per_day
                       * self.MARINE_DIESEL_PER_T * f)

        # Crew (Jones Act wages + benefits)
        crew_annual = crew * self.CREW_COST_PER_PERSON_ANNUAL * f

        # Port dues (approx $50/GT per year)
        GT_approx = 0.4 * (1.025 * LBP * 23 * 8.5 * 0.78)   # rough GT
        port_dues = 50 * GT_approx * f / 1000   # USD

        # Maintenance and repair (2.5% of build per year)
        maint = build_cost * 0.025

        # Autonomy system annual maintenance
        autonomy_maint = autonomy_hw * self.AUTONOMY_SYSTEM_ANNUAL_MAINT

        # Insurance (~0.5% of build)
        insurance = build_cost * 0.005

        annual_opex = fuel_annual + crew_annual + port_dues + maint + autonomy_maint + insurance

        return {
            "steel_cost_M":          (steel_fab + steel_labor) / 1e6,
            "outfit_cost_M":         outfit_cost / 1e6,
            "machinery_cost_M":      machinery_cost / 1e6,
            "autonomy_hw_cost_M":    autonomy_hw / 1e6,
            "unrep_cost_M":          unrep_cost / 1e6,
            "build_cost_M":          build_cost / 1e6,
            "fuel_cost_annual_M":    fuel_annual / 1e6,
            "crew_cost_annual_M":    crew_annual / 1e6,
            "annual_opex_M":         annual_opex / 1e6,
            "cost_per_dwt_USD":      build_cost / max(dwt_approx, 1),
        }


def compute_cost(dp) -> None:
    """Populate dp.cost. Mutates dp in place."""
    pc = ParametricCost()

    h  = dp.hull
    m  = dp.mission
    w  = dp.weights
    pw = dp.power

    r = pc.compute(
        W_steel_t    = w.W_steel_t    if w.W_steel_t > 0 else 1000,
        W_outfit_t   = w.W_outfit_t   if w.W_outfit_t > 0 else 400,
        W_machinery_t= w.W_machinery_t if w.W_machinery_t > 0 else 200,
        W_autonomy_t = 35.0,   # default near-autonomous grade
        PB_kW        = pw.PB_kW if pw.PB_kW > 0 else 5000,
        LBP          = h.LBP,
        crew         = m.crew_target,
        fuel_rate_t_per_day = pw.fuel_rate_t_per_day if pw.fuel_rate_t_per_day > 0 else 10,
    )

    dp.cost.steel_cost_M       = r["steel_cost_M"]
    dp.cost.outfit_cost_M      = r["outfit_cost_M"]
    dp.cost.machinery_cost_M   = r["machinery_cost_M"]
    dp.cost.build_cost_M       = r["build_cost_M"]
    dp.cost.annual_opex_M      = r["annual_opex_M"]
    dp.cost.fuel_cost_annual_M = r["fuel_cost_annual_M"]
    dp.cost.crew_cost_annual_M = r["crew_cost_annual_M"]
