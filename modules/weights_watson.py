"""
weights_watson.py
=================
Parametric weight estimation for product tankers using Watson & Barrass (2002)
regression equations, with updates from Papanikolaou (2014) for modern tankers.

References:
  Watson, D.G.M. (1998) "Practical Ship Design." Elsevier.
  Barrass, C.B. & Derrett, D.R. (2006) "Ship Stability for Masters and Mates."
  Papanikolaou, A. (2014) "Ship Design: Methodologies of Preliminary Design."

Weight groups follow the SWBS (Ship Work Breakdown Structure) at a parametric level:
  Group 1: Hull structure (steel weight)
  Group 2: Propulsion machinery
  Group 3: Electric plant (subset of machinery for diesel-electric)
  Group 4: Command / navigation / autonomy systems
  Group 5: Auxiliary systems
  Group 6: Outfit and furnishings
  Group 7: Cargo systems (piping manifolds, pumps, cargo heating)

Lightship = Groups 1–7
Deadweight = Cargo + Fuel + FW + Stores + Crew + Effects
Displacement = Lightship + Deadweight

Greg Tanker Synthesis Model — v0.1
"""

import numpy as np


class WatsonWeights:
    """
    Parametric weight estimator for product tankers.

    Watson (1998) uses a Steel Weight Coefficient (Ws/E) where:
      E = L*(B+T) + 0.85*L*(D-T) + 0.85*sum(lh) + 0.75*sum(ls)
    For tankers, simplified to:
      E = LBP * (B + D + T/3)   (Watson tanker form)
    """

    def compute(self, LBP, B, D, T, Cb, PB_kW,
                crew=10, autonomy_grade=2, num_cargo_pumps=4,
                cargo_heating=False) -> dict:
        """
        Parameters
        ----------
        autonomy_grade : int
            0 = conventional, 1 = high automation, 2 = near-autonomous,
            3 = fully autonomous (crew=0)
        """

        # ----------------------------------------------------------------
        # Watson E number (structural weight parameter)
        # ----------------------------------------------------------------
        E = LBP * (B + D + T / 3.0)

        # ----------------------------------------------------------------
        # Group 1: Hull steel weight
        # Watson tanker regression: Ws = Cs * E^1.36
        # Cs ~0.032-0.040 for tankers (use 0.034 for double-hull product)
        # ----------------------------------------------------------------
        Cs = 0.034
        W_steel = Cs * E ** 1.36

        # Cb correction (heavier structure for fuller hulls)
        W_steel *= (1 + 0.5 * (Cb - 0.70))

        # ----------------------------------------------------------------
        # Group 2+3: Machinery weight
        # Correlation: W_mach = Cm * PB^0.84
        # Cm ~ 0.72 for medium-speed diesel, 0.85 for diesel-electric
        # ----------------------------------------------------------------
        Cm_mach = 0.72
        W_machinery = Cm_mach * PB_kW ** 0.84 / 1000  # convert to tonnes

        # ----------------------------------------------------------------
        # Group 4: Command / navigation / autonomy systems
        # Conventional bridge + nav: ~80t for 150m tanker
        # Autonomy grade 2: add ~40t for sensors, compute, comms, redundancy
        # ----------------------------------------------------------------
        W_nav_base = 0.50 * (LBP / 150) ** 0.7   # [t], scales with ship size
        autonomy_adder = [0.0, 0.20, 0.45, 0.80]  # added fraction of nav_base
        W_autonomy = W_nav_base * (1 + autonomy_adder[min(autonomy_grade, 3)])

        # ----------------------------------------------------------------
        # Group 5: Auxiliary systems (HVAC, fire, ballast, mooring)
        # Watson: W_aux ~ 0.004 * displacement_vol * rho for tankers
        # ----------------------------------------------------------------
        disp_vol = Cb * LBP * B * T
        W_aux = 0.004 * disp_vol * 1.025 * 0.003  # [t]
        W_aux = max(W_aux, 25.0 + 5.0 * LBP / 100)  # floor

        # ----------------------------------------------------------------
        # Group 6: Outfit (accommodation, safety, deck gear)
        # Watson: W_outfit = Co * L * B
        # Co ~ 0.40–0.55 for tankers; reduce for low crew count
        # ----------------------------------------------------------------
        Co = 0.40 - 0.015 * (10 - min(crew, 10))   # lower Co for smaller crew
        W_outfit = Co * LBP * B / 1000              # [t]

        # UNREP station addition (~15t for one station)
        W_unrep = 15.0   # one standard UNREP hose handling rig

        # ----------------------------------------------------------------
        # Group 7: Cargo systems
        # Piping, pumps (centrifugal, ~250kW each), manifolds, IG system
        # ----------------------------------------------------------------
        W_cargo_sys = (num_cargo_pumps * 4.5          # pump sets
                       + LBP * 0.08                   # piping weight
                       + (12.0 if cargo_heating else 0.0)
                       + 8.0)                         # manifold & misc

        # ----------------------------------------------------------------
        # Lightship total
        # ----------------------------------------------------------------
        W_lightship = (W_steel + W_machinery + W_autonomy
                       + W_aux + W_outfit + W_unrep + W_cargo_sys)

        # Watson margin: 2–4% on lightship for tankers
        W_lightship *= 1.03

        # ----------------------------------------------------------------
        # LCG and VCG estimates (simplified)
        # ----------------------------------------------------------------
        LCG = 0.44 * LBP   # typical tanker LCG from AP, ~44% LBP
        VCG = 0.62 * D      # typical loaded VCG ~62% depth

        # ----------------------------------------------------------------
        # DWT components — to check against target
        # ----------------------------------------------------------------
        # Fuel already computed in resistance module; estimate stores/water
        W_fw     = crew * 0.15 * 30   # FW: 150 kg/person/day × 30 days
        W_stores = crew * 0.08 * 30   # Stores: 80 kg/person/day
        W_lube   = PB_kW * 0.0008    # Lube oil estimate

        return {
            "W_steel_t":      W_steel,
            "W_machinery_t":  W_machinery,
            "W_autonomy_t":   W_autonomy,
            "W_aux_t":        W_aux,
            "W_outfit_t":     W_outfit + W_unrep,
            "W_cargo_sys_t":  W_cargo_sys,
            "W_lightship_t":  W_lightship,
            "LCG_m":          LCG,
            "VCG_m":          VCG,
            "W_fw_t":         W_fw,
            "W_stores_t":     W_stores,
            "W_lube_t":       W_lube,
            "E_number":       E,
        }


def compute_weights(dp) -> None:
    """Populate dp.weights from Watson parametric method. Mutates dp in place."""
    ww = WatsonWeights()

    h  = dp.hull
    m  = dp.mission
    pr = dp.propulsion
    pw = dp.power

    r = ww.compute(
        LBP=h.LBP, B=h.B, D=h.D, T=h.T, Cb=h.Cb,
        PB_kW=pw.PB_kW if pw.PB_kW > 0 else 5000.0,
        crew=m.crew_target,
        autonomy_grade=2,
    )

    disp_t = h.displacement_t
    W_lsw  = r["W_lightship_t"]
    W_fuel = pw.fuel_capacity_t if pw.fuel_capacity_t > 0 else 800.0
    W_dw   = disp_t - W_lsw

    dp.weights.W_steel_t     = r["W_steel_t"]
    dp.weights.W_outfit_t    = r["W_outfit_t"]
    dp.weights.W_machinery_t = r["W_machinery_t"]
    dp.weights.W_lightship_t = W_lsw
    dp.weights.W_deadweight_t = W_dw
    dp.weights.W_cargo_t     = max(W_dw - W_fuel - r["W_fw_t"] - r["W_stores_t"], 0)
    dp.weights.LCG_m         = r["LCG_m"]
    dp.weights.VCG_m         = r["VCG_m"]


# ---------------------------------------------------------------------------
# OpenMDAO Component wrapper
# ---------------------------------------------------------------------------

try:
    import openmdao.api as om

    class WeightsComp(om.ExplicitComponent):

        def setup(self):
            self.add_input("LBP",   val=145.0)
            self.add_input("B",     val=23.0)
            self.add_input("D",     val=12.5)
            self.add_input("T",     val=8.5)
            self.add_input("Cb",    val=0.78)
            self.add_input("PB_kW", val=5000.0)
            self.add_input("crew",  val=10.0)

            self.add_output("W_steel_t",     val=0.0)
            self.add_output("W_machinery_t", val=0.0)
            self.add_output("W_lightship_t", val=0.0)
            self.add_output("W_deadweight_t",val=0.0)
            self.add_output("W_cargo_t",     val=0.0)
            self.add_output("LCG_m",         val=0.0)
            self.add_output("VCG_m",         val=0.0)

        def setup_partials(self):
            self.declare_partials("*", "*", method="fd")

        def compute(self, inputs, outputs):
            ww = WatsonWeights()
            r = ww.compute(
                LBP=float(inputs["LBP"]),
                B=float(inputs["B"]),
                D=float(inputs["D"]),
                T=float(inputs["T"]),
                Cb=float(inputs["Cb"]),
                PB_kW=float(inputs["PB_kW"]),
                crew=int(inputs["crew"]),
            )
            Cb  = float(inputs["Cb"])
            LBP = float(inputs["LBP"])
            B   = float(inputs["B"])
            T   = float(inputs["T"])
            disp_t = Cb * LBP * B * T * 1.025

            outputs["W_steel_t"]     = r["W_steel_t"]
            outputs["W_machinery_t"] = r["W_machinery_t"]
            outputs["W_lightship_t"] = r["W_lightship_t"]
            outputs["W_deadweight_t"]= max(disp_t - r["W_lightship_t"], 0)
            outputs["W_cargo_t"]     = max(disp_t - r["W_lightship_t"] - 800 - r["W_fw_t"], 0)
            outputs["LCG_m"]         = r["LCG_m"]
            outputs["VCG_m"]         = r["VCG_m"]

except ImportError:
    WeightsComp = None
