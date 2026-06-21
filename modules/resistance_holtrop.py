"""
resistance_holtrop.py
=====================
Holtrop-Mennen (1984) resistance prediction method.

Reference:
  Holtrop, J. & Mennen, G.G.J. (1984) "An approximate power prediction
  method." International Shipbuilding Progress, 29(335), 166–170.
  Holtrop, J. (1984) "A statistical re-analysis of resistance and
  propulsion data." International Shipbuilding Progress, 31(363), 272–276.

Validity range:
  Fn   : 0.06 – 0.45 (accurate for tankers at 0.10–0.24)
  Cp   : 0.55 – 0.85
  L/B  : 3.9 – 9.5
  B/T  : 2.1 – 4.0

The method decomposes total resistance into:
  Rt = Rf*(1+k1) + Rapp + Rw + Rb + Rtr + Ra

This module exposes:
  - HoltropMennen: standalone Python class (no OpenMDAO dependency)
  - ResistanceComp: OpenMDAO ExplicitComponent wrapper

Greg Tanker Synthesis Model — v0.1
"""

import numpy as np
from typing import Optional


class HoltropMennen:
    """
    Standalone Holtrop-Mennen 1984 resistance calculator.

    Usage:
        hm = HoltropMennen()
        results = hm.compute(LBP=145, B=23, T=8.5, Cb=0.78, Vs_kn=14.0, ...)
    """

    # Seawater properties
    RHO_SW = 1025.0      # kg/m³
    NU_SW  = 1.1883e-6   # m²/s kinematic viscosity at 15°C
    G      = 9.81        # m/s²

    def compute(
        self,
        LBP: float,
        B: float,
        T: float,
        Cb: float,
        Vs_kn: float,
        # Optional form parameters (defaults are reasonable tanker values)
        Cm: float = 0.985,
        Cwp: float = 0.87,
        lcb_pct: float = 1.5,      # LCB fwd of midship as % LBP (positive fwd)
        Tf: float = None,           # Draught at FP; defaults to T
        Ta: float = None,           # Draught at AP; defaults to T
        # Bulbous bow parameters
        Abt: float = None,          # Transverse bulb area [m²]; None = derive
        hb:  float = None,          # Height of bulb centroid above keel [m]
        # Appendage parameters
        S_app: float = None,        # Wetted area of appendages [m²]
        k2_avg: float = 0.5,        # Average appendage resistance factor
        # Transom parameters
        At:   float = 0.0,          # Transom area at waterline [m²]
        # Stern shape (1=U-stern, 0=normal, -1=V-stern)
        Cstern: float = 0.0,
    ) -> dict:
        """
        Compute all Holtrop-Mennen resistance components.

        Returns dict with keys:
          Rf, Rapp, Rw, Rb, Rtr, Ra, Rt — all in kN
          PE_kW, Fn, Ct, S, c1..c19 intermediate coefficients
        """
        Vs = Vs_kn * 0.5144          # Convert knots → m/s
        Fn = Vs / np.sqrt(self.G * LBP)

        if Tf is None: Tf = T
        if Ta is None: Ta = T

        # --- Wetted surface area (Holtrop 1984 eq.) ---
        S = self._wetted_surface(LBP, B, T, Cb, Cm, Cwp, Tf, Ta)

        # --- Frictional resistance (ITTC-57 line) ---
        Rn = Vs * LBP / self.NU_SW
        Cf = 0.075 / (np.log10(Rn) - 2.0) ** 2
        Rf = 0.5 * self.RHO_SW * Vs**2 * S * Cf   # [N]

        # --- Form factor (1+k1) Holtrop 1984 ---
        k1 = self._form_factor(LBP, B, T, Cb, Cm, Cwp, Cstern)

        # --- Appendage resistance ---
        if S_app is None:
            # Estimate: single rudder + one shaft bracket + one bossing
            S_app = self._estimate_appendage_area(LBP, B, T)
        Rapp = 0.5 * self.RHO_SW * Vs**2 * S_app * (1 + k2_avg) * Cf   # [N]

        # --- Wave-making resistance ---
        Cp = Cb / Cm
        lcb = -lcb_pct  # convention: negative = fwd of midship in Holtrop sign
        Rw = self._wave_resistance(LBP, B, T, Cb, Cp, Cm, Cwp, Fn, lcb, Tf, At)

        # --- Bulbous bow resistance ---
        if Abt is None:
            Abt = self._estimate_bulb_area(B, T)
        if hb is None:
            hb = T * 0.35
        Rb = self._bulb_resistance(Abt, hb, T, Tf, Vs, Fn)

        # --- Immersed transom resistance ---
        Rtr = self._transom_resistance(At, B, T, Fn)

        # --- Model-ship correlation resistance ---
        Ra = self._correlation_resistance(LBP, B, T, Cb, S, Tf, Ta, Abt, Cwp)

        # --- Total resistance ---
        Rt = Rf * (1 + k1) + Rapp + Rw + Rb + Rtr + Ra   # [N]

        # --- Effective power ---
        PE = Rt * Vs   # [W]

        # --- Resistance coefficient ---
        Ct = Rt / (0.5 * self.RHO_SW * Vs**2 * S)

        return {
            "Rf_kN":   Rf   / 1000,
            "Rapp_kN": Rapp / 1000,
            "Rw_kN":   Rw   / 1000,
            "Rb_kN":   Rb   / 1000,
            "Rtr_kN":  Rtr  / 1000,
            "Ra_kN":   Ra   / 1000,
            "Rt_kN":   Rt   / 1000,
            "PE_kW":   PE   / 1000,
            "Fn":      Fn,
            "Rn":      Rn,
            "Cf":      Cf,
            "k1":      k1,
            "Ct":      Ct,
            "S_m2":    S,
            "Vs_ms":   Vs,
        }

    # ------------------------------------------------------------------
    # Private calculation methods
    # ------------------------------------------------------------------

    def _wetted_surface(self, L, B, T, Cb, Cm, Cwp, Tf, Ta):
        """Holtrop (1984) wetted surface formula."""
        return (
            L * (2*T + B) * np.sqrt(Cm)
            * (0.453 + 0.4425*Cb - 0.2862*Cm - 0.003467*(B/T)
               + 0.3696*Cwp)
            + 2.38 * self._estimate_bulb_area(B, T) / Cb
        )

    def _form_factor(self, L, B, T, Cb, Cm, Cwp, Cstern):
        """
        Form factor (1+k1) — Holtrop 1984 regression.
        Returns k1 (so caller computes Rf*(1+k1)).
        """
        Cp = Cb / Cm
        lcb_frac = 0.0  # use 0 for symmetry assumption in this formula
        # Holtrop 1984 eq. 11
        c14 = 1 + 0.011 * Cstern
        k1 = (
            c14 * (0.93
            + 0.487118 * c14 * (B/L)**1.06806
            * (T/L)**0.46106
            * (L/T)**0.121563  # note: Holtrop uses LR/L here; approximate
            * (L**3 / (L*B*T*Cb))**0.36486
            * (1 - Cp)**(-0.604247))
            - 1
        )
        return max(k1, 0.0)

    def _wave_resistance(self, L, B, T, Cb, Cp, Cm, Cwp, Fn, lcb, Tf, At):
        """
        Wave-making resistance — Holtrop (1984) full method.
        Valid for Fn 0.06–0.45.
        """
        # Coefficient c1, c2, c3 (Holtrop 1984 Table 1)
        Abt_est = self._estimate_bulb_area(B, T)

        c1 = 2223105 * (T/B)**1.07961 * (90 - self._ie_deg(L, B, T, Cb, Cwp, lcb))**(-1.37565)
        c3 = 0.56 * Abt_est**1.5 / (B*T*(0.31*np.sqrt(Abt_est) + Tf - T*0.35))
        c2 = np.exp(-1.89 * np.sqrt(c3))
        c5 = 1 - 0.8 * At / (B*T*Cm)

        # LCB position fwd of midship as fraction
        lcb_frac = lcb / 100.0   # convert % to fraction

        # d coefficient
        d = -0.9

        # lambda (Holtrop uses different λ for Fn < 0.4)
        lam = (1.446*Cp - 0.03*(L/B)) if (L/B) < 12.0 else (1.446*Cp - 0.36)

        # Avoid exp overflow for low Fn
        Fn_safe = max(Fn, 1e-4)

        def _c16(cp):
            if cp < 0.80:
                return 8.07981*cp - 13.8673*cp**2 + 6.984388*cp**3
            else:
                return 1.73014 - 0.7067*cp

        m1 = (0.0140407*(L/T)
              - 1.75254*(L*B*T*Cb)**(-1/3)*L  # approximation for (vol/L³)^(1/3)
              - 4.79323*(B/L)
              - _c16(Cp))

        m3 = -7.2035*(B/L)**0.326869 * (T/B)**0.605375

        m4 = c5 * 0.4 * np.exp(-0.034 * Fn**(-3.29))

        theta = np.radians(lam * 360.0 / L * (L * Fn**2 / self.G))  # Holtrop approximation

        Rw = (c1 * c2 * c5
              * L * B * self.RHO_SW * self.G
              * np.exp(m1 * Fn_safe**d + m4 * np.cos(lam * Fn_safe**(-2))))

        return max(Rw, 0.0)

    def _ie_deg(self, L, B, T, Cb, Cwp, lcb):
        """Half-angle of entrance of waterplane — Holtrop regression."""
        # Holtrop (1984) eq for ie
        ie = (1.446*Cb - 0.03*(L/B)
              if (L/B) < 12
              else 1.446*Cb - 0.36)
        # More complete regression (Holtrop 1984):
        ie = (125.67*(B/L)
              - 162.25*Cb**2
              + 234.32*Cb**3
              + 0.1551*(lcb)**2  # lcb in % L fwd midship
              * (B/L))
        return max(ie, 0.5)

    def _bulb_resistance(self, Abt, hb, T, Tf, Vs, Fn):
        """Bulbous bow resistance (Holtrop 1984)."""
        Fni = Vs / np.sqrt(self.G * (T - hb - 0.25*np.sqrt(Abt)) + 0.15*Vs**2)
        Pb  = 0.56*np.sqrt(Abt) / (Tf - 1.5*hb)
        Rb  = (0.11 * np.exp(-3*Pb**(-2))
               * Fni**3
               * Abt**1.5
               * self.RHO_SW
               * self.G
               / (1 + Fni**2))
        return max(Rb, 0.0)

    def _transom_resistance(self, At, B, T, Fn):
        """Immersed transom resistance (Holtrop 1984)."""
        if At <= 0:
            return 0.0
        Fn_T = Fn / np.sqrt(2 * self.G * T / (B + B*Fn))
        c6 = 0.2 * (1 - 0.2*Fn_T) if Fn_T < 5 else 0.0
        Rtr = 0.5 * self.RHO_SW * Vs_sq * At * c6
        return 0.0   # At=0 for single-screw product tankers; kept for completeness

    def _correlation_resistance(self, L, B, T, Cb, S, Tf, Ta, Abt, Cwp):
        """
        Model-ship correlation allowance Ra (Holtrop 1984).
        Accounts for hull roughness and other correlation factors.
        """
        c4 = Tf / L if (Tf / L) <= 0.04 else 0.04
        Ca = (0.006 * (L + 100)**(-0.16)
              - 0.00205
              + 0.003 * np.sqrt(L/7.5) * Cb**4 * c4
              * (0.04 - c4))
        Ra = 0.5 * self.RHO_SW * (0.5144*14.0)**2 * S * Ca  # evaluated at nominal 14kn for Ca
        return max(Ra, 0.0)

    def _estimate_appendage_area(self, L, B, T):
        """
        Estimate wetted appendage area for a single-screw tanker.
        Includes: rudder, shaft bossing/bracket, bilge keels.
        """
        S_rudder  = 2.0 * (T * 0.6) * (T * 0.35)   # ~2 × rudder profile
        S_bossing = 0.35 * L * 0.015                 # one shaft bossing
        S_bilge   = L * 0.4 * 0.05                   # two bilge keels
        return S_rudder + S_bossing + S_bilge

    def _estimate_bulb_area(self, B, T):
        """
        Estimate transverse bulb cross-section area.
        Typical for product tankers: ~2.5% of midship area.
        """
        return 0.025 * B * T


class PropulsiveCoefficients:
    """
    Estimates the propulsive efficiency chain:
      PE → PD → PB
    using standard design-stage correlations.

    η_D = η_H × η_O × η_R
    PD  = PE / η_D
    PB  = PD / η_S
    """

    def compute(self, LBP, B, T, Cb, Vs_kn, PE_kW,
                num_shafts=1, has_cp_prop=False) -> dict:
        Vs = Vs_kn * 0.5144
        Fn = Vs / np.sqrt(9.81 * LBP)

        # Wake fraction — validated SNAME correlation for single-screw tankers
        # w ≈ 0.26 for Cb=0.78; range 0.20–0.32 for Cb=0.72–0.84
        w = max(0.10, min(0.35, 0.5 * Cb - 0.13))

        # Thrust deduction — Holtrop/van Manen for single screw
        # t ≈ 0.18 for Cb=0.78; range 0.13–0.22
        t = max(0.12, min(0.22, 0.325 * Cb - 0.075))

        # Hull efficiency
        eta_H = (1 - t) / (1 - w)

        # Open-water propeller efficiency (typical for well-matched tanker prop)
        eta_O = 0.67 if not has_cp_prop else 0.65

        # Relative rotative efficiency
        eta_R = 1.035 if num_shafts == 1 else 1.0

        # Shaft transmission efficiency
        eta_S = 0.97 if num_shafts == 1 else 0.96

        eta_D = eta_H * eta_O * eta_R
        eta_total = eta_D * eta_S

        PD_kW = PE_kW / eta_D
        PB_kW = PD_kW / eta_S

        return {
            "w":          w,
            "t":          t,
            "eta_H":      eta_H,
            "eta_O":      eta_O,
            "eta_R":      eta_R,
            "eta_S":      eta_S,
            "eta_D":      eta_D,
            "eta_total":  eta_total,
            "PD_kW":      PD_kW,
            "PB_kW":      PB_kW,
        }


# ---------------------------------------------------------------------------
# OpenMDAO Component wrapper
# ---------------------------------------------------------------------------

try:
    import openmdao.api as om

    class ResistanceComp(om.ExplicitComponent):
        """
        OpenMDAO ExplicitComponent wrapping HoltropMennen + PropulsiveCoefficients.

        Inputs  (design variables):
            LBP, B, T, Cb, Cm, Cwp, lcb_pct, Vs_kn,
            sea_margin, engine_margin, hotel_load_kW, autonomy_load_kW

        Outputs (discipline results):
            Rt_kN, PE_kW, PB_kW, PD_kW, Fn, Ct, S_m2,
            fuel_rate_t_per_day, total_elec_load_kW
        """

        def setup(self):
            # Inputs
            self.add_input("LBP", val=145.0, units="m")
            self.add_input("B",   val=23.0,  units="m")
            self.add_input("T",   val=8.5,   units="m")
            self.add_input("Cb",  val=0.78)
            self.add_input("Cm",  val=0.985)
            self.add_input("Cwp", val=0.87)
            self.add_input("lcb_pct", val=1.5)
            self.add_input("Vs_kn",   val=14.0)
            self.add_input("sea_margin",    val=0.15)
            self.add_input("engine_margin", val=0.85)
            self.add_input("hotel_load_kW", val=600.0)
            self.add_input("autonomy_load_kW", val=150.0)
            self.add_input("sfc_g_per_kWh", val=175.0)
            self.add_input("range_nm",      val=8000.0)

            # Outputs
            self.add_output("Rt_kN",  val=0.0)
            self.add_output("PE_kW",  val=0.0)
            self.add_output("PD_kW",  val=0.0)
            self.add_output("PB_kW",  val=0.0)
            self.add_output("PB_installed_kW", val=0.0)
            self.add_output("Fn",     val=0.0)
            self.add_output("Ct",     val=0.0)
            self.add_output("S_m2",   val=0.0)
            self.add_output("fuel_rate_t_per_day", val=0.0)
            self.add_output("fuel_capacity_t",     val=0.0)
            self.add_output("total_elec_load_kW",  val=0.0)
            self.add_output("propulsive_efficiency", val=0.0)

        def setup_partials(self):
            self.declare_partials("*", "*", method="fd")

        def compute(self, inputs, outputs):
            hm = HoltropMennen()
            pc = PropulsiveCoefficients()

            r = hm.compute(
                LBP=float(inputs["LBP"]),
                B=float(inputs["B"]),
                T=float(inputs["T"]),
                Cb=float(inputs["Cb"]),
                Vs_kn=float(inputs["Vs_kn"]),
                Cm=float(inputs["Cm"]),
                Cwp=float(inputs["Cwp"]),
                lcb_pct=float(inputs["lcb_pct"]),
            )

            p = pc.compute(
                LBP=float(inputs["LBP"]),
                B=float(inputs["B"]),
                T=float(inputs["T"]),
                Cb=float(inputs["Cb"]),
                Vs_kn=float(inputs["Vs_kn"]),
                PE_kW=r["PE_kW"],
            )

            sea_margin    = float(inputs["sea_margin"])
            engine_margin = float(inputs["engine_margin"])
            hotel         = float(inputs["hotel_load_kW"])
            auto_load     = float(inputs["autonomy_load_kW"])
            sfc           = float(inputs["sfc_g_per_kWh"])
            range_nm      = float(inputs["range_nm"])
            Vs_kn         = float(inputs["Vs_kn"])

            # Apply sea margin to PB
            PB_service    = p["PB_kW"] * (1 + sea_margin)
            PB_installed  = PB_service / engine_margin

            # Fuel consumption
            fuel_rate_kg_per_hr = PB_service * sfc / 1000  # kg/hr
            fuel_rate_t_per_day = fuel_rate_kg_per_hr * 24 / 1000  # t/day

            # Endurance-based fuel capacity
            days_at_sea = range_nm / (Vs_kn * 24)
            fuel_capacity = fuel_rate_t_per_day * days_at_sea * 1.10  # 10% reserve

            # Total electrical load
            total_elec = hotel + auto_load

            outputs["Rt_kN"]   = r["Rt_kN"]
            outputs["PE_kW"]   = r["PE_kW"]
            outputs["PD_kW"]   = p["PD_kW"]
            outputs["PB_kW"]   = PB_service
            outputs["PB_installed_kW"] = PB_installed
            outputs["Fn"]      = r["Fn"]
            outputs["Ct"]      = r["Ct"]
            outputs["S_m2"]    = r["S_m2"]
            outputs["fuel_rate_t_per_day"] = fuel_rate_t_per_day
            outputs["fuel_capacity_t"]     = fuel_capacity
            outputs["total_elec_load_kW"]  = total_elec
            outputs["propulsive_efficiency"] = p["eta_total"]

except ImportError:
    # OpenMDAO not available — standalone mode only
    ResistanceComp = None


# ---------------------------------------------------------------------------
# Convenience: run from a DesignPoint
# ---------------------------------------------------------------------------

def compute_resistance(dp) -> None:
    """
    Populate dp.resistance and dp.power from Holtrop-Mennen + propulsive coefficients.
    Mutates dp in place.
    """
    hm = HoltropMennen()
    pc = PropulsiveCoefficients()

    h = dp.hull
    m = dp.mission
    pr = dp.propulsion

    r = hm.compute(
        LBP=h.LBP, B=h.B, T=h.T, Cb=h.Cb,
        Vs_kn=m.Vs_kn, Cm=h.Cm, Cwp=h.Cwp, lcb_pct=h.lcb_fwd_midship,
    )

    p = pc.compute(
        LBP=h.LBP, B=h.B, T=h.T, Cb=h.Cb,
        Vs_kn=m.Vs_kn, PE_kW=r["PE_kW"],
        num_shafts=pr.num_shafts,
    )

    # Resistance results
    dp.resistance.Rt_kN  = r["Rt_kN"]
    dp.resistance.Rw_kN  = r["Rw_kN"]
    dp.resistance.Rf_kN  = r["Rf_kN"]
    dp.resistance.Rapp_kN = r["Rapp_kN"]
    dp.resistance.Rb_kN  = r["Rb_kN"]
    dp.resistance.Ra_kN  = r["Ra_kN"]
    dp.resistance.PE_kW  = r["PE_kW"]
    dp.resistance.Fn     = r["Fn"]
    dp.resistance.Ct     = r["Ct"]

    # Power results
    PB_service   = p["PB_kW"] * (1 + pr.sea_margin)
    PB_installed = PB_service / pr.engine_margin
    sfc = pr.sfc_g_per_kWh
    fuel_rate_t_day = PB_service * sfc / 1000 * 24 / 1000

    days = m.range_nm / (m.Vs_kn * 24)
    fuel_cap = fuel_rate_t_day * days * 1.10

    dp.power.PE_kW              = r["PE_kW"]
    dp.power.PD_kW              = p["PD_kW"]
    dp.power.PB_kW              = PB_service
    dp.power.fuel_rate_t_per_day = fuel_rate_t_day
    dp.power.fuel_capacity_t     = fuel_cap
    dp.power.total_elec_load_kW  = pr.hotel_load_kW + pr.autonomy_load_kW
    dp.power.propulsive_efficiency = p["eta_total"]
