"""
stability_prelim.py
===================
Preliminary (concept design) stability calculations for product tankers.

Computes:
  - KB (centre of buoyancy above keel) — Morrish formula
  - BM_T (transverse metacentric radius) — from waterplane area
  - GM_T in lightship and loaded conditions
  - Freeboard check vs Load Line Convention (simplified)
  - Basic intact stability pass/fail (IMO IS Code 2008, criteria 2.1 only)

No full loading manual at this stage — this screens infeasible designs.
Full NAPA stability book is a later-stage deliverable.

Greg Tanker Synthesis Model — v0.1
"""

import numpy as np


class PrelimStability:

    def compute(self, LBP, B, D, T, Cb, Cwp, Cm,
                W_lightship_t, VCG_lightship_m,
                W_fuel_t=0, W_cargo_t=0) -> dict:
        """
        Parameters
        ----------
        VCG_lightship_m : float  Vertical centre of gravity, lightship [m from keel]

        Returns
        -------
        dict with stability indicators and pass/fail flags
        """
        g = 9.81
        rho = 1.025

        # ----------------------------------------------------------------
        # Hydrostatic properties at design draught
        # ----------------------------------------------------------------
        # Waterplane area
        Awp = Cwp * LBP * B

        # Volume of displacement
        V = Cb * LBP * B * T

        # KB — Morrish / Normand formula (very accurate for tankers)
        KB = T * (5/6 - Cb / (3 * Cwp))

        # BM_T — transverse metacentric radius
        # I_T ≈ Cwp^3 * L * B^3 / 12  (approximate for ship-shaped waterplanes)
        It = (Cwp ** 3) * LBP * (B ** 3) / 12.0   # simplified; actual needs form
        # Better approximation for tankers:
        It = (0.0735 + 0.0667 * Cwp) * LBP * B ** 3
        BMt = It / V

        # KM_T
        KMt = KB + BMt

        # ----------------------------------------------------------------
        # Lightship GM
        # ----------------------------------------------------------------
        KG_lightship = VCG_lightship_m
        GM_lightship = KMt - KG_lightship

        # ----------------------------------------------------------------
        # Loaded condition (full cargo, full fuel)
        # Using simplified loaded KG from weight stacking
        # ----------------------------------------------------------------
        disp_lightship = W_lightship_t
        disp_loaded    = V * rho   # full load displacement [t]

        # Cargo KG (approx midheight of cargo tanks)
        KG_cargo  = T * 0.50   # tanks fill to ~50% depth centroid
        KG_fuel   = T * 0.30   # double-bottom fuel tanks ~30% T

        # Moment sum
        M_light   = disp_lightship * KG_lightship
        M_cargo   = W_cargo_t * KG_cargo
        M_fuel    = W_fuel_t  * KG_fuel

        denom = disp_lightship + W_cargo_t + W_fuel_t
        if denom > 0:
            KG_loaded = (M_light + M_cargo + M_fuel) / denom
        else:
            KG_loaded = KG_lightship

        # Free surface correction (simplified — one slack cargo tank ~5% B²)
        FSC = rho * (0.05 * LBP * B**3 / 12) / (V * rho)

        KG_loaded_corr = KG_loaded + FSC
        GM_loaded = KMt - KG_loaded_corr

        # ----------------------------------------------------------------
        # Freeboard (simplified Plimsoll calculation)
        # Load Line Convention Reg. 28 tabular value approximation
        # ----------------------------------------------------------------
        # Freeboard length (use LBP)
        # Tabular freeboard from Type B vessel table (oil tankers are B-60)
        # Approximate regression: fb_tabular ≈ 1600 + 17*(L-100) for L 100-200m
        fb_tabular_mm = 1600 + 17 * max(LBP - 100, 0)   # [mm]
        fb_tabular_m  = fb_tabular_mm / 1000

        # B/D correction (if B/D > 12/L, reduce freeboard)
        BD_corr = 0.0
        if (B / D) > (12.0 / LBP * LBP):
            BD_corr = 0.0  # simplified: no correction at this stage

        # Actual freeboard at design draught
        freeboard_actual = D - T

        fb_pass = freeboard_actual >= (fb_tabular_m - 0.05)  # 50mm tolerance

        # ----------------------------------------------------------------
        # IMO IS Code 2008 — Criterion 2.1 (area under GZ curve)
        # Simplified check using Rahola-type approximation
        # For full compliance, NAPA is needed
        # ----------------------------------------------------------------
        # At concept stage: check GM >= minimum threshold
        # IMO requires GM >= 0.15m (grain loading criterion, approx)
        # Product tankers typically need GM >= 0.30m loaded

        gm_min_loaded = 0.30   # [m] — conservative tanker requirement
        gm_pass = GM_loaded >= gm_min_loaded
        gm_lightship_pass = GM_lightship >= 0.15

        # ----------------------------------------------------------------
        # Summary
        # ----------------------------------------------------------------
        return {
            "KB_m":           KB,
            "BMt_m":          BMt,
            "KMt_m":          KMt,
            "GM_lightship_m": GM_lightship,
            "GM_loaded_m":    GM_loaded,
            "KG_lightship_m": KG_lightship,
            "KG_loaded_m":    KG_loaded_corr,
            "freeboard_m":    freeboard_actual,
            "freeboard_req_m": fb_tabular_m,
            "FSC_m":          FSC,
            "gm_pass":        bool(gm_pass),
            "gm_lightship_pass": bool(gm_lightship_pass),
            "freeboard_pass": bool(fb_pass),
            "intact_stable":  bool(gm_pass and fb_pass),
        }


def compute_stability(dp) -> None:
    """Populate dp.stability. Mutates dp in place."""
    ps = PrelimStability()

    h  = dp.hull
    w  = dp.weights
    pw = dp.power

    r = ps.compute(
        LBP=h.LBP, B=h.B, D=h.D, T=h.T,
        Cb=h.Cb, Cwp=h.Cwp, Cm=h.Cm,
        W_lightship_t=w.W_lightship_t if w.W_lightship_t > 0 else h.displacement_t * 0.35,
        VCG_lightship_m=w.VCG_m if w.VCG_m > 0 else h.D * 0.62,
        W_fuel_t=pw.fuel_capacity_t,
        W_cargo_t=w.W_cargo_t,
    )

    dp.stability.KB_m         = r["KB_m"]
    dp.stability.BM_t         = r["BMt_m"]
    dp.stability.GM_t         = r["GM_lightship_m"]
    dp.stability.GM_loaded_m  = r["GM_loaded_m"]
    dp.stability.freeboard_m  = r["freeboard_m"]
    dp.stability.intact_stable = r["intact_stable"]
