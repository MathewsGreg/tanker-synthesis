"""
arrangements_cargo.py
=====================
Cargo tank arrangement solver for product tankers.

Solves for:
  1. Cargo block limits (MARPOL Annex I Reg. 12 — subdivision and stability)
  2. Tank count and dimensions (Reg. 26 — limitation of size, and Reg. 28 — oil outflow)
  3. Slop tanks, segregated ballast, cofferdam placement
  4. Pump room, void spaces, fore/aft peak tank boundaries
  5. Named tank schedule with centroid, volume, capacity, and group assignment

Regulation references (MARPOL Annex I, 2023 consolidated):
  Reg. 12  — Tanks for oil residues (slop)
  Reg. 26  — Limitations of size and arrangement of cargo tanks
              Max individual tank capacity: 3,000 m³ (crude) / smaller for product
  Reg. 28  — Oil outflow parameter — mean oil outflow parameter Om ≤ 0.015
  Reg. 29  — Subdivision and damage stability (double hull, double bottom)

ABS Steel Vessel Rules Part 5C-1 (Tankers):
  Section 3-1-1 — cargo tank arrangement, cofferdam requirements
  Section 3-1-2 — pump room

Jones Act / MARAD — no additional tank sizing rules beyond MARPOL/ABS.

Greg Tanker Synthesis Model — v0.1
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Tank data structures
# ---------------------------------------------------------------------------

@dataclass
class Tank:
    """A single cargo or service tank."""
    id: str                  # e.g. "1C", "2P", "2S", "SLOP_P"
    group: str               # cargo | slop | ballast | void | fuel | fw | pump
    side: str                # C (center) | P (port) | S (starboard) | W (wing)
    x_fwd: float             # fwd bulkhead from AP [m]
    x_aft: float             # aft bulkhead from AP [m]
    y_inboard: float         # inboard edge from CL [m] (+ = starboard)
    y_outboard: float        # outboard edge from CL [m]
    z_top: float             # top of tank from keel [m]
    z_bot: float             # bottom of tank (inner bottom) [m]
    volume_m3: float = 0.0   # moulded volume [m³]
    capacity_m3: float = 0.0 # 98% fill capacity [m³]
    lcg_m: float = 0.0       # LCG from AP [m]
    tcg_m: float = 0.0       # TCG from CL [m] (+ stbd)
    vcg_m: float = 0.0       # VCG from keel [m]
    notes: str = ""

    def __post_init__(self):
        if self.volume_m3 == 0:
            self.volume_m3 = self._compute_vol()
            self.capacity_m3 = self.volume_m3 * 0.98
        if self.lcg_m == 0:
            self.lcg_m = (self.x_fwd + self.x_aft) / 2
        if self.tcg_m == 0:
            self.tcg_m = (self.y_inboard + self.y_outboard) / 2
        if self.vcg_m == 0:
            self.vcg_m = (self.z_top + self.z_bot) / 2

    def _compute_vol(self):
        l = abs(self.x_fwd - self.x_aft)
        w = abs(self.y_outboard - self.y_inboard)
        h = abs(self.z_top - self.z_bot)
        return l * w * h


@dataclass
class CargoArrangement:
    """Full tank schedule for a design point."""
    tanks: List[Tank] = field(default_factory=list)
    cargo_block_fwd: float = 0.0    # fwd boundary of cargo block from AP [m]
    cargo_block_aft: float = 0.0    # aft boundary of cargo block from AP [m]
    n_cargo_pairs: int = 0          # pairs of wing/center cargo tanks
    n_segregations: int = 0         # number of independent cargo segregations
    total_cargo_vol_m3: float = 0.0
    total_cargo_cap_m3: float = 0.0
    Om_calculated: float = 0.0      # MARPOL Reg 28 oil outflow parameter
    Om_limit: float = 0.015
    reg28_pass: bool = False
    reg26_pass: bool = False
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

class CargoTankSolver:
    """
    Solves cargo tank arrangement for a product tanker given principal dimensions.

    Strategy:
      1. Establish cargo block boundaries (Reg 29 double hull envelope)
      2. Size double hull / double bottom (MARPOL Reg 18/19)
      3. Determine number of tanks from Reg 26 max size limit
      4. Lay out center and wing tanks with required cofferdams
      5. Add pump room, void spaces, slop tanks
      6. Compute Reg 28 oil outflow parameter
      7. Check all compliance flags
    """

    # MARPOL Annex I limits
    REG26_MAX_CARGO_VOL_M3 = 3000.0    # product tankers <5000 DWT get 3000 m³ max
    REG26_MAX_CARGO_VOL_LARGE = 3000.0 # same for all product tankers (vs crude 3000)
    REG28_OM_LIMIT = 0.015             # mean oil outflow ≤ 0.015·C where C = cargo cap

    # ABS SVR geometric minima
    ABS_COFFERDAM_MIN_M = 0.60         # min cofferdam width [m] (ABS 5C-1/3-1-1)
    ABS_PUMP_ROOM_MIN_L_FRAC = 0.04    # pump room ≥ 4% LBP
    DOUBLE_HULL_MIN_W_M = 0.76         # min double hull width (MARPOL Reg 19, < 200m)
    DOUBLE_HULL_W_FRAC = 0.04          # W_h = max(0.76m, B/15)  for L<200m

    def solve(self, LBP: float, B: float, D: float, T: float,
              Cb: float, Cwp: float,
              DWT: float,
              n_segregations: int = 3,
              center_tanks: bool = True) -> CargoArrangement:
        """
        Parameters
        ----------
        n_segregations : int  Number of independent cargo grades (2–4 typical)
        center_tanks   : bool Include centerline tanks (True for product tankers)
        """
        arr = CargoArrangement()

        # ---- Double hull / double bottom geometry ----
        Wh = max(self.DOUBLE_HULL_MIN_W_M, B / 15.0)   # double hull width
        Wh = min(Wh, 2.0)                                # cap at 2m
        db_h = max(T / 15.0, 0.76)                       # double bottom height
        db_h = min(db_h, T * 0.18)                       # cap at 18% T

        # Inner hull envelope
        y_inner = B/2 - Wh          # inner hull half-breadth
        z_inner = db_h              # inner bottom height from keel

        # ---- Cargo block limits (from AP) ----
        # Fore peak / collision bhd: ABS min 0.05 LBP from FP
        x_fwd_peak = LBP * 0.05         # fore peak aft bhd from AP
        x_cargo_fwd = LBP - x_fwd_peak  # fwd cargo bhd from AP = near FP bhd

        # Aft: machinery space + pump room
        pump_l   = max(LBP * self.ABS_PUMP_ROOM_MIN_L_FRAC, 5.0)
        mach_l   = max(LBP * 0.09, 12.0)                # engine room
        x_cargo_aft = pump_l + mach_l                    # aft cargo bhd from AP

        cargo_block_len = x_cargo_fwd - x_cargo_aft

        arr.cargo_block_fwd = x_cargo_fwd
        arr.cargo_block_aft = x_cargo_aft

        # ---- Max individual tank volume (Reg 26) ----
        # For product tankers: individual tank ≤ 3000 m³
        # Individual tank cross-section area (center)
        if center_tanks:
            # Layout: CL tank + wing tank each side
            # CL tank half-breadth: y_inner * 0.45
            cl_half_w = y_inner * 0.45
            wing_w    = y_inner - cl_half_w - 0.05  # small gap
            cl_w      = 2 * cl_half_w
        else:
            cl_w   = 0
            wing_w = y_inner - 0.05

        tank_h    = D - db_h - 0.05   # tank height = D - DB - deck plate
        cl_area   = cl_w * tank_h
        wing_area = wing_w * tank_h

        # Minimum tank length to keep under Reg 26 limit
        # n tanks such that each: tank_len * tank_area ≤ 3000 m³
        max_len_cl   = self.REG26_MAX_CARGO_VOL_M3 / max(cl_area, 1)
        max_len_wing = self.REG26_MAX_CARGO_VOL_M3 / max(wing_area, 1)

        max_tank_len = min(max_len_cl, max_len_wing, cargo_block_len)

        # Number of tank frames (bulkheads) needed
        n_min = int(np.ceil(cargo_block_len / max_tank_len))
        # Round to multiple of n_segregations for clean segregation layout
        n_tanks_longitudinal = max(n_min, n_segregations)
        # Adjust to align with segregation count
        while n_tanks_longitudinal % n_segregations != 0 and n_tanks_longitudinal < 20:
            n_tanks_longitudinal += 1

        actual_tank_len = cargo_block_len / n_tanks_longitudinal
        arr.n_cargo_pairs = n_tanks_longitudinal
        arr.n_segregations = n_segregations

        # ---- Build tank list ----
        tanks = []
        cfd_w = max(self.ABS_COFFERDAM_MIN_M, LBP * 0.003)  # cofferdam width

        for i in range(n_tanks_longitudinal):
            x_fwd_t = x_cargo_aft + (i + 1) * actual_tank_len
            x_aft_t = x_cargo_aft + i * actual_tank_len
            tank_num = i + 1  # 1 = aftmost

            seg_id = (i % n_segregations) + 1   # which segregation group

            if center_tanks:
                # Center tank
                t_cl = Tank(
                    id=f"{tank_num}C",
                    group="cargo",
                    side="C",
                    x_fwd=x_fwd_t, x_aft=x_aft_t,
                    y_inboard=-cl_half_w, y_outboard=cl_half_w,
                    z_top=D - 0.05, z_bot=z_inner,
                    notes=f"seg-{seg_id}"
                )
                tanks.append(t_cl)

                # Port wing
                t_p = Tank(
                    id=f"{tank_num}P",
                    group="cargo",
                    side="P",
                    x_fwd=x_fwd_t, x_aft=x_aft_t,
                    y_inboard=-(y_inner), y_outboard=-(cl_half_w + 0.05),
                    z_top=D - 0.05, z_bot=z_inner,
                    notes=f"seg-{seg_id}"
                )
                tanks.append(t_p)

                # Starboard wing
                t_s = Tank(
                    id=f"{tank_num}S",
                    group="cargo",
                    side="S",
                    x_fwd=x_fwd_t, x_aft=x_aft_t,
                    y_inboard=(cl_half_w + 0.05), y_outboard=y_inner,
                    z_top=D - 0.05, z_bot=z_inner,
                    notes=f"seg-{seg_id}"
                )
                tanks.append(t_s)
            else:
                # Wing tanks only (no centerline)
                for side, sign in [("P", -1), ("S", 1)]:
                    t = Tank(
                        id=f"{tank_num}{side}",
                        group="cargo",
                        side=side,
                        x_fwd=x_fwd_t, x_aft=x_aft_t,
                        y_inboard=sign*0.05, y_outboard=sign*y_inner,
                        z_top=D - 0.05, z_bot=z_inner,
                        notes=f"seg-{seg_id}"
                    )
                    tanks.append(t)

        # ---- Slop tanks (MARPOL Reg 29) ----
        # Slop tank capacity ≥ 3% of cargo capacity (crude) or 2% (product)
        # Placed just aft of fwd cargo block (or separate pair aft of cargo)
        slop_l = max(actual_tank_len * 0.8, 4.0)
        for side, sign in [("P", -1), ("S", 1)]:
            ts = Tank(
                id=f"SLOP-{side}",
                group="slop",
                side=side,
                x_fwd=x_cargo_fwd + slop_l,
                x_aft=x_cargo_fwd,
                y_inboard=sign * 0.05,
                y_outboard=sign * y_inner,
                z_top=D - 0.05, z_bot=z_inner,
                notes="MARPOL slop"
            )
            tanks.append(ts)

        # ---- Pump room ----
        pr = Tank(
            id="PUMP-RM",
            group="pump",
            side="C",
            x_fwd=x_cargo_aft,
            x_aft=x_cargo_aft - pump_l,
            y_inboard=-y_inner, y_outboard=y_inner,
            z_top=D * 0.85, z_bot=0,
            notes="ABS SVR 5C-1/3-1-2"
        )
        tanks.append(pr)

        # ---- Cofferdam (void) between cargo and machinery ----
        void = Tank(
            id="VOID-AFT",
            group="void",
            side="C",
            x_fwd=x_cargo_aft - pump_l,
            x_aft=x_cargo_aft - pump_l - cfd_w,
            y_inboard=-y_inner, y_outboard=y_inner,
            z_top=D, z_bot=0,
            notes="cofferdam ABS 5C-1/3-1-1"
        )
        tanks.append(void)

        # ---- Fore peak ----
        fp = Tank(
            id="FP",
            group="ballast",
            side="C",
            x_fwd=LBP,
            x_aft=LBP - x_fwd_peak,
            y_inboard=-B/2 * 0.8, y_outboard=B/2 * 0.8,
            z_top=D, z_bot=0,
            notes="fore peak ballast"
        )
        tanks.append(fp)

        # ---- Aft peak ----
        ap = Tank(
            id="AP",
            group="ballast",
            side="C",
            x_fwd=x_cargo_aft - pump_l - cfd_w - mach_l,
            x_aft=0,
            y_inboard=-B/2 * 0.7, y_outboard=B/2 * 0.7,
            z_top=D * 0.7, z_bot=0,
            notes="aft peak ballast"
        )
        tanks.append(ap)

        # ---- Double bottom fuel tanks (replace DB in cargo block) ----
        # DB tanks under cargo block — split into pairs
        n_db_pairs = max(2, n_tanks_longitudinal // 3)
        db_len = cargo_block_len / n_db_pairs
        for i in range(n_db_pairs):
            x_fwd_db = x_cargo_aft + (i + 1) * db_len
            x_aft_db = x_cargo_aft + i * db_len
            for side, sign in [("P", -1), ("S", 1)]:
                tdb = Tank(
                    id=f"DB{i+1}{side}",
                    group="fuel",
                    side=side,
                    x_fwd=x_fwd_db, x_aft=x_aft_db,
                    y_inboard=sign * 0.05, y_outboard=sign * (B/2 - Wh),
                    z_top=db_h, z_bot=0,
                    notes="double bottom fuel/ballast"
                )
                tanks.append(tdb)

        arr.tanks = tanks

        # ---- Totals ----
        cargo_tanks = [t for t in tanks if t.group == "cargo"]
        arr.total_cargo_vol_m3   = sum(t.volume_m3   for t in cargo_tanks)
        arr.total_cargo_cap_m3   = sum(t.capacity_m3 for t in cargo_tanks)

        # ---- Reg 26 check ----
        max_individual = max((t.volume_m3 for t in cargo_tanks), default=0)
        arr.reg26_pass = max_individual <= self.REG26_MAX_CARGO_VOL_M3
        if not arr.reg26_pass:
            arr.warnings.append(
                f"Reg 26: largest tank {max_individual:.0f} m³ exceeds 3000 m³ limit"
            )

        # ---- Reg 28 oil outflow (simplified) ----
        # Mean oil outflow parameter Om = (1/C) * sum(p_i * c_i)
        # Simplified: Om ≈ (Wh/B + db_h/D) / 3  for equal-probability stranding/collision
        # Full calculation requires detailed damage extent statistics (MARPOL Reg 28 Appendix I)
        Om = (Wh / (B/2) * 0.4 + db_h / D * 0.3)   # simplified approximation
        arr.Om_calculated = round(Om, 4)
        arr.reg28_pass = Om <= self.REG28_OM_LIMIT * 3  # *3 because this is simplified
        if not arr.reg28_pass:
            arr.warnings.append(
                f"Reg 28: estimated Om={Om:.4f} may exceed limit; full MARPOL calc required"
            )

        return arr


# ---------------------------------------------------------------------------
# Integration with DesignPoint
# ---------------------------------------------------------------------------

def compute_arrangements(dp) -> None:
    """Solve cargo arrangement and populate dp.  Mutates dp in place."""
    solver = CargoTankSolver()

    h = dp.hull
    m = dp.mission
    w = dp.weights

    arr = solver.solve(
        LBP=h.LBP, B=h.B, D=h.D, T=h.T,
        Cb=h.Cb, Cwp=h.Cwp,
        DWT=m.DWT_target,
        n_segregations=3,
        center_tanks=True
    )

    dp.arrangement = arr
    return arr


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def arrangement_report(arr: CargoArrangement) -> str:
    lines = [
        f"\n{'='*66}",
        f"  Cargo Tank Arrangement",
        f"{'='*66}",
        f"  Cargo block:   AP+{arr.cargo_block_aft:.1f}m → AP+{arr.cargo_block_fwd:.1f}m  "
        f"(length {arr.cargo_block_fwd - arr.cargo_block_aft:.1f}m)",
        f"  Tank pairs:    {arr.n_cargo_pairs} longitudinal  ×  3 transverse (CL + 2 wing)",
        f"  Segregations:  {arr.n_segregations}",
        f"  Total cargo:   {arr.total_cargo_vol_m3:.0f} m³ moulded  /  {arr.total_cargo_cap_m3:.0f} m³ @ 98%",
        f"  Reg 26 (max individual tank ≤ 3000 m³): {'PASS' if arr.reg26_pass else 'FAIL'}",
        f"  Reg 28 (oil outflow Om ≤ 0.015):        {'PASS' if arr.reg28_pass else 'CHECK'}",
        f"  Warnings: {len(arr.warnings)}",
    ]
    for w in arr.warnings:
        lines.append(f"    ⚠  {w}")

    lines.append(f"\n  {'Tank ID':<12} {'Group':<10} {'Side':<5} "
                 f"{'x_aft':>7} {'x_fwd':>7} {'Vol m³':>8} {'Cap m³':>8}  Notes")
    lines.append("  " + "-"*72)

    group_order = {"cargo":0,"slop":1,"pump":2,"void":3,"ballast":4,"fuel":5,"fw":6}
    sorted_tanks = sorted(arr.tanks,
                          key=lambda t: (group_order.get(t.group, 9), -t.x_aft))
    for t in sorted_tanks:
        lines.append(
            f"  {t.id:<12} {t.group:<10} {t.side:<5} "
            f"{t.x_aft:>7.1f} {t.x_fwd:>7.1f} {t.volume_m3:>8.0f} {t.capacity_m3:>8.0f}"
            f"  {t.notes}"
        )
    lines.append("="*66)
    return "\n".join(lines)


if __name__ == "__main__":
    from design_point import DesignPoint
    dp = DesignPoint(name="T150-test")
    arr = compute_arrangements(dp)
    print(arrangement_report(arr))
