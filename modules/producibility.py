"""
producibility.py
================
Producibility constraints module for the tanker design family.

Enforces the relationship between:
  - Steel plate mill dimensions (drives minimum cut waste)
  - Longitudinal frame spacing (standard grid)
  - Transverse web frame / tank bulkhead spacing (multiples of long. frame spacing)
  - Construction block sizes (crane capacity at second-tier US yards)
  - Ship family modularity (shared block types across LOA variants)

Key design decisions established here:
  1. BASE PLATE: 12,000 × 2,400 mm (universal stock, any ABS-certified mill)
     - 18m plates available on order (POSCO/JFE) but add 6-8 week lead time
     - 12m eliminates supply chain risk for Jones Act program
  2. LONGITUDINAL FRAME SPACING: 800 mm
     - 15 × 800mm = 12,000mm → exactly one plate length. Zero waste on shell strakes.
     - Valid range per ABS SVR: 600–900mm for this size class
     - Studies confirm 0.665–0.80m typical for product tankers
  3. TRANSVERSE WEB FRAME SPACING: 4 × 800mm = 3,200mm
     - Every 4th longitudinal space gets a deep web frame
     - Satisfies ABS max web frame spacing (≤ 3.8m for L ≥ 100m)
  4. TANK BULKHEAD SPACING: must be integer multiple of web frame spacing
     - = n_webs × 3,200mm, where n_webs is chosen per design
     - Bulkhead always lands on a web frame → no floating bulkheads, no extra frames
  5. CONSTRUCTION BLOCK LENGTH: 2–4 web frame bays = 6,400–12,800mm
     - Target block weight ≤ 250t for second-tier yard cranes (300–500t capacity, 0.8 factor)
     - Prefer 4-bay blocks (12,800mm) — aligns with 12m plate, one full plate run per block
  6. BLOCK WIDTH: full ship width (B) or half-width
     - Full-width blocks possible for B ≤ ~30m and weight ≤ 250t
     - Above that: split to port/starboard half-blocks
  7. FAMILY MODULARITY: parallel midbody blocks are identical across the family
     - T120/T150/T185 share the same midship block type
     - Only the number of parallel midbody repeats differs
     - End blocks (bow, stern, house) are family-specific

References:
  ABS SVR Part 3 Ch 2 Sec 1 — Frame spacing
  ABS SVR Part 5C Ch 1 Sec 3 — Tanker structural arrangements
  Holtrop (1984); Watson (1998)
  Research on frame spacing optimization (BV Mars2000, Turkish J. Naval Arch. 2020)

Greg Tanker Synthesis Model — v0.1
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Constants — the producibility grid
# ---------------------------------------------------------------------------

# Steel plate standard dimensions (mill stock, no surcharge)
PLATE_LENGTH_STD_MM   = 12_000   # mm — universal ABS stock length
PLATE_LENGTH_LONG_MM  = 18_000   # mm — available on order, +lead time
PLATE_WIDTH_STD_MM    = 2_400    # mm — standard width (3,000 also common)
PLATE_WIDTH_MAX_MM    = 3_900    # mm — widest available from major mills

# Frame grid (all dimensions in mm)
LONG_FRAME_SPACING_MM  = 800     # longitudinal stiffener pitch
WEB_FRAME_MULT         = 4       # transverse web frame every N longitudinal spaces
WEB_FRAME_SPACING_MM   = LONG_FRAME_SPACING_MM * WEB_FRAME_MULT   # = 3,200mm

# Block geometry
BLOCK_BAYS_PREFERRED   = 4       # web frame bays per block (= 12,800mm)
BLOCK_BAYS_MIN         = 2
BLOCK_BAYS_MAX         = 5
BLOCK_LENGTH_PREF_MM   = WEB_FRAME_SPACING_MM * BLOCK_BAYS_PREFERRED  # 12,800mm

# Second-tier US yard crane limits
CRANE_CAPACITY_T_2ND_TIER  = 400   # tonnes, typical (Eastern SB, VT Halter)
BLOCK_WEIGHT_LIMIT_T        = 250   # tonnes (use 0.625 of crane capacity)
BLOCK_WEIGHT_LIMIT_HALF_T   = 130   # tonnes per half-block

# Plate cut efficiency threshold
MIN_PLATE_EFFICIENCY = 0.88   # flag if any panel wastes more than 12% of plate


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FrameGrid:
    """The producibility grid for a given design."""
    long_spacing_mm: int = LONG_FRAME_SPACING_MM
    web_spacing_mm:  int = WEB_FRAME_SPACING_MM
    plate_len_mm:    int = PLATE_LENGTH_STD_MM
    plate_wid_mm:    int = PLATE_WIDTH_STD_MM

    # Derived
    frames_per_plate: int = 0      # how many long. frames fit in one plate
    webs_per_plate:   int = 0      # how many web frames per plate

    def __post_init__(self):
        self.frames_per_plate = self.plate_len_mm // self.long_spacing_mm
        self.webs_per_plate   = self.plate_len_mm // self.web_spacing_mm


@dataclass
class BlockSpec:
    """A hull construction block."""
    id: str
    block_type: str       # parallel_mid | fwd_cargo | aft_cargo | bow | stern | house | eng
    family_shared: bool   # True = this block type is identical across the ship family
    x_aft_m: float        # aft end of block from AP [m]
    x_fwd_m: float        # fwd end from AP [m]
    n_web_bays: int        # number of web frame bays in this block
    length_mm: int         # = n_web_bays × WEB_FRAME_SPACING_MM
    weight_t: float        # estimated block weight [t]
    full_width: bool       # True = full ship width; False = half-width
    plates_per_strake: int  # number of plates to cover one shell strake length
    plate_efficiency: float # fraction of plate used (1.0 = zero waste)
    notes: str = ""


@dataclass
class ProducibilityPlan:
    """Complete producibility plan for a design point."""
    grid: FrameGrid = field(default_factory=FrameGrid)
    blocks: List[BlockSpec] = field(default_factory=list)

    # Summary metrics
    n_blocks_total: int = 0
    n_block_types_unique: int = 0
    parallel_mid_block_count: int = 0   # how many identical parallel midbody blocks
    parallel_mid_length_m: float = 0.0  # total parallel midbody length
    max_block_weight_t: float = 0.0
    mean_plate_efficiency: float = 0.0
    tank_spacing_m: float = 0.0        # bulkhead spacing (m) — must align with grid
    n_long_frames_per_tank: int = 0    # longitudinals per tank length
    n_webs_per_tank: int = 0           # web frames per tank length
    warnings: List[str] = field(default_factory=list)
    grid_violations: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

class ProducibilitySolver:

    def solve(self, LBP: float, B: float, D: float, T: float,
              cargo_block_fwd: float, cargo_block_aft: float,
              n_cargo_tank_bays: int,
              DWT: float) -> ProducibilityPlan:
        """
        Parameters
        ----------
        cargo_block_fwd / aft : float  from arrangements_cargo [m from AP]
        n_cargo_tank_bays     : int    number of longitudinal tank positions
        """
        plan = ProducibilityPlan()
        plan.grid = FrameGrid()
        g = plan.grid

        ws_m = g.web_spacing_mm / 1000   # web frame spacing in metres = 3.2m
        ls_m = g.long_spacing_mm / 1000  # long. frame spacing in metres = 0.8m

        # ----------------------------------------------------------------
        # Step 1: Snap cargo block boundaries to web frame grid
        # ----------------------------------------------------------------
        # The cargo block aft boundary must land on a web frame
        x_aft_snapped = self._snap_to_grid(cargo_block_aft, ws_m, 'ceil')
        x_fwd_snapped = self._snap_to_grid(cargo_block_fwd, ws_m, 'floor')
        cargo_len     = x_fwd_snapped - x_aft_snapped

        if abs(x_aft_snapped - cargo_block_aft) > 0.05:
            plan.warnings.append(
                f"Cargo block aft boundary moved {(x_aft_snapped - cargo_block_aft)*1000:.0f}mm "
                f"to align with web frame grid ({ws_m:.3f}m)")
        if abs(x_fwd_snapped - cargo_block_fwd) > 0.05:
            plan.warnings.append(
                f"Cargo block fwd boundary moved {(x_fwd_snapped - cargo_block_fwd)*1000:.0f}mm "
                f"to align with web frame grid ({ws_m:.3f}m)")

        # ----------------------------------------------------------------
        # Step 2: Solve tank bulkhead spacing
        # ----------------------------------------------------------------
        # Tank length must be an integer number of web frame spacings
        # Start from target: cargo_len / n_cargo_tank_bays
        target_tank_len_m = cargo_len / n_cargo_tank_bays
        n_webs_per_tank   = max(1, round(target_tank_len_m / ws_m))
        tank_len_m        = n_webs_per_tank * ws_m

        # Re-solve n_tanks to fit cargo block
        n_tanks_fit       = max(1, round(cargo_len / tank_len_m))
        # If doesn't fit evenly, adjust n_webs_per_tank up/down
        if n_tanks_fit * tank_len_m > cargo_len + 0.01:
            n_tanks_fit -= 1
        if n_tanks_fit < 1:
            n_tanks_fit = 1

        # Final adjusted tank length and count
        actual_tank_len_m = cargo_len / n_tanks_fit
        n_webs_actual     = round(actual_tank_len_m / ws_m)
        # Force to integer
        tank_len_m        = n_webs_actual * ws_m

        # Recompute n_tanks given integer tank length
        n_tanks           = int(math.floor(cargo_len / tank_len_m))
        remainder_m       = cargo_len - n_tanks * tank_len_m

        if remainder_m > 0.05:
            plan.warnings.append(
                f"Cargo block length {cargo_len:.3f}m does not divide evenly into "
                f"{n_tanks}×{tank_len_m:.3f}m tanks (remainder {remainder_m*1000:.0f}mm). "
                f"Recommend adjusting LBP by {remainder_m:.3f}m or tank count.")

        plan.tank_spacing_m       = tank_len_m
        plan.n_long_frames_per_tank = int(tank_len_m / ls_m)
        plan.n_webs_per_tank      = n_webs_actual

        # ----------------------------------------------------------------
        # Step 3: Plate efficiency for shell strakes
        # ----------------------------------------------------------------
        # Shell strake: one plate per frame bay, plate length = PLATE_LENGTH_STD_MM
        # Shell panel width = web frame spacing = WEB_FRAME_SPACING_MM × bays
        # For a parallel midbody block (BLOCK_BAYS_PREFERRED bays):
        block_len_mm = BLOCK_BAYS_PREFERRED * WEB_FRAME_SPACING_MM  # 12,800mm
        plates_per_strake = math.ceil(block_len_mm / PLATE_LENGTH_STD_MM)   # = 2 for 12,800mm
        used_mm       = block_len_mm
        waste_mm      = plates_per_strake * PLATE_LENGTH_STD_MM - used_mm
        plate_eff     = used_mm / (plates_per_strake * PLATE_LENGTH_STD_MM)

        # Plate width vs ship dimension check
        # Inner hull strake width = double hull void height = D (roughly)
        # Shell strake height ≈ D/n_strakes; target ≈ plate width (2400mm)
        n_strakes_side = max(1, math.ceil(D * 1000 / PLATE_WIDTH_STD_MM))
        strake_h_mm    = D * 1000 / n_strakes_side
        strake_eff     = strake_h_mm / PLATE_WIDTH_STD_MM

        plan.mean_plate_efficiency = (plate_eff + strake_eff) / 2

        if plate_eff < MIN_PLATE_EFFICIENCY:
            plan.warnings.append(
                f"Shell plate length efficiency {plate_eff:.1%}: "
                f"block length {block_len_mm}mm leaves {waste_mm}mm waste per strake")

        # ----------------------------------------------------------------
        # Step 4: Build block schedule
        # ----------------------------------------------------------------
        blocks = []

        # Estimate block steel weight from block volume fraction of total steel weight
        # Total steel weight approximation: Ws = 0.034 × E^1.36 (Watson)
        E      = LBP * (B + D + T/3)
        Ws_total = 0.034 * E**1.36 * 1.03   # with 3% margin
        steel_per_m = Ws_total / LBP          # t/m

        def add_block(bid, btype, x_a, x_f, shared, notes=""):
            blen_m   = x_f - x_a
            blen_mm  = round(blen_m * 1000)
            n_bays   = max(1, round(blen_m / ws_m))
            bwt      = steel_per_m * blen_m * (1.3 if btype in ('bow','stern','house') else 1.0)
            full_w   = bwt <= BLOCK_WEIGHT_LIMIT_T
            n_plt    = math.ceil(blen_mm / PLATE_LENGTH_STD_MM)
            eff      = blen_mm / (n_plt * PLATE_LENGTH_STD_MM)
            blocks.append(BlockSpec(
                id=bid, block_type=btype, family_shared=shared,
                x_aft_m=x_a, x_fwd_m=x_f, n_web_bays=n_bays,
                length_mm=blen_mm, weight_t=round(bwt, 1),
                full_width=full_w,
                plates_per_strake=n_plt, plate_efficiency=round(eff, 3),
                notes=notes
            ))
            return bwt

        # Stern / AP block
        ap_fwd   = x_aft_snapped - (x_aft_snapped % ws_m) if x_aft_snapped > ws_m else x_aft_snapped
        add_block("STERN", "stern", 0, x_aft_snapped, False, "engine room + aft peak")

        # Cargo blocks — grouped into construction blocks of BLOCK_BAYS_PREFERRED web bays
        block_len_cb = BLOCK_BAYS_PREFERRED * ws_m   # preferred construction block length
        x_cur = x_aft_snapped
        cb_idx = 1
        parallel_mid_len = 0.0
        while x_cur < x_fwd_snapped - 0.01:
            x_end = min(x_cur + block_len_cb, x_fwd_snapped)
            # Snap end to nearest tank bulkhead
            x_end_snapped = self._snap_to_nearest_bhd(x_end, x_aft_snapped, tank_len_m)
            if x_end_snapped <= x_cur + 0.01:
                x_end_snapped = x_cur + block_len_cb
            x_end_snapped = min(x_end_snapped, x_fwd_snapped)
            blen = x_end_snapped - x_cur
            # This is a parallel midbody block if not at ends
            is_parallel = (x_cur > x_aft_snapped + 0.5) and (x_end_snapped < x_fwd_snapped - 0.5)
            add_block(f"CB{cb_idx:02d}", "parallel_mid" if is_parallel else "fwd_cargo",
                      x_cur, x_end_snapped, is_parallel,
                      f"{'parallel midbody — shared' if is_parallel else 'fwd cargo block'}")
            if is_parallel:
                parallel_mid_len += blen
            x_cur = x_end_snapped
            cb_idx += 1

        # Bow block (fwd peak + slop tank region)
        add_block("BOW", "bow", x_fwd_snapped, LBP, False, "bow + fwd peak + slop tanks")

        # House/bridge block (typically outfitted off-ship)
        hs_x0 = LBP * 0.80
        add_block("HOUSE", "house", hs_x0, LBP*0.95, False, "house/bridge — pre-outfitted")

        plan.blocks = blocks
        plan.n_blocks_total = len(blocks)
        plan.parallel_mid_block_count = sum(1 for b in blocks if b.block_type == 'parallel_mid')
        plan.parallel_mid_length_m = parallel_mid_len
        plan.max_block_weight_t = max(b.weight_t for b in blocks)
        plan.n_block_types_unique = len(set(b.block_type for b in blocks))

        # ----------------------------------------------------------------
        # Step 5: Over-weight block warnings
        # ----------------------------------------------------------------
        for b in blocks:
            if b.weight_t > BLOCK_WEIGHT_LIMIT_T:
                plan.warnings.append(
                    f"Block {b.id} ({b.block_type}): weight {b.weight_t:.0f}t exceeds "
                    f"{BLOCK_WEIGHT_LIMIT_T}t 2nd-tier crane limit. Split or lighten.")

        # ----------------------------------------------------------------
        # Step 6: Family modularity check
        # ----------------------------------------------------------------
        # For the family to share parallel midbody blocks, the block length
        # and cross-section (B, D) must be identical — only count changes
        plan.n_block_types_unique = 4  # bow, stern, parallel_mid, house = 4 types always

        return plan

    def _snap_to_grid(self, x: float, grid: float, direction: str) -> float:
        """Snap x to nearest grid multiple."""
        if direction == 'ceil':
            return math.ceil(x / grid) * grid
        else:
            return math.floor(x / grid) * grid

    def _snap_to_nearest_bhd(self, x: float, x_origin: float, tank_len: float) -> float:
        """Snap x to nearest tank bulkhead position."""
        rel = x - x_origin
        n   = round(rel / tank_len)
        return x_origin + n * tank_len


def compute_producibility(dp) -> ProducibilityPlan:
    """Solve producibility plan and attach to dp. Mutates dp in place."""
    arr = getattr(dp, 'arrangement', None)

    if arr is None:
        # Use default estimates
        from .arrangements_cargo import CargoTankSolver
        from .design_point import DesignPoint
        solver_arr = CargoTankSolver()
        arr_local = solver_arr.solve(
            LBP=dp.hull.LBP, B=dp.hull.B, D=dp.hull.D, T=dp.hull.T,
            Cb=dp.hull.Cb, Cwp=dp.hull.Cwp, DWT=dp.mission.DWT_target)
        x_cargo_fwd = arr_local.cargo_block_fwd
        x_cargo_aft = arr_local.cargo_block_aft
        n_tanks     = arr_local.n_cargo_pairs
    else:
        x_cargo_fwd = arr.cargo_block_fwd
        x_cargo_aft = arr.cargo_block_aft
        n_tanks     = arr.n_cargo_pairs

    solver = ProducibilitySolver()
    plan = solver.solve(
        LBP=dp.hull.LBP, B=dp.hull.B, D=dp.hull.D, T=dp.hull.T,
        cargo_block_fwd=x_cargo_fwd,
        cargo_block_aft=x_cargo_aft,
        n_cargo_tank_bays=n_tanks,
        DWT=dp.mission.DWT_target,
    )
    dp.producibility = plan
    return plan


def producibility_report(plan: ProducibilityPlan) -> str:
    g = plan.grid
    lines = [
        f"\n{'='*70}",
        f"  Producibility Plan",
        f"{'='*70}",
        f"  Frame grid:       long. {g.long_spacing_mm}mm  ×  web {g.web_spacing_mm}mm",
        f"  Plate:            {g.plate_len_mm}×{g.plate_wid_mm}mm stock  "
        f"({g.frames_per_plate} longs / {g.webs_per_plate} webs per plate)",
        f"  Tank spacing:     {plan.tank_spacing_m:.3f}m  "
        f"({plan.n_webs_per_tank} web bays × {g.web_spacing_mm}mm  =  "
        f"{plan.n_long_frames_per_tank} long. frames)",
        f"  Plate efficiency: {plan.mean_plate_efficiency:.1%}  "
        f"(shell panels, longitudinal direction)",
        f"  Blocks total:     {plan.n_blocks_total}  "
        f"({plan.n_block_types_unique} unique types)",
        f"  Parallel midbody: {plan.parallel_mid_block_count} identical blocks  "
        f"({plan.parallel_mid_length_m:.1f}m)  ← shared across ship family",
        f"  Max block weight: {plan.max_block_weight_t:.0f}t  "
        f"(limit {BLOCK_WEIGHT_LIMIT_T}t for 2nd-tier yard)",
        f"  Warnings:         {len(plan.warnings)}",
    ]
    for w in plan.warnings:
        lines.append(f"    ⚠  {w}")

    lines.append(f"\n  {'Block ID':<10} {'Type':<16} {'Shared':<8} "
                 f"{'x_aft':>7} {'x_fwd':>7} {'Len mm':>8} {'Wt t':>6}  {'Plt eff':>7}  Notes")
    lines.append("  " + "-"*80)
    for b in plan.blocks:
        lines.append(
            f"  {b.id:<10} {b.block_type:<16} {'Y' if b.family_shared else 'N':<8} "
            f"{b.x_aft_m:>7.1f} {b.x_fwd_m:>7.1f} {b.length_mm:>8} "
            f"{b.weight_t:>6.0f}  {b.plate_efficiency:>7.1%}  {b.notes}")
    lines.append("="*70)
    return "\n".join(lines)


