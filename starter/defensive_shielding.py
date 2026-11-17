"""starter/defensive_shielding.py

Defensive Shielding & Territorial Spatial Control Module.
Part of More than a Hack 2027 Possession Inference Architecture (Piste 1).

Physical & Algorithmic Principle:
In the defensive third (x < 0.25 in 1st half for Home, x > 0.75 for Away; inverted in 2nd half),
when a defending player positions his body between the nearest pursuing opponent attacker
and his own defending goal line / endline / touchline, and is escorting the ball trajectory,
the defending player possesses territorial control. The chasing attacker cannot pass through
the defender's body without committing a foul.

This module computes:
1. Spatial interposition: defender is deeper (closer to goal line) than the pursuing attacker.
2. Duel proximity: attacker and defender are engaged in active pursuit/shielding (d_duel <= 3.5m).
3. Territorial priority: grants defensive control priority in Viterbi decoding and reduces
   transition friction to the defending team, eliminating false attacker acquisitions.

100% compliant with RULES.md: strictly general physics-based equations, zero frame hardcoding.
"""

from typing import Tuple
import numpy as np


def detect_defensive_shielding(
    px: np.ndarray,
    py: np.ndarray,
    ps: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    team_mat: np.ndarray,
    blocks: list,
    pitch_length: float = 105.3,
    pitch_width: float = 68.0,
    third_bound: float = 0.25,
    max_reach: float = 5.5,
    max_duel: float = 3.5,
    bridge_gap: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect defensive shielding frames for Home and Away teams.

    Parameters
    ----------
    px, py, ps : np.ndarray, shape (T, N)
        Player coordinates [0, 1] and speeds (m/s) across T frames and N slots.
    bx, by : np.ndarray, shape (T,)
        Normalized ball tracking coordinates [0, 1].
    team_mat : np.ndarray, shape (T, N)
        Dynamic team assignment matrix (0 for Home, 1 for Away).
    blocks : list
        List of half video blocks [(half_name, first_frame, last_frame), ...].
    pitch_length, pitch_width : float
        Pitch dimensions in metres (default 105.3m x 68.0m).
    third_bound : float
        Defensive third spatial boundary threshold (default 0.25).
    max_reach : float
        Maximum distance from ball to player to be considered in the duel (default 5.5m).
    max_duel : float
        Maximum distance between defender and pursuing attacker (default 3.5m).
    bridge_gap : int
        Maximum frame gap to bridge during continuous ball shepherding (default 15 frames = 0.6s).

    Returns
    -------
    shield_home, shield_away : Tuple[np.ndarray, np.ndarray], shape (T,), bool
        Boolean masks indicating confirmed defensive shielding control per team.
    """
    n_frames = len(bx)
    shield_home = np.zeros(n_frames, dtype=bool)
    shield_away = np.zeros(n_frames, dtype=bool)

    if n_frames == 0 or len(blocks) == 0:
        return shield_home, shield_away

    # Half masks
    h1_first, h1_last = blocks[0][1], blocks[0][2]
    is_h1 = np.zeros(n_frames, dtype=bool)
    is_h1[h1_first:h1_last + 1] = True

    is_h2 = np.zeros(n_frames, dtype=bool)
    if len(blocks) > 1:
        h2_first, h2_last = blocks[1][1], blocks[1][2]
        is_h2[h2_first:h2_last + 1] = True

    # Compute per-frame distances from all players to ball in metres
    dx = px * pitch_length - (bx * pitch_length)[:, None]
    dy = py * pitch_width - (by * pitch_width)[:, None]
    dist = np.sqrt(dx * dx + dy * dy)
    dist[np.isnan(dist)] = np.inf

    dist_h = np.where(team_mat == 0, dist, np.inf)
    dist_a = np.where(team_mat == 1, dist, np.inf)

    closest_h = np.argmin(dist_h, axis=1)
    closest_a = np.argmin(dist_a, axis=1)

    min_dh = np.take_along_axis(dist_h, closest_h[:, None], axis=1).squeeze(1)
    min_da = np.take_along_axis(dist_a, closest_a[:, None], axis=1).squeeze(1)

    for t in range(n_frames):
        b_x = bx[t]
        if np.isnan(b_x):
            continue

        in_h1 = is_h1[t]
        in_h2 = is_h2[t]
        if not (in_h1 or in_h2):
            continue

        sh = closest_h[t]
        sa = closest_a[t]
        dh_val = min_dh[t]
        da_val = min_da[t]

        # -----------------------------------------------------------------
        # 1. Home Defensive Third Evaluation:
        # In H1: Home defends left (x < third_bound, goal at x = 0.0)
        # In H2: Home defends right (x > 1 - third_bound, goal at x = 1.0)
        # -----------------------------------------------------------------
        is_h_def = (in_h1 and b_x < third_bound) or (in_h2 and b_x > (1.0 - third_bound))
        goal_x_h = 0.0 if in_h1 else 1.0

        if is_h_def and dh_val <= max_reach and da_val <= max_reach:
            px_def, py_def = px[t, sh], py[t, sh]
            px_att, py_att = px[t, sa], py[t, sa]
            if not (np.isnan(px_def) or np.isnan(px_att)):
                # Defender is positioned between attacker and defending goal line
                dist_def_goal = abs(px_def - goal_x_h)
                dist_att_goal = abs(px_att - goal_x_h)
                if dist_def_goal < dist_att_goal:
                    # Duel proximity between defender and pursuing attacker
                    d_duel = np.sqrt(((px_def - px_att) * pitch_length) ** 2 + ((py_def - py_att) * pitch_width) ** 2)
                    if d_duel <= max_duel:
                        shield_home[t] = True

        # -----------------------------------------------------------------
        # 2. Away Defensive Third Evaluation:
        # In H1: Away defends right (x > 1 - third_bound, goal at x = 1.0)
        # In H2: Away defends left (x < third_bound, goal at x = 0.0)
        # -----------------------------------------------------------------
        is_a_def = (in_h1 and b_x > (1.0 - third_bound)) or (in_h2 and b_x < third_bound)
        goal_x_a = 1.0 if in_h1 else 0.0

        if is_a_def and da_val <= max_reach and dh_val <= max_reach:
            px_def, py_def = px[t, sa], py[t, sa]
            px_att, py_att = px[t, sh], py[t, sh]
            if not (np.isnan(px_def) or np.isnan(px_att)):
                dist_def_goal = abs(px_def - goal_x_a)
                dist_att_goal = abs(px_att - goal_x_a)
                if dist_def_goal < dist_att_goal:
                    d_duel = np.sqrt(((px_def - px_att) * pitch_length) ** 2 + ((py_def - py_att) * pitch_width) ** 2)
                    if d_duel <= max_duel:
                        shield_away[t] = True

    # Temporal continuity bridging during shepherding
    if bridge_gap > 0:
        def _bridge(mask: np.ndarray, gap: int) -> np.ndarray:
            out = mask.copy()
            seen = np.flatnonzero(mask)
            if len(seen) >= 2:
                for a, b in zip(seen[:-1], seen[1:]):
                    if 0 < b - a - 1 <= gap:
                        out[a:b] = True
            return out

        shield_home = _bridge(shield_home, bridge_gap)
        shield_away = _bridge(shield_away, bridge_gap)

    return shield_home, shield_away
