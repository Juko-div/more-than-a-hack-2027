"""src/features/kinematic_features.py

Kinematic ball control, pass flight continuation state machine, and spatial
stoppage feature engineering for tactical tracking data.
"""

from typing import Dict, Optional, Tuple
import numpy as np

# Canonical Pitch Dimensions
PITCH_LENGTH_METERS: float = 105.3
PITCH_WIDTH_METERS: float = 68.0
DEFAULT_FPS: float = 25.0

# Team Slot Slices
HOME_SLOTS: slice = slice(0, 24)
AWAY_SLOTS: slice = slice(24, 46)


def _safe_nanmin(arr: np.ndarray, axis: int = 1) -> np.ndarray:
    """Vectorized row-wise minimum ignoring NaNs without RuntimeWarnings."""
    valid_any = np.any(np.isfinite(arr), axis=axis)
    out = np.full(arr.shape[0], np.nan, dtype=np.float32)
    if np.any(valid_any):
        temp = np.where(np.isfinite(arr), arr, np.inf)
        min_vals = np.min(temp[valid_any], axis=axis)
        out[valid_any] = min_vals.astype(np.float32)
    return out


def compute_relative_velocity(
    px: np.ndarray,
    py: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    dt: float = 0.04,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute relative velocity vector magnitude between ball and all players.

    Parameters
    ----------
    px, py : np.ndarray, shape (T, N)
        Player coordinates in meters.
    bx, by : np.ndarray, shape (T,)
        Ball coordinates in meters.
    dt : float, default=0.04
        Time step between frames in seconds (default 0.04s for 25 fps).

    Returns
    -------
    dist : np.ndarray, shape (T, N), float32
        Euclidean distances from each player to the ball.
    v_rel : np.ndarray, shape (T, N), float32
        Relative velocity magnitude ||v_ball - v_player|| in m/s.
    """
    T, N = px.shape
    if T == 0:
        return np.empty((0, N), dtype=np.float32), np.empty((0, N), dtype=np.float32)

    # Distance: (T, N)
    dx = px - bx[:, None]
    dy = py - by[:, None]
    dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)

    # Ball velocity: (T,)
    v_bx = np.full(T, np.nan, dtype=np.float32)
    v_by = np.full(T, np.nan, dtype=np.float32)
    if T >= 2:
        v_bx[0] = (bx[1] - bx[0]) / dt
        v_by[0] = (by[1] - by[0]) / dt
        v_bx[-1] = (bx[-1] - bx[-2]) / dt
        v_by[-1] = (by[-1] - by[-2]) / dt
    if T >= 3:
        v_bx[1:-1] = (bx[2:] - bx[:-2]) / (2.0 * dt)
        v_by[1:-1] = (by[2:] - by[:-2]) / (2.0 * dt)

    # Player velocities: (T, N)
    v_px = np.full((T, N), np.nan, dtype=np.float32)
    v_py = np.full((T, N), np.nan, dtype=np.float32)
    if T >= 2:
        v_px[0] = (px[1] - px[0]) / dt
        v_py[0] = (py[1] - py[0]) / dt
        v_px[-1] = (px[-1] - px[-2]) / dt
        v_py[-1] = (py[-1] - py[-2]) / dt
    if T >= 3:
        v_px[1:-1] = (px[2:] - px[:-2]) / (2.0 * dt)
        v_py[1:-1] = (py[2:] - py[:-2]) / (2.0 * dt)

    # Relative velocity: (T, N)
    dvx = v_px - v_bx[:, None]
    dvy = v_py - v_by[:, None]
    v_rel = np.sqrt(dvx * dvx + dvy * dvy).astype(np.float32)

    return dist, v_rel


def detect_kinematic_control(
    dist: np.ndarray,
    v_rel: np.ndarray,
    home_slots: slice = HOME_SLOTS,
    away_slots: slice = AWAY_SLOTS,
    dist_thresh: float = 2.0,
    vel_thresh: float = 3.0,
    sustain_frames: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect sustained kinematic ball control per team.

    Parameters
    ----------
    dist : np.ndarray, shape (T, N)
        Distance from each player to ball.
    v_rel : np.ndarray, shape (T, N)
        Relative velocity magnitude between player and ball.
    home_slots, away_slots : slice
        Track slot indices for Home and Away teams.
    dist_thresh : float
        Control radius in meters (default 2.0m).
    vel_thresh : float
        Relative velocity threshold in m/s (default 3.0m/s).
    sustain_frames : int
        Minimum consecutive frames required for control (default 3 frames = 120ms).

    Returns
    -------
    home_control, away_control : Tuple[np.ndarray, np.ndarray], shape (T,), bool
        Boolean masks indicating confirmed sustained possession per team.
    """
    T, N = dist.shape
    if T == 0:
        return np.array([], dtype=bool), np.array([], dtype=bool)

    cand = np.isfinite(dist) & np.isfinite(v_rel) & (dist <= dist_thresh) & (v_rel <= vel_thresh)

    if sustain_frames <= 1:
        home_ctrl = np.any(cand[:, home_slots], axis=1)
        away_ctrl = np.any(cand[:, away_slots], axis=1)
        return home_ctrl, away_ctrl

    sustained = cand.copy()
    for k in range(1, sustain_frames):
        sustained[k:] = sustained[k:] & cand[:-k]
        sustained[:k] = False

    home_ctrl = np.any(sustained[:, home_slots], axis=1)
    away_ctrl = np.any(sustained[:, away_slots], axis=1)
    return home_ctrl, away_ctrl


def track_pass_continuation(
    ball_speed: np.ndarray,
    home_control: np.ndarray,
    away_control: np.ndarray,
    speed_thresh: float = 4.0,
    max_flight_frames: int = 75,
    grace_frames: int = 15,
) -> np.ndarray:
    """Maintain active possession during ball flight phases via state machine.

    Parameters
    ----------
    ball_speed : np.ndarray, shape (T,)
        Instantaneous ball speed in m/s.
    home_control, away_control : np.ndarray, shape (T,), bool
        Confirmed control indicators per team.
    speed_thresh : float
        Minimum ball speed to qualify as pass flight (default 4.0 m/s).
    max_flight_frames : int
        Maximum ball flight duration before timeout (default 75 frames = 3.0s).
    grace_frames : int
        Grace period for reception on ball deceleration (default 15 frames = 0.6s).

    Returns
    -------
    pass_team : np.ndarray, shape (T,), int32
        Active possessing team: 0 (Home), 1 (Away), -1 (None / Contested).
    """
    T = len(ball_speed)
    pass_team = np.full(T, -1, dtype=np.int32)
    active_team = -1
    flight_counter = 0

    for t in range(T):
        c_h = bool(home_control[t])
        c_a = bool(away_control[t])
        bs = ball_speed[t]
        has_bs = bool(np.isfinite(bs))

        if c_h and not c_a:
            active_team = 0
            flight_counter = 0
            pass_team[t] = 0
        elif c_a and not c_h:
            active_team = 1
            flight_counter = 0
            pass_team[t] = 1
        elif active_team != -1:
            if has_bs and bs > speed_thresh:
                flight_counter += 1
                if flight_counter <= max_flight_frames:
                    pass_team[t] = active_team
                else:
                    active_team = -1
            else:
                flight_counter += 1
                if flight_counter <= grace_frames:
                    pass_team[t] = active_team
                else:
                    active_team = -1

    return pass_team


def compute_team_spatial_dispersion(
    px: np.ndarray,
    py: np.ndarray,
    is_home: bool,
    halftime_frame: int = 100000,
) -> np.ndarray:
    """Compute outfield spatial dispersion (equivalent ellipse area in m^2).

    Parameters
    ----------
    px, py : np.ndarray, shape (T, N_team)
        Player coordinates in meters for a single team.
    is_home : bool
        True if Home (Barcelona), False if Away (Athletic Club).
    halftime_frame : int, default=100000
        Frame dividing first and second halves for goalkeeper identification.

    Returns
    -------
    dispersion : np.ndarray, shape (T,), float32
        Outfield tactical dispersion area in m^2.
    """
    T, M = px.shape
    dispersion = np.zeros(T, dtype=np.float32)

    for t in range(T):
        valid = np.isfinite(px[t]) & np.isfinite(py[t])
        x_pts = px[t, valid]
        y_pts = py[t, valid]
        n_pts = len(x_pts)
        if n_pts < 4:
            continue

        # Exclude goalkeeper based on half and team attacking direction
        is_first_half = (t < halftime_frame)
        if (is_home and is_first_half) or (not is_home and not is_first_half):
            gk_idx = int(np.argmin(x_pts))
        else:
            gk_idx = int(np.argmax(x_pts))

        out_x = np.delete(x_pts, gk_idx)
        out_y = np.delete(y_pts, gk_idx)

        sx = float(np.std(out_x))
        sy = float(np.std(out_y))
        dispersion[t] = float(np.pi * 4.0 * sx * sy)

    return dispersion


def compute_ball_rest_indicator(
    ball_speed: np.ndarray,
    thresh: float = 0.2,
    min_frames: int = 25,
) -> np.ndarray:
    """Detect sustained ball rest intervals (v_ball < 0.2 m/s for >= 25 frames).

    Parameters
    ----------
    ball_speed : np.ndarray, shape (T,)
        Ball speed in m/s.
    thresh : float, default=0.2
        Maximum speed for rest (default 0.2 m/s).
    min_frames : int, default=25
        Minimum consecutive frames required (default 25 frames = 1.0s).

    Returns
    -------
    ball_rest_flag : np.ndarray, shape (T,), bool
    """
    T = len(ball_speed)
    if T == 0:
        return np.array([], dtype=bool)

    is_rest = np.zeros(T, dtype=bool)
    is_slow = np.isfinite(ball_speed) & (ball_speed < thresh)

    cur_run = 0
    start_idx = 0
    for t in range(T):
        if is_slow[t]:
            if cur_run == 0:
                start_idx = t
            cur_run += 1
        else:
            if cur_run >= min_frames:
                is_rest[start_idx:t] = True
            cur_run = 0
    if cur_run >= min_frames:
        is_rest[start_idx:T] = True

    return is_rest


def compute_pitch_boundary_buffer(
    bx: np.ndarray,
    by: np.ndarray,
    pitch_length: float = PITCH_LENGTH_METERS,
    pitch_width: float = PITCH_WIDTH_METERS,
    buffer: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ball distance outside pitch boundary and boundary excursion flag.

    Parameters
    ----------
    bx, by : np.ndarray, shape (T,)
        Ball coordinates in meters.
    pitch_length, pitch_width : float
        Pitch dimensions in meters.
    buffer : float, default=1.0
        Buffer threshold in meters (default 1.0m).

    Returns
    -------
    out_of_bounds : np.ndarray, shape (T,), bool
    d_out : np.ndarray, shape (T,), float32
        Distance in meters outside the pitch boundary (0.0 if inside).
    """
    T = len(bx)
    if T == 0:
        return np.array([], dtype=bool), np.array([], dtype=np.float32)

    valid = np.isfinite(bx) & np.isfinite(by)
    dx_out = np.zeros(T, dtype=np.float32)
    dy_out = np.zeros(T, dtype=np.float32)
    d_out = np.full(T, np.nan, dtype=np.float32)
    out_of_bounds = np.zeros(T, dtype=bool)

    dx_out[valid] = np.maximum(0.0, np.maximum(-bx[valid], bx[valid] - pitch_length))
    dy_out[valid] = np.maximum(0.0, np.maximum(-by[valid], by[valid] - pitch_width))
    d_out[valid] = np.sqrt(dx_out[valid]**2 + dy_out[valid]**2).astype(np.float32)
    out_of_bounds[valid] = d_out[valid] > buffer

    return out_of_bounds, d_out


def extract_kinematic_features(
    px: np.ndarray,
    py: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    ball_speed: np.ndarray,
    is_interpolated: Optional[np.ndarray] = None,
    pitch_length: float = PITCH_LENGTH_METERS,
    pitch_width: float = PITCH_WIDTH_METERS,
    fps: float = DEFAULT_FPS,
) -> Dict[str, np.ndarray]:
    """Compute complete kinematic control, pass continuation, and spatial stoppage features.

    Parameters
    ----------
    px : np.ndarray, shape (T, 46), float32
        Player X coordinates in meters (or normalized [0, 1]).
    py : np.ndarray, shape (T, 46), float32
        Player Y coordinates in meters (or normalized [0, 1]).
    bx : np.ndarray, shape (T,), float32
        Ball X coordinates in meters (or normalized [0, 1]).
    by : np.ndarray, shape (T,), float32
        Ball Y coordinates in meters (or normalized [0, 1]).
    ball_speed : np.ndarray, shape (T,), float32
        Instantaneous ball speed in m/s.
    is_interpolated : Optional[np.ndarray], shape (T,), bool
        Boolean mask of interpolated ball frames.
    pitch_length : float
        Pitch length in meters (default 105.3).
    pitch_width : float
        Pitch width in meters (default 68.0).
    fps : float
        Sampling rate in frames per second (default 25.0).

    Returns
    -------
    dict[str, np.ndarray] containing:
        - 'd_home_min'       : (T,) float32, minimum distance from Home players to ball (m)
        - 'd_away_min'       : (T,) float32, minimum distance from Away players to ball (m)
        - 'rel_vel_home'     : (T,) float32, minimum relative velocity to ball for Home players (m/s)
        - 'rel_vel_away'     : (T,) float32, minimum relative velocity to ball for Away players (m/s)
        - 'home_control'     : (T,) bool, sustained kinematic control by Home
        - 'away_control'     : (T,) bool, sustained kinematic control by Away
        - 'pass_team'        : (T,) int32, active possessing team during pass flight (-1, 0, 1)
        - 'team_spread_home' : (T,) float32, outfield spatial dispersion area for Home (m^2)
        - 'team_spread_away' : (T,) float32, outfield spatial dispersion area for Away (m^2)
        - 'ball_speed'       : (T,) float32, ball speed in m/s
        - 'ball_rest_flag'   : (T,) bool, stationary ball indicator (>= 25 frames < 0.2 m/s)
        - 'boundary_dist'    : (T,) float32, distance outside pitch boundary (m, 0 if inside)
        - 'out_of_bounds'    : (T,) bool, ball outside 1.0m pitch boundary buffer
        - 'is_interpolated'  : (T,) bool, interpolated ball mask
    """
    T = len(bx)
    dt = 1.0 / float(fps)

    # Convert coordinates to metric if normalized
    px_m = px.copy().astype(np.float32)
    py_m = py.copy().astype(np.float32)
    bx_m = bx.copy().astype(np.float32)
    by_m = by.copy().astype(np.float32)

    # Auto-detect normalized [0, 1] inputs: if max valid coordinate <= 1.5, scale to meters
    finite_x = px_m[np.isfinite(px_m)]
    if len(finite_x) > 0 and np.max(finite_x) <= 1.5:
        px_m *= pitch_length
        py_m *= pitch_width
        bx_m *= pitch_length
        by_m *= pitch_width

    # Relative velocity & distances
    dist, v_rel = compute_relative_velocity(px_m, py_m, bx_m, by_m, dt=dt)

    d_home_min = _safe_nanmin(dist[:, HOME_SLOTS], axis=1)
    d_away_min = _safe_nanmin(dist[:, AWAY_SLOTS], axis=1)

    rel_vel_home = _safe_nanmin(v_rel[:, HOME_SLOTS], axis=1)
    rel_vel_away = _safe_nanmin(v_rel[:, AWAY_SLOTS], axis=1)

    # Sustained control
    home_ctrl, away_ctrl = detect_kinematic_control(
        dist, v_rel,
        home_slots=HOME_SLOTS,
        away_slots=AWAY_SLOTS,
        dist_thresh=2.0,
        vel_thresh=3.0,
        sustain_frames=3,
    )

    # Pass flight continuation
    pass_team = track_pass_continuation(
        ball_speed=ball_speed,
        home_control=home_ctrl,
        away_control=away_ctrl,
        speed_thresh=4.0,
        max_flight_frames=75,
        grace_frames=15,
    )

    # Spatial dispersion
    spread_home = compute_team_spatial_dispersion(
        px_m[:, HOME_SLOTS], py_m[:, HOME_SLOTS], is_home=True
    )
    spread_away = compute_team_spatial_dispersion(
        px_m[:, AWAY_SLOTS], py_m[:, AWAY_SLOTS], is_home=False
    )

    # Ball rest
    ball_rest_flag = compute_ball_rest_indicator(ball_speed, thresh=0.2, min_frames=25)

    # Pitch boundary buffer
    out_of_bounds, boundary_dist = compute_pitch_boundary_buffer(
        bx_m, by_m, pitch_length=pitch_length, pitch_width=pitch_width, buffer=1.0
    )

    if is_interpolated is None:
        is_interpolated = np.zeros(T, dtype=bool)

    return {
        "d_home_min": d_home_min,
        "d_away_min": d_away_min,
        "rel_vel_home": rel_vel_home,
        "rel_vel_away": rel_vel_away,
        "home_control": home_ctrl,
        "away_control": away_ctrl,
        "pass_team": pass_team,
        "team_spread_home": spread_home,
        "team_spread_away": spread_away,
        "ball_speed": ball_speed.astype(np.float32),
        "ball_rest_flag": ball_rest_flag,
        "boundary_dist": boundary_dist,
        "out_of_bounds": out_of_bounds,
        "is_interpolated": is_interpolated.astype(bool),
    }
