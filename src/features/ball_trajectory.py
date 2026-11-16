"""src/features/ball_trajectory.py

Finite-difference ball velocity, momentum-preserving gap interpolation, and
kinematic ball indicators for tactical tracking data.

Key capabilities:
1. Momentum-preserving ball gap interpolation (gaps <= 40 frames, 1.6s).
2. Physical velocity vector consistency checks (chord speed limit, directional alignment, speed continuity).
3. Finite-difference ball velocity (vx, vy, speed) with adaptive window and boundary handling.
4. Ball-at-rest indicator (speed < 0.2 m/s for >= 25 frames).
5. Pitch boundary excursion indicator with configurable buffer margin.
"""

from typing import Dict, Optional, Tuple
import numpy as np

PITCH_LENGTH_METERS: float = 105.3
PITCH_WIDTH_METERS: float = 68.0
DEFAULT_FPS: float = 25.0


def interpolate_ball_trajectory(
    bx: np.ndarray,
    by: np.ndarray,
    max_gap: int = 40,
    fps: float = DEFAULT_FPS,
    pitch_length: float = PITCH_LENGTH_METERS,
    pitch_width: float = PITCH_WIDTH_METERS,
    check_consistency: bool = True,
    max_speed: float = 35.0,
    min_cos_sim: float = 0.5,
    max_accel_slack: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Momentum-preserving interpolation of missing ball positions across short gaps.

    Parameters
    ----------
    bx, by : np.ndarray
        Normalized ball tracking coordinates in [0, 1] (shape: (n_frames,)), with np.nan for missing frames.
    max_gap : int, default=40
        Maximum consecutive missing frames to consider for interpolation (40 frames = 1.6s).
    fps : float, default=25.0
        Sampling frequency in Hz.
    pitch_length, pitch_width : float
        Pitch dimensions in meters.
    check_consistency : bool, default=True
        Whether to enforce velocity vector direction and speed consistency before interpolating.
    max_speed : float, default=35.0
        Maximum physically plausible ball speed in m/s (35 m/s = 126 km/h).
    min_cos_sim : float, default=0.5
        Minimum cosine similarity between velocity vectors and chord direction.
    max_accel_slack : float, default=5.0
        Speed tolerance in m/s for chord velocity relative to boundary speeds.

    Returns
    -------
    bx_interp : np.ndarray
        Interpolated normalized x coordinates (shape: (n_frames,)).
    by_interp : np.ndarray
        Interpolated normalized y coordinates (shape: (n_frames,)).
    is_interpolated : np.ndarray
        Boolean mask of shape (n_frames,), True where ball was interpolated.
    """
    n_frames = len(bx)
    if n_frames == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=bool)

    out_x = bx.copy().astype(np.float32)
    out_y = by.copy().astype(np.float32)
    is_interpolated = np.zeros(n_frames, dtype=bool)

    seen = np.flatnonzero(np.isfinite(bx) & np.isfinite(by))
    if len(seen) < 2:
        return out_x, out_y, is_interpolated

    for a, b in zip(seen[:-1], seen[1:]):
        k = b - a - 1
        if 0 < k <= max_gap:
            if not check_consistency:
                w = np.arange(1, k + 1, dtype=np.float32) / float(k + 1)
                out_x[a + 1 : b] = bx[a] + (bx[b] - bx[a]) * w
                out_y[a + 1 : b] = by[a] + (by[b] - by[a]) * w
                is_interpolated[a + 1 : b] = True
                continue

            dt = float(k + 1) / float(fps)
            dx_chord = float((bx[b] - bx[a]) * pitch_length)
            dy_chord = float((by[b] - by[a]) * pitch_width)
            dist = float(np.sqrt(dx_chord**2 + dy_chord**2))
            v_chord = dist / dt

            # 1. Reject impossible supersonic/teleportation speeds
            if v_chord > max_speed:
                continue

            # 2. Stationary ball pass-through
            if v_chord < 0.5 and dist < 0.5:
                w = np.arange(1, k + 1, dtype=np.float32) / float(k + 1)
                out_x[a + 1 : b] = bx[a] + (bx[b] - bx[a]) * w
                out_y[a + 1 : b] = by[a] + (by[b] - by[a]) * w
                is_interpolated[a + 1 : b] = True
                continue

            # 3. Estimate boundary velocities (up to 4 valid frames)
            h_pre = min(4, a)
            pre_idx = max(0, a - h_pre)
            dt_pre = float(a - pre_idx) / float(fps) if a > pre_idx else 1.0 / float(fps)
            vx_pre = float((bx[a] - bx[pre_idx]) * pitch_length / dt_pre)
            vy_pre = float((by[a] - by[pre_idx]) * pitch_width / dt_pre)
            v_pre = float(np.sqrt(vx_pre**2 + vy_pre**2))

            h_post = min(4, n_frames - 1 - b)
            post_idx = min(n_frames - 1, b + h_post)
            dt_post = float(post_idx - b) / float(fps) if post_idx > b else 1.0 / float(fps)
            vx_post = float((bx[post_idx] - bx[b]) * pitch_length / dt_post)
            vy_post = float((by[post_idx] - by[b]) * pitch_width / dt_post)
            v_post = float(np.sqrt(vx_post**2 + vy_post**2))

            u_chord = np.array([dx_chord, dy_chord], dtype=np.float32) / (dist + 1e-6)
            u_pre = np.array([vx_pre, vy_pre], dtype=np.float32) / (v_pre + 1e-6)
            u_post = np.array([vx_post, vy_post], dtype=np.float32) / (v_post + 1e-6)

            cos_pre_chord = float(np.dot(u_pre, u_chord)) if v_pre > 1.0 else 1.0
            cos_post_chord = float(np.dot(u_post, u_chord)) if v_post > 1.0 else 1.0
            cos_pre_post = float(np.dot(u_pre, u_post)) if (v_pre > 1.0 and v_post > 1.0) else 1.0

            max_boundary = max(v_pre, v_post)
            speed_bounded = v_chord <= max(max_boundary * 1.5, max_boundary + max_accel_slack, 2.0)
            dir_aligned = (cos_pre_chord >= min_cos_sim) and (cos_post_chord >= min_cos_sim) and (cos_pre_post >= 0.0)

            if speed_bounded and dir_aligned:
                w = np.arange(1, k + 1, dtype=np.float32) / float(k + 1)
                out_x[a + 1 : b] = bx[a] + (bx[b] - bx[a]) * w
                out_y[a + 1 : b] = by[a] + (by[b] - by[a]) * w
                is_interpolated[a + 1 : b] = True

    return out_x, out_y, is_interpolated


def compute_ball_velocity(
    bx: np.ndarray,
    by: np.ndarray,
    window: int = 6,
    fps: float = DEFAULT_FPS,
    pitch_length: float = PITCH_LENGTH_METERS,
    pitch_width: float = PITCH_WIDTH_METERS,
    max_speed: float = 45.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes ball velocity vector (vx, vy) and scalar speed in m/s using finite differences.

    Parameters
    ----------
    bx, by : np.ndarray
        Normalized ball tracking coordinates (shape: (n_frames,)).
    window : int, default=6
        Finite-difference window width in frames (half-window h = window // 2).
    fps : float, default=25.0
        Tracking frame rate in Hz.
    pitch_length, pitch_width : float
        Pitch dimensions in meters.
    max_speed : float, default=45.0
        Maximum allowable physical speed (m/s).

    Returns
    -------
    vx : np.ndarray
        Ball velocity along pitch length (m/s), shape (n_frames,).
    vy : np.ndarray
        Ball velocity along pitch width (m/s), shape (n_frames,).
    speed : np.ndarray
        Instantaneous ball speed (m/s), shape (n_frames,).
    """
    n_frames = len(bx)
    if n_frames == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    h = max(1, window // 2)
    dt_unit = 1.0 / float(fps)

    vx = np.full(n_frames, np.nan, dtype=np.float32)
    vy = np.full(n_frames, np.nan, dtype=np.float32)

    # 1. Fast central difference for interior frames with valid endpoints
    for t in range(n_frames):
        if not (np.isfinite(bx[t]) and np.isfinite(by[t])):
            continue

        # Try symmetric step from h down to 1
        computed = False
        for step in range(h, 0, -1):
            t_pre = t - step
            t_post = t + step
            if 0 <= t_pre and t_post < n_frames:
                if (np.isfinite(bx[t_pre]) and np.isfinite(bx[t_post]) and
                    np.isfinite(by[t_pre]) and np.isfinite(by[t_post])):
                    dt = 2.0 * step * dt_unit
                    vx[t] = (bx[t_post] - bx[t_pre]) * pitch_length / dt
                    vy[t] = (by[t_post] - by[t_pre]) * pitch_width / dt
                    computed = True
                    break

        if not computed:
            # Try forward difference
            for step in range(h, 0, -1):
                t_post = t + step
                if t_post < n_frames and np.isfinite(bx[t_post]) and np.isfinite(by[t_post]):
                    dt = step * dt_unit
                    vx[t] = (bx[t_post] - bx[t]) * pitch_length / dt
                    vy[t] = (by[t_post] - by[t]) * pitch_width / dt
                    computed = True
                    break

        if not computed:
            # Try backward difference
            for step in range(h, 0, -1):
                t_pre = t - step
                if t_pre >= 0 and np.isfinite(bx[t_pre]) and np.isfinite(by[t_pre]):
                    dt = step * dt_unit
                    vx[t] = (bx[t] - bx[t_pre]) * pitch_length / dt
                    vy[t] = (by[t] - by[t_pre]) * pitch_width / dt
                    computed = True
                    break

    speed = np.sqrt(vx**2 + vy**2)
    speed = np.clip(speed, 0.0, max_speed)
    return vx, vy, speed


def compute_ball_rest_indicator(
    speed: np.ndarray,
    rest_speed_threshold: float = 0.20,
    min_rest_frames: int = 25,
    causal: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes boolean indicator and run-length duration of ball at rest.

    Parameters
    ----------
    speed : np.ndarray
        Ball speed in m/s (shape: (n_frames,)).
    rest_speed_threshold : float, default=0.20
        Speed threshold in m/s below which ball is considered stationary.
    min_rest_frames : int, default=25
        Minimum consecutive frames (e.g. 1.0s at 25 fps) to assert rest flag.
    causal : bool, default=True
        If True, ball_rest_flag is True only from frame where duration reaches min_rest_frames.
        If False, ball_rest_flag marks the entire run where length >= min_rest_frames.

    Returns
    -------
    ball_rest_flag : np.ndarray
        Boolean array, True if ball has been stationary for >= min_rest_frames.
    ball_rest_duration : np.ndarray
        Integer array of consecutive stationary frames up to frame t.
    """
    n_frames = len(speed)
    if n_frames == 0:
        return np.array([], dtype=bool), np.array([], dtype=np.int32)

    is_rest_inst = np.isfinite(speed) & (speed < rest_speed_threshold)
    rest_duration = np.zeros(n_frames, dtype=np.int32)

    current_count = 0
    for t in range(n_frames):
        if is_rest_inst[t]:
            current_count += 1
            rest_duration[t] = current_count
        else:
            current_count = 0

    if causal:
        ball_rest_flag = rest_duration >= min_rest_frames
    else:
        ball_rest_flag = np.zeros(n_frames, dtype=bool)
        cur_run = 0
        start_idx = 0
        for t in range(n_frames):
            if is_rest_inst[t]:
                if cur_run == 0:
                    start_idx = t
                cur_run += 1
            else:
                if cur_run >= min_rest_frames:
                    ball_rest_flag[start_idx:t] = True
                cur_run = 0
        if cur_run >= min_rest_frames:
            ball_rest_flag[start_idx:n_frames] = True

    return ball_rest_flag, rest_duration


def compute_pitch_boundary_indicator(
    bx: np.ndarray,
    by: np.ndarray,
    margin: float = 0.005,
) -> np.ndarray:
    """Computes boolean mask of ball outside pitch boundary buffer.

    Parameters
    ----------
    bx, by : np.ndarray
        Normalized ball tracking coordinates [0, 1].
    margin : float, default=0.005
        Normalized buffer distance outside the boundary.

    Returns
    -------
    out_of_bounds : np.ndarray
        Boolean array, True where ball is outside the buffered pitch boundary.
    """
    if len(bx) == 0:
        return np.array([], dtype=bool)

    seen = np.isfinite(bx) & np.isfinite(by)
    out = np.zeros(len(bx), dtype=bool)
    out[seen] = (
        (bx[seen] < -margin)
        | (bx[seen] > 1.0 + margin)
        | (by[seen] < -margin)
        | (by[seen] > 1.0 + margin)
    )
    return out


def process_ball_trajectory(
    bx: np.ndarray,
    by: np.ndarray,
    max_gap: int = 40,
    velocity_window: int = 6,
    fps: float = DEFAULT_FPS,
    pitch_length: float = PITCH_LENGTH_METERS,
    pitch_width: float = PITCH_WIDTH_METERS,
    check_consistency: bool = True,
) -> Dict[str, np.ndarray]:
    """Full processing pipeline for ball tracking trajectory.

    Returns dictionary containing:
    - 'bx', 'by': Interpolated normalized ball coordinates.
    - 'is_interpolated': Boolean mask of interpolated frames.
    - 'vx', 'vy': Ball velocity vector components (m/s).
    - 'speed': Ball speed magnitude (m/s).
    - 'ball_rest_flag': Stationary ball indicator.
    - 'ball_rest_duration': Stationary duration in frames.
    - 'out_of_bounds': Out-of-bounds indicator.
    """
    bx_interp, by_interp, is_interp = interpolate_ball_trajectory(
        bx, by,
        max_gap=max_gap,
        fps=fps,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        check_consistency=check_consistency,
    )
    vx, vy, speed = compute_ball_velocity(
        bx_interp, by_interp,
        window=velocity_window,
        fps=fps,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )
    rest_flag, rest_dur = compute_ball_rest_indicator(speed)
    out_mask = compute_pitch_boundary_indicator(bx_interp, by_interp)

    return {
        "bx": bx_interp,
        "by": by_interp,
        "is_interpolated": is_interp,
        "vx": vx,
        "vy": vy,
        "speed": speed,
        "ball_rest_flag": rest_flag,
        "ball_rest_duration": rest_dur,
        "out_of_bounds": out_mask,
    }
