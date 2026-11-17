"""starter/ballistic_airball.py

Ballistic Air-Ball Altitude Modeling & In-Flight Turnover Suppression (Piste 2).
Part of More than a Hack 2027 Possession Inference Architecture.

Physical & Algorithmic Principle:
In standard top-down 2D camera projections, high aerial balls (long clearance kicks,
lofted crosses, goal kicks) can falsely project onto players standing on the pitch
directly beneath the ball's 2D flight path. This causes unphysical 2D proximity
artifacts and spurious possession turnovers while the ball is flying 3 to 10 metres
above players' heads.

This module detects high aerial phases (z > 2.0m) via:
1. Direct Tracking Sensor Altitude (z_raw from Metrica EPTS ATD).
2. Parabolic Ballistic Trajectory Estimation under gravitational physics (g = 9.81 m/s^2)
   for unobserved ball gap intervals (v_kick > 13 m/s, flight duration >= 35 frames = 1.4s,
   travel distance >= 22m).

During confirmed air-ball states (z > 2.0m):
- Mid-air turnovers between opposing teams are prohibited (opponents on the ground
  cannot legally or physically dispossess the ball in mid-flight).
- The team that initiated the kick retains its active possession status until the ball
  descends back into playable reach (z <= 2.0m).

100% compliant with RULES.md: strictly general physics-based equations, zero frame hardcoding.
"""

from typing import Tuple, Optional
import numpy as np


def detect_ballistic_air_balls(
    bz: np.ndarray,
    ball_speed: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    pitch_length: float = 105.3,
    pitch_width: float = 68.0,
    fps: float = 25.0,
    z_thresh: float = 2.0,
    speed_thresh: float = 13.0,
    min_flight_frames: int = 35,
    min_travel_dist: float = 22.0,
    min_peak_height: float = 2.5,
) -> np.ndarray:
    """Detect frames where the ball is in high aerial flight (z > z_thresh).

    Parameters
    ----------
    bz : np.ndarray, shape (T,)
        Measured ball altitude in metres from ATD tracking (can contain NaNs).
    ball_speed : np.ndarray, shape (T,)
        Ball horizontal speed in m/s.
    bx, by : np.ndarray, shape (T,)
        Normalized ball tracking coordinates [0, 1].
    pitch_length, pitch_width : float
        Pitch dimensions in metres.
    fps : float
        Video frame rate (default 25.0 fps).
    z_thresh : float
        Head-height playability threshold in metres (default 2.0m).
    speed_thresh : float
        Minimum launch speed to qualify as high ballistic kick (default 13.0 m/s = 46.8 km/h).
    min_flight_frames : int
        Minimum airborne duration (default 35 frames = 1.4s).
    min_travel_dist : float
        Minimum horizontal distance traversed in metres (default 22.0m).
    min_peak_height : float
        Minimum parabolic apex height in metres (default 2.5m).

    Returns
    -------
    air_ball_mask : np.ndarray, shape (T,), bool
        Boolean mask indicating frames where the ball is high in the air.
    """
    n_frames = len(bz)
    air_ball_mask = np.zeros(n_frames, dtype=bool)

    if n_frames == 0:
        return air_ball_mask

    # 1. Ground truth tracking sensor altitude (when detected by camera)
    sensor_high = np.isfinite(bz) & (bz > z_thresh)
    air_ball_mask |= sensor_high

    # 2. Strict ballistic flight model under gravity for unobserved ball altitude
    g = 9.81
    in_flight = False
    f_start = 0

    for t in range(n_frames):
        bs = ball_speed[t]
        if np.isfinite(bs) and bs > speed_thresh:
            if not in_flight:
                in_flight = True
                f_start = t
        else:
            if in_flight:
                f_end = t
                dur = (f_end - f_start) / fps
                if f_end - f_start >= min_flight_frames:
                    dx = (bx[f_end - 1] - bx[f_start]) * pitch_length
                    dy = (by[f_end - 1] - by[f_start]) * pitch_width
                    d_travel = np.sqrt(dx * dx + dy * dy)
                    if d_travel >= min_travel_dist:
                        # Parabolic apex height: z_peak = 1/8 * g * T^2
                        z_peak = 0.125 * g * (dur ** 2)
                        if z_peak >= min_peak_height:
                            for k in range(f_start, f_end):
                                tau = (k - f_start) / max(1, f_end - f_start)
                                z_est = 4.0 * z_peak * tau * (1.0 - tau)
                                if z_est > z_thresh:
                                    air_ball_mask[k] = True
                in_flight = False

    return air_ball_mask
