"""tests/features/test_kinematic_features.py

Unit tests for kinematic relative velocity, sustained ball control,
pass flight continuation state machine, and spatial stoppage signals.
"""

import numpy as np
import pytest

from src.features.kinematic_features import (
    HOME_SLOTS,
    AWAY_SLOTS,
    compute_ball_rest_indicator,
    compute_pitch_boundary_buffer,
    compute_relative_velocity,
    compute_team_spatial_dispersion,
    detect_kinematic_control,
    extract_kinematic_features,
    track_pass_continuation,
)


def test_relative_velocity_stationary_ball():
    """Ball at rest; player running past at 5 m/s within 0.5m must NOT register control."""
    T = 10
    N = 46
    dt = 0.04

    # Ball stationary at (50, 30)
    bx = np.full(T, 50.0, dtype=np.float32)
    by = np.full(T, 30.0, dtype=np.float32)

    # Player 0 running along X at 5.0 m/s at Y=30.5 (d = 0.5m at t=5)
    px = np.full((T, N), np.nan, dtype=np.float32)
    py = np.full((T, N), np.nan, dtype=np.float32)

    for t in range(T):
        px[t, 0] = 50.0 + (t - 5) * dt * 5.0
        py[t, 0] = 30.5

    dist, v_rel = compute_relative_velocity(px, py, bx, by, dt=dt)

    assert dist[5, 0] == pytest.approx(0.5, abs=0.01)
    assert v_rel[5, 0] == pytest.approx(5.0, abs=0.1)

    home_ctrl, away_ctrl = detect_kinematic_control(
        dist, v_rel, dist_thresh=2.0, vel_thresh=3.0, sustain_frames=1
    )
    # Relative velocity (5.0 m/s) exceeds 3.0 m/s threshold -> no control
    assert not home_ctrl[5]


def test_relative_velocity_dribble():
    """Player and ball moving synchronously at 6 m/s separated by 0.4m must register control."""
    T = 10
    N = 46
    dt = 0.04
    speed = 6.0

    bx = (50.0 + np.arange(T) * dt * speed).astype(np.float32)
    by = np.full(T, 30.0, dtype=np.float32)

    px = np.full((T, N), np.nan, dtype=np.float32)
    py = np.full((T, N), np.nan, dtype=np.float32)

    # Player 0 running with ball at 0.4m offset
    px[:, 0] = bx - 0.4
    py[:, 0] = by

    dist, v_rel = compute_relative_velocity(px, py, bx, by, dt=dt)

    assert np.allclose(dist[:, 0], 0.4, atol=1e-3)
    assert np.allclose(v_rel[1:-1, 0], 0.0, atol=1e-3)

    home_ctrl, away_ctrl = detect_kinematic_control(
        dist, v_rel, dist_thresh=2.0, vel_thresh=3.0, sustain_frames=3
    )
    # Control registered on sustained frames
    assert np.all(home_ctrl[3:-1])


def test_pass_flyby_rejection():
    """Pass flying at 15 m/s passing 0.8m from stationary defender must be rejected."""
    T = 15
    N = 46
    dt = 0.04

    # Ball moving at 15 m/s along X, crossing near X=50 at t=10
    bx = (44.0 + np.arange(T) * dt * 15.0).astype(np.float32)
    by = np.full(T, 30.8, dtype=np.float32)

    # Away defender (slot 24) stationary at (50, 30)
    px = np.full((T, N), np.nan, dtype=np.float32)
    py = np.full((T, N), np.nan, dtype=np.float32)
    px[:, 24] = 50.0
    py[:, 24] = 30.0

    dist, v_rel = compute_relative_velocity(px, py, bx, by, dt=dt)

    # Closest approach at t=10 is (50, 30.8) vs (50, 30) -> 0.8m
    min_dist = np.nanmin(dist[:, 24])
    assert min_dist <= 1.0

    # Relative velocity is ~15 m/s
    assert np.all(v_rel[1:-1, 24] > 14.0)

    home_ctrl, away_ctrl = detect_kinematic_control(
        dist, v_rel, dist_thresh=2.0, vel_thresh=3.0, sustain_frames=1
    )
    assert not np.any(away_ctrl)


def test_sustained_control_window():
    """Candidate control for 2 frames (< K=3) is False; for 3 frames (= K=3) is True."""
    T = 10
    N = 46
    dist = np.full((T, N), 5.0, dtype=np.float32)
    v_rel = np.full((T, N), 5.0, dtype=np.float32)

    # Slot 0 candidate control active for exactly 2 frames (t=2, 3)
    dist[2:4, 0] = 1.0
    v_rel[2:4, 0] = 1.0

    home_ctrl, _ = detect_kinematic_control(dist, v_rel, sustain_frames=3)
    assert not np.any(home_ctrl)

    # Now make it active for 3 frames (t=2, 3, 4)
    dist[4, 0] = 1.0
    v_rel[4, 0] = 1.0

    home_ctrl, _ = detect_kinematic_control(dist, v_rel, sustain_frames=3)
    assert not home_ctrl[2]
    assert not home_ctrl[3]
    assert home_ctrl[4] == True  # 3rd consecutive frame


def test_pass_continuation_preserves_possession():
    """Home control at 0..10 followed by pass flight at 12 m/s retains Home possession."""
    T = 40
    ball_speed = np.full(T, 12.0, dtype=np.float32)
    home_ctrl = np.zeros(T, dtype=bool)
    away_ctrl = np.zeros(T, dtype=bool)

    home_ctrl[0:11] = True  # frames 0..10

    pass_team = track_pass_continuation(
        ball_speed, home_ctrl, away_ctrl, speed_thresh=4.0, max_flight_frames=75
    )

    # Home possession maintained across flight frames 11..35
    assert np.all(pass_team[0:36] == 0)


def test_pass_continuation_interception_turnover():
    """Home kick at frame 10 intercepted by Away control at frame 25 flips possession."""
    T = 40
    ball_speed = np.full(T, 10.0, dtype=np.float32)
    home_ctrl = np.zeros(T, dtype=bool)
    away_ctrl = np.zeros(T, dtype=bool)

    home_ctrl[0:10] = True
    away_ctrl[25:35] = True

    pass_team = track_pass_continuation(
        ball_speed, home_ctrl, away_ctrl, speed_thresh=4.0
    )

    assert np.all(pass_team[0:10] == 0)
    assert np.all(pass_team[10:25] == 0)  # flight retains Home
    assert np.all(pass_team[25:35] == 1)  # flips to Away


def test_pass_continuation_timeout():
    """Pass flight without receiver decays to -1 after max_flight_frames."""
    T = 100
    ball_speed = np.full(T, 8.0, dtype=np.float32)
    home_ctrl = np.zeros(T, dtype=bool)
    away_ctrl = np.zeros(T, dtype=bool)

    home_ctrl[0:5] = True

    pass_team = track_pass_continuation(
        ball_speed, home_ctrl, away_ctrl, speed_thresh=4.0, max_flight_frames=75
    )

    # First 5 frames are Home
    assert pass_team[4] == 0
    # Flight frames up to 5 + 75 = 80 remain Home
    assert pass_team[78] == 0
    # After frame 80, timed out to -1
    assert pass_team[81] == -1


def test_spatial_dispersion_tactical_compression():
    """Wide live formation has high dispersion (>1000); compressed huddle has low (<300)."""
    T = 2
    N_team = 11

    # Frame 0: Wide live formation (40m x 30m)
    px = np.zeros((T, N_team), dtype=np.float32)
    py = np.zeros((T, N_team), dtype=np.float32)

    # GK at X=5.0
    px[0, 0] = 5.0
    py[0, 0] = 34.0
    # 10 outfield players spread from X=30 to 70, Y=10 to 60
    for i in range(1, 11):
        px[0, i] = 30.0 + (i - 1) * 4.0
        py[0, i] = 10.0 + ((i - 1) % 5) * 12.0

    # Frame 1: Compressed huddle (8m x 8m)
    px[1, 0] = 5.0
    py[1, 0] = 34.0
    for i in range(1, 11):
        px[1, i] = 50.0 + ((i - 1) % 3) * 2.0
        py[1, i] = 30.0 + ((i - 1) // 3) * 2.0

    disp = compute_team_spatial_dispersion(px, py, is_home=True, halftime_frame=100)

    assert disp[0] > 1000.0
    assert disp[1] < 300.0


def test_spatial_dispersion_goalkeeper_filtering():
    """Verifies that goalkeeper position does not distort outfield dispersion."""
    T = 1
    N_team = 10

    px = np.zeros((T, N_team), dtype=np.float32)
    py = np.zeros((T, N_team), dtype=np.float32)

    # Home GK at X=2.0m in first half
    px[0, 0] = 2.0
    py[0, 0] = 34.0

    # 9 outfield players in a tight cluster at X=50..52
    for i in range(1, 10):
        px[0, i] = 50.0 + (i % 3) * 1.0
        py[0, i] = 34.0 + (i // 3) * 1.0

    disp = compute_team_spatial_dispersion(px, py, is_home=True, halftime_frame=100)

    # Without GK filtering, X=2.0 would inflate std(X) to ~15m and dispersion to thousands.
    # With GK filtering, outfield spread is very small (< 100 m^2).
    assert disp[0] < 100.0

    # Edge case: fewer than 4 detected players returns 0.0
    px_sparse = np.full((1, N_team), np.nan, dtype=np.float32)
    py_sparse = np.full((1, N_team), np.nan, dtype=np.float32)
    px_sparse[0, :3] = [10.0, 20.0, 30.0]
    py_sparse[0, :3] = [10.0, 20.0, 30.0]
    disp_sparse = compute_team_spatial_dispersion(px_sparse, py_sparse, is_home=True)
    assert disp_sparse[0] == 0.0


def test_ball_rest_indicator_threshold():
    """Ball speed < 0.2 m/s for 20 frames is False; for 26 frames is True on all 26 frames."""
    speed20 = np.full(20, 0.10, dtype=np.float32)
    flag20 = compute_ball_rest_indicator(speed20, thresh=0.2, min_frames=25)
    assert not np.any(flag20)

    speed26 = np.full(26, 0.10, dtype=np.float32)
    flag26 = compute_ball_rest_indicator(speed26, thresh=0.2, min_frames=25)
    assert np.all(flag26)


def test_pitch_boundary_buffer_geometry():
    """Tests metric pitch boundary excursion with 1.0m buffer."""
    bx = np.array([50.0, 105.8, 107.0], dtype=np.float32)
    by = np.array([30.0, 34.0, 34.0], dtype=np.float32)

    out_of_bounds, d_out = compute_pitch_boundary_buffer(bx, by, pitch_length=105.3, pitch_width=68.0, buffer=1.0)

    # Interior
    assert d_out[0] == 0.0
    assert out_of_bounds[0] == False

    # 105.8 - 105.3 = 0.5m outside, <= 1.0m buffer
    assert d_out[1] == pytest.approx(0.5, abs=1e-3)
    assert out_of_bounds[1] == False

    # 107.0 - 105.3 = 1.7m outside, > 1.0m buffer
    assert d_out[2] == pytest.approx(1.7, abs=1e-3)
    assert out_of_bounds[2] == True


def test_extract_kinematic_features_contract():
    """Verifies interface contract: all 14 keys returned with exact shapes and valid types."""
    T = 50
    N = 46

    px = np.random.uniform(0.1, 0.9, (T, N)).astype(np.float32)
    py = np.random.uniform(0.1, 0.9, (T, N)).astype(np.float32)
    bx = np.random.uniform(0.1, 0.9, T).astype(np.float32)
    by = np.random.uniform(0.1, 0.9, T).astype(np.float32)
    ball_speed = np.random.uniform(0.0, 20.0, T).astype(np.float32)

    # Inject some NaNs
    bx[10:15] = np.nan
    by[10:15] = np.nan
    ball_speed[10:15] = np.nan

    res = extract_kinematic_features(px, py, bx, by, ball_speed)

    expected_keys = [
        "d_home_min", "d_away_min",
        "rel_vel_home", "rel_vel_away",
        "home_control", "away_control",
        "pass_team",
        "team_spread_home", "team_spread_away",
        "ball_speed", "ball_rest_flag",
        "boundary_dist", "out_of_bounds",
        "is_interpolated",
    ]

    for k in expected_keys:
        assert k in res, f"Missing key: {k}"
        assert len(res[k]) == T, f"Key {k} has wrong length: {len(res[k])} vs {T}"

    assert res["d_home_min"].dtype == np.float32
    assert res["d_away_min"].dtype == np.float32
    assert res["rel_vel_home"].dtype == np.float32
    assert res["rel_vel_away"].dtype == np.float32
    assert res["home_control"].dtype == bool
    assert res["away_control"].dtype == bool
    assert res["pass_team"].dtype == np.int32
    assert res["team_spread_home"].dtype == np.float32
    assert res["team_spread_away"].dtype == np.float32
    assert res["ball_speed"].dtype == np.float32
    assert res["ball_rest_flag"].dtype == bool
    assert res["boundary_dist"].dtype == np.float32
    assert res["out_of_bounds"].dtype == bool
    assert res["is_interpolated"].dtype == bool
