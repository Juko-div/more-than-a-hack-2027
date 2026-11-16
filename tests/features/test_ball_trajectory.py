"""tests/features/test_ball_trajectory.py

Unit tests for finite-difference ball velocity, momentum-preserving gap interpolation,
and kinematic ball indicators.
"""

import numpy as np
import pytest

from src.features.ball_trajectory import (
    PITCH_LENGTH_METERS,
    PITCH_WIDTH_METERS,
    compute_ball_rest_indicator,
    compute_ball_velocity,
    compute_pitch_boundary_indicator,
    interpolate_ball_trajectory,
    process_ball_trajectory,
)


def test_interpolate_empty_and_all_nan():
    """Tests edge cases: empty input, all-NaN array, and single valid point."""
    empty_x = np.array([], dtype=np.float32)
    empty_y = np.array([], dtype=np.float32)
    bx_i, by_i, mask_i = interpolate_ball_trajectory(empty_x, empty_y)
    assert len(bx_i) == 0
    assert len(by_i) == 0
    assert len(mask_i) == 0

    all_nan_x = np.full(100, np.nan, dtype=np.float32)
    all_nan_y = np.full(100, np.nan, dtype=np.float32)
    bx_i, by_i, mask_i = interpolate_ball_trajectory(all_nan_x, all_nan_y)
    assert np.all(np.isnan(bx_i))
    assert np.all(np.isnan(by_i))
    assert not np.any(mask_i)

    single_x = all_nan_x.copy()
    single_y = all_nan_y.copy()
    single_x[10] = 0.5
    single_y[10] = 0.5
    bx_i, by_i, mask_i = interpolate_ball_trajectory(single_x, single_y)
    assert bx_i[10] == 0.5
    assert not np.any(mask_i)


def test_interpolate_constant_velocity_flight():
    """Tests interpolation of a plausible linear pass at 15 m/s."""
    # 100 frames at 25 fps = 4.0 seconds
    fps = 25.0
    dt = 1.0 / fps
    speed = 15.0  # m/s along X
    # X in meters: from 10.0m to 10.0 + 100 * dt * 15.0 = 70.0m
    x_m = 10.0 + np.arange(100) * dt * speed
    y_m = np.full(100, 34.0)

    bx = (x_m / PITCH_LENGTH_METERS).astype(np.float32)
    by = (y_m / PITCH_WIDTH_METERS).astype(np.float32)

    # Inject 10-frame gap from 40 to 49
    bx_gap = bx.copy()
    by_gap = by.copy()
    bx_gap[40:50] = np.nan
    by_gap[40:50] = np.nan

    bx_i, by_i, mask_i = interpolate_ball_trajectory(bx_gap, by_gap, max_gap=40, fps=fps)

    assert np.all(mask_i[40:50])
    assert not np.any(mask_i[:40])
    assert not np.any(mask_i[50:])
    np.testing.assert_allclose(bx_i, bx, atol=1e-5)
    np.testing.assert_allclose(by_i, by, atol=1e-5)


def test_interpolate_rejects_teleportation_jump():
    """Tests that supersonic jumps (> 35 m/s) are rejected."""
    # Ball at (10m, 10m) at t=10; missing t=11, 12; ball at (80m, 60m) at t=13
    bx = np.full(20, np.nan, dtype=np.float32)
    by = np.full(20, np.nan, dtype=np.float32)

    bx[10] = 10.0 / PITCH_LENGTH_METERS
    by[10] = 10.0 / PITCH_WIDTH_METERS

    bx[13] = 80.0 / PITCH_LENGTH_METERS
    by[13] = 60.0 / PITCH_WIDTH_METERS

    bx_i, by_i, mask_i = interpolate_ball_trajectory(bx, by, max_gap=40)
    assert not np.any(mask_i)
    assert np.isnan(bx_i[11])
    assert np.isnan(bx_i[12])


def test_interpolate_rejects_tackle_direction_reversal():
    """Tests that sudden direction reversals across a gap (tackle) are rejected."""
    fps = 25.0
    dt = 1.0 / fps
    speed = 12.0  # m/s

    # Ball moving +X for t=0..20
    x_m = np.zeros(45, dtype=np.float32)
    for t in range(21):
        x_m[t] = 10.0 + t * dt * speed

    # Gap at t=21..25 (5 frames)

    # Ball moving -X for t=26..40
    for t in range(26, 45):
        x_m[t] = 20.0 - (t - 26) * dt * speed

    y_m = np.full(45, 34.0, dtype=np.float32)

    bx = x_m / PITCH_LENGTH_METERS
    by = y_m / PITCH_WIDTH_METERS
    bx[21:26] = np.nan
    by[21:26] = np.nan

    bx_i, by_i, mask_i = interpolate_ball_trajectory(bx, by, max_gap=40, fps=fps)
    assert not np.any(mask_i)
    assert np.all(np.isnan(bx_i[21:26]))


def test_interpolate_gap_length_limit():
    """Tests that gaps > max_gap are preserved as NaNs while gaps <= max_gap are filled."""
    fps = 25.0
    dt = 1.0 / fps
    speed = 8.0

    x_m = 10.0 + np.arange(100) * dt * speed
    y_m = np.full(100, 34.0)
    bx = (x_m / PITCH_LENGTH_METERS).astype(np.float32)
    by = (y_m / PITCH_WIDTH_METERS).astype(np.float32)

    # 1. Gap of 41 frames (t=10..50)
    bx_gap41 = bx.copy()
    by_gap41 = by.copy()
    bx_gap41[10:51] = np.nan
    by_gap41[10:51] = np.nan
    _, _, mask41 = interpolate_ball_trajectory(bx_gap41, by_gap41, max_gap=40)
    assert not np.any(mask41)

    # 2. Gap of 35 frames (t=10..44)
    bx_gap35 = bx.copy()
    by_gap35 = by.copy()
    bx_gap35[10:45] = np.nan
    by_gap35[10:45] = np.nan
    bx_i35, _, mask35 = interpolate_ball_trajectory(bx_gap35, by_gap35, max_gap=40)
    assert np.all(mask35[10:45])
    assert np.all(np.isfinite(bx_i35[10:45]))


def test_interpolate_stationary_ball():
    """Tests interpolation of a stationary ball at rest without zero-division."""
    bx = np.full(30, 0.5, dtype=np.float32)
    by = np.full(30, 0.5, dtype=np.float32)
    bx[10:20] = np.nan
    by[10:20] = np.nan

    bx_i, by_i, mask_i = interpolate_ball_trajectory(bx, by, max_gap=40)
    assert np.all(mask_i[10:20])
    np.testing.assert_allclose(bx_i[10:20], 0.5)
    np.testing.assert_allclose(by_i[10:20], 0.5)


def test_compute_velocity_accuracy_and_window():
    """Tests finite-difference velocity calculation on constant motion."""
    fps = 25.0
    dt = 1.0 / fps
    vx_true = 10.0  # m/s
    vy_true = 5.0   # m/s

    n = 50
    x_m = np.arange(n) * dt * vx_true
    y_m = np.arange(n) * dt * vy_true

    bx = (x_m / PITCH_LENGTH_METERS).astype(np.float32)
    by = (y_m / PITCH_WIDTH_METERS).astype(np.float32)

    vx, vy, speed = compute_ball_velocity(bx, by, window=6, fps=fps)

    np.testing.assert_allclose(vx[3:-3], vx_true, atol=1e-3)
    np.testing.assert_allclose(vy[3:-3], vy_true, atol=1e-3)
    expected_speed = np.sqrt(vx_true**2 + vy_true**2)
    np.testing.assert_allclose(speed[3:-3], expected_speed, atol=1e-3)


def test_compute_velocity_boundary_handling():
    """Verifies boundary frames receive valid forward/backward difference velocities."""
    fps = 25.0
    dt = 1.0 / fps
    vx_true = 8.0
    vy_true = 0.0

    n = 20
    x_m = np.arange(n) * dt * vx_true
    y_m = np.full(n, 34.0)

    bx = (x_m / PITCH_LENGTH_METERS).astype(np.float32)
    by = (y_m / PITCH_WIDTH_METERS).astype(np.float32)

    vx, vy, speed = compute_ball_velocity(bx, by, window=6, fps=fps)

    # First frames must be finite
    assert np.all(np.isfinite(vx[:3]))
    assert np.all(np.isfinite(speed[:3]))
    # Last frames must be finite
    assert np.all(np.isfinite(vx[-3:]))
    assert np.all(np.isfinite(speed[-3:]))


def test_ball_rest_indicator():
    """Tests causal detection of stationary ball (speed < 0.20 m/s for >= 25 frames)."""
    # 30 frames stationary (0.10 m/s), then 20 frames moving (5.0 m/s)
    speed = np.full(50, 0.10, dtype=np.float32)
    speed[30:] = 5.0

    rest_flag, rest_dur = compute_ball_rest_indicator(speed, rest_speed_threshold=0.20, min_rest_frames=25, causal=True)

    # Frames 0..23: duration is 1..24 (< 25), so rest_flag is False
    assert not np.any(rest_flag[:24])
    # Frames 24..29: duration is 25..30 (>= 25), so rest_flag is True
    assert np.all(rest_flag[24:30])
    # Frames 30..49: speed is 5.0 m/s, so rest_flag is False
    assert not np.any(rest_flag[30:])


def test_compute_pitch_boundary_indicator():
    """Tests normalized boundary indicator with buffer margin."""
    bx = np.array([0.5, -0.01, 1.02, 0.5, np.nan], dtype=np.float32)
    by = np.array([0.5, 0.5, 0.5, -0.01, np.nan], dtype=np.float32)

    out = compute_pitch_boundary_indicator(bx, by, margin=0.005)
    assert out[0] == False
    assert out[1] == True
    assert out[2] == True
    assert out[3] == True
    assert out[4] == False


def test_full_pipeline_process_ball_trajectory():
    """Tests integrated process_ball_trajectory function."""
    n = 200
    bx = np.full(n, 0.5, dtype=np.float32)
    by = np.full(n, 0.5, dtype=np.float32)
    bx[50:60] = np.nan
    by[50:60] = np.nan

    res = process_ball_trajectory(bx, by, max_gap=40, velocity_window=6)

    assert "bx" in res
    assert "by" in res
    assert "is_interpolated" in res
    assert "vx" in res
    assert "vy" in res
    assert "speed" in res
    assert "ball_rest_flag" in res
    assert "ball_rest_duration" in res
    assert "out_of_bounds" in res

    assert len(res["bx"]) == n
    assert np.all(res["is_interpolated"][50:60])
    assert np.all(np.isfinite(res["speed"]))
