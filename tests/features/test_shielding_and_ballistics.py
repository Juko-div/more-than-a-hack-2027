"""Unit tests for Piste 1 (Defensive Shielding) and Piste 2 (Ballistic Air-Ball)."""

import numpy as np
import pytest

from starter.defensive_shielding import detect_defensive_shielding
from starter.ballistic_airball import detect_ballistic_air_balls


class TestDefensiveShielding:
    """Test suite for defensive shielding and territorial spatial control."""

    def test_empty_inputs(self):
        sh, sa = detect_defensive_shielding(
            np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2)),
            np.empty(0), np.empty(0), np.empty((0, 2)), []
        )
        assert len(sh) == 0
        assert len(sa) == 0
        assert sh.dtype == bool
        assert sa.dtype == bool

    def test_all_nan_ball_no_trigger(self):
        T, N = 100, 22
        px = np.full((T, N), 0.1, dtype=np.float32)
        py = np.full((T, N), 0.5, dtype=np.float32)
        ps = np.full((T, N), 5.0, dtype=np.float32)
        bx = np.full(T, np.nan, dtype=np.float32)
        by = np.full(T, np.nan, dtype=np.float32)
        team_mat = np.zeros((T, N), dtype=np.int32)
        team_mat[:, 11:] = 1
        blocks = [("H1", 0, 49), ("H2", 50, 99)]

        sh, sa = detect_defensive_shielding(px, py, ps, bx, by, team_mat, blocks)
        assert not np.any(sh)
        assert not np.any(sa)

    def test_shielding_h1_home_defensive_third(self):
        # 10 frames in H1
        # Home defends left (x < 0.25).
        # Defender at x=0.10, y=0.10; Attacker at x=0.12, y=0.10 (closer to x=0.0 is defender).
        # Ball at x=0.11, y=0.10.
        T, N = 10, 2
        px = np.zeros((T, N), dtype=np.float32)
        py = np.zeros((T, N), dtype=np.float32)
        ps = np.full((T, N), 6.0, dtype=np.float32)

        # Slot 0 = Home, Slot 1 = Away
        px[:, 0] = 0.10  # Defender deeper (closer to x=0.0)
        py[:, 0] = 0.10
        px[:, 1] = 0.12  # Attacker chasing behind
        py[:, 1] = 0.10

        bx = np.full(T, 0.11, dtype=np.float32)
        by = np.full(T, 0.10, dtype=np.float32)

        team_mat = np.zeros((T, N), dtype=np.int32)
        team_mat[:, 1] = 1

        blocks = [("H1", 0, 9)]

        sh, sa = detect_defensive_shielding(px, py, ps, bx, by, team_mat, blocks, bridge_gap=0)
        assert np.all(sh)
        assert not np.any(sa)

    def test_no_shielding_when_attacker_closer_to_goal(self):
        # In H1: Attacker at x=0.08, Defender at x=0.12 (Attacker is through/deeper)
        T, N = 10, 2
        px = np.zeros((T, N), dtype=np.float32)
        py = np.zeros((T, N), dtype=np.float32)
        ps = np.full((T, N), 6.0, dtype=np.float32)

        px[:, 0] = 0.12  # Defender behind attacker
        py[:, 0] = 0.10
        px[:, 1] = 0.08  # Attacker deeper
        py[:, 1] = 0.10

        bx = np.full(T, 0.09, dtype=np.float32)
        by = np.full(T, 0.10, dtype=np.float32)

        team_mat = np.zeros((T, N), dtype=np.int32)
        team_mat[:, 1] = 1

        blocks = [("H1", 0, 9)]

        sh, sa = detect_defensive_shielding(px, py, ps, bx, by, team_mat, blocks, bridge_gap=0)
        assert not np.any(sh)
        assert not np.any(sa)

    def test_bridging_gap(self):
        # 20 frames, shield triggered at frame 2 and frame 8 (gap = 5 <= 10)
        T, N = 20, 2
        px = np.zeros((T, N), dtype=np.float32)
        py = np.zeros((T, N), dtype=np.float32)
        ps = np.full((T, N), 6.0, dtype=np.float32)

        # Non-triggering by default (outside defensive third)
        px[:, 0] = 0.50
        px[:, 1] = 0.52
        bx = np.full(T, 0.51, dtype=np.float32)
        by = np.full(T, 0.50, dtype=np.float32)

        # Trigger at frame 2 and frame 8
        for f in [2, 8]:
            px[f, 0] = 0.10
            py[f, 0] = 0.10
            px[f, 1] = 0.12
            py[f, 1] = 0.10
            bx[f] = 0.11
            by[f] = 0.10

        team_mat = np.zeros((T, N), dtype=np.int32)
        team_mat[:, 1] = 1
        blocks = [("H1", 0, 19)]

        sh, _ = detect_defensive_shielding(px, py, ps, bx, by, team_mat, blocks, bridge_gap=10)
        assert np.all(sh[2:9])


class TestBallisticAirBall:
    """Test suite for ballistic air-ball modeling and aerial phase detection."""

    def test_empty_inputs(self):
        mask = detect_ballistic_air_balls(
            np.empty(0), np.empty(0), np.empty(0), np.empty(0)
        )
        assert len(mask) == 0
        assert mask.dtype == bool

    def test_sensor_high_altitude(self):
        T = 20
        bz = np.full(T, 1.0, dtype=np.float32)
        bz[5:15] = 3.5  # High sensor altitude
        bs = np.full(T, 5.0, dtype=np.float32)
        bx = np.linspace(0.2, 0.3, T, dtype=np.float32)
        by = np.linspace(0.4, 0.4, T, dtype=np.float32)

        mask = detect_ballistic_air_balls(bz, bs, bx, by, z_thresh=2.0)
        assert np.all(mask[5:15])
        assert not np.any(mask[:5])
        assert not np.any(mask[15:])

    def test_unobserved_altitude_ballistic_trajectory(self):
        # 50 frames with high speed and long distance travel under gravity
        T = 60
        bz = np.full(T, np.nan, dtype=np.float32)
        bs = np.full(T, 5.0, dtype=np.float32)
        bs[10:50] = 16.0  # > 13 m/s for 40 frames
        bx = np.full(T, 0.2, dtype=np.float32)
        # 0.2 to 0.6 on pitch length 105.3m = 42m travel distance
        bx[10:50] = np.linspace(0.2, 0.6, 40, dtype=np.float32)
        by = np.full(T, 0.5, dtype=np.float32)

        mask = detect_ballistic_air_balls(
            bz, bs, bx, by,
            fps=25.0, z_thresh=2.0, speed_thresh=13.0,
            min_flight_frames=35, min_travel_dist=22.0, min_peak_height=2.5,
        )
        # Apex should exceed z_thresh
        assert np.any(mask[10:50])
        # Boundaries outside flight should be False
        assert not np.any(mask[:10])
        assert not np.any(mask[50:])
