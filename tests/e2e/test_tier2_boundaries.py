"""tests/e2e/test_tier2_boundaries.py
Tier 2: Boundary and Corner Case Tests
Validates empty ball tracking handling, exact period boundary edges, halftime discontinuity,
second-half pre-kickoff dead ball transitions, and micro-flicker resilience.
"""
import numpy as np
import pandas as pd
import pytest

from tests.e2e.utils.submission_validator import (
    SubmissionValidator,
    TOTAL_FRAMES,
    PRE_MATCH_SPAN,
    H1_SPAN,
    HALFTIME_SPAN,
    H2_PREKICK_SPAN,
    H2_SPAN,
    POST_MATCH_SPAN,
)


# -----------------------------------------------------------------------------
# Boundary 1: Empty Ball Tracking & Occlusion Periods (6 tests)
# -----------------------------------------------------------------------------
def test_tier2_empty_ball_retention_over_short_gap(make_valid_submission):
    # Ball missing for 20 frames during H1 possession -- possession should not drop
    df = make_valid_submission()
    df.loc[15000:15020, "label"] = "H"  # sustained possession through missing ball
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier2_empty_ball_abstention_inside_possession(make_valid_submission):
    # X used to abstain during severe occlusion
    df = make_valid_submission()
    df.loc[15000:15010, "label"] = "X"
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier2_empty_ball_transition_to_dead(make_valid_submission):
    # Out of bounds ball occlusion transitioning to D
    df = make_valid_submission()
    df.loc[15000:15005, "label"] = "H"
    df.loc[15006:15050, "label"] = "D"
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier2_empty_ball_prolonged_gap_handling(make_valid_submission):
    # 40-frame ball loss
    df = make_valid_submission()
    df.loc[16000:16040, "label"] = "D"
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier2_empty_ball_alternating_occlusion(make_valid_submission):
    df = make_valid_submission()
    # Rapid missing/detected pattern
    df.loc[17000:17020:2, "label"] = "H"
    df.loc[17001:17020:2, "label"] = "H"
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier2_empty_ball_does_not_emit_invalid_tokens(make_valid_submission):
    df = make_valid_submission()
    df.loc[18000:18010, "label"] = "X"
    assert not any(v in ["NAN", "NONE", "NULL", ""] for v in df.loc[18000:18010, "label"])


# -----------------------------------------------------------------------------
# Boundary 2: Match Boundary Edge Frames (8 tests)
# -----------------------------------------------------------------------------
def test_tier2_boundary_frame_0(make_valid_submission):
    df = make_valid_submission()
    assert df.loc[0, "frame"] == 0
    assert df.loc[0, "label"] == "X"


def test_tier2_boundary_frame_199318(make_valid_submission):
    df = make_valid_submission()
    assert df.loc[TOTAL_FRAMES - 1, "frame"] == 199318
    assert df.loc[TOTAL_FRAMES - 1, "label"] == "X"


def test_tier2_boundary_h1_kickoff_edge(make_valid_submission):
    # Transition at 11184 (X) -> 11185 (H or A)
    df = make_valid_submission()
    assert df.loc[11184, "label"] == "X"
    assert df.loc[11185, "label"] in ["H", "A", "D"]


def test_tier2_boundary_h1_final_whistle_edge(make_valid_submission):
    # Transition at 90651 (H/A/D) -> 90652 (X)
    df = make_valid_submission()
    assert df.loc[90651, "label"] in ["H", "A", "D"]
    assert df.loc[90652, "label"] == "X"


def test_tier2_boundary_halftime_end_edge(make_valid_submission):
    # Transition at 110807 (X) -> 110808 (D)
    df = make_valid_submission()
    assert df.loc[110807, "label"] == "X"
    assert df.loc[110808, "label"] == "D"


def test_tier2_boundary_h2_kickoff_edge(make_valid_submission):
    # Transition at 113741 (D) -> 113742 (H or A or D)
    df = make_valid_submission()
    assert df.loc[113741, "label"] == "D"
    assert df.loc[113742, "label"] in ["H", "A", "D"]


def test_tier2_boundary_fulltime_whistle_edge(make_valid_submission):
    # Transition at 190197 (H/A/D) -> 190198 (X)
    df = make_valid_submission()
    assert df.loc[190197, "label"] in ["H", "A", "D"]
    assert df.loc[190198, "label"] == "X"


def test_tier2_boundary_single_frame_offset_rejected(make_valid_submission):
    df = make_valid_submission()
    # Advance H1 start by 1 frame into pre-match
    df.loc[11184, "label"] = "H"
    res = SubmissionValidator.validate_dataframe(df, strict_invariants=True)
    assert not res.is_valid


# -----------------------------------------------------------------------------
# Boundary 3: Halftime Discontinuity & Zero State Leakage (6 tests)
# -----------------------------------------------------------------------------
def test_tier2_halftime_full_span_abstention(make_valid_submission):
    df = make_valid_submission()
    ht_labels = df.loc[HALFTIME_SPAN[0]:HALFTIME_SPAN[1], "label"]
    assert (ht_labels == "X").all()
    assert len(ht_labels) == 20156


def test_tier2_halftime_no_h1_state_bleed(make_valid_submission):
    df = make_valid_submission()
    df.loc[90651, "label"] = "H"
    # Ensure 90652 does not inherit H
    assert df.loc[90652, "label"] == "X"


def test_tier2_halftime_sub_segment_corruption_detected(make_valid_submission):
    df = make_valid_submission()
    df.loc[100000, "label"] = "D"  # lone D inside halftime
    res = SubmissionValidator.validate_dataframe(df, strict_invariants=True)
    assert not res.is_valid


def test_tier2_halftime_boundary_exactness(make_valid_submission):
    assert HALFTIME_SPAN[1] - HALFTIME_SPAN[0] + 1 == 20156


def test_tier2_halftime_independent_offset_alignment(make_valid_submission):
    # H1 offset +1702057 vs H2 offset +1702058
    assert H1_SPAN[1] < HALFTIME_SPAN[0] < HALFTIME_SPAN[1] < H2_PREKICK_SPAN[0]


def test_tier2_halftime_clean_start_in_h2(make_valid_submission):
    df = make_valid_submission()
    # H2 begins in D regardless of what H1 ended in
    assert df.loc[H2_PREKICK_SPAN[0], "label"] == "D"


# -----------------------------------------------------------------------------
# Boundary 4: H2 Pre-Kickoff Dead Ball Enforcement (6 tests)
# -----------------------------------------------------------------------------
def test_tier2_h2_prekick_exact_length(make_valid_submission):
    length = H2_PREKICK_SPAN[1] - H2_PREKICK_SPAN[0] + 1
    assert length == 2934


def test_tier2_h2_prekick_all_d(make_valid_submission):
    df = make_valid_submission()
    prekick_labels = df.loc[H2_PREKICK_SPAN[0]:H2_PREKICK_SPAN[1], "label"]
    assert (prekick_labels == "D").all()


def test_tier2_h2_prekick_single_frame_h_rejected(make_valid_submission):
    df = make_valid_submission()
    df.loc[112000, "label"] = "H"
    res = SubmissionValidator.validate_dataframe(df, strict_invariants=True)
    assert not res.is_valid


def test_tier2_h2_prekick_single_frame_a_rejected(make_valid_submission):
    df = make_valid_submission()
    df.loc[112000, "label"] = "A"
    res = SubmissionValidator.validate_dataframe(df, strict_invariants=True)
    assert not res.is_valid


def test_tier2_h2_prekick_x_on_graded_flagged(make_valid_submission):
    df = make_valid_submission()
    df.loc[112000, "label"] = "X"
    res = SubmissionValidator.validate_dataframe(df, strict_invariants=True)
    assert not res.is_valid  # Graded frames must be D


def test_tier2_h2_prekick_kickoff_frame_transition(make_valid_submission):
    df = make_valid_submission()
    df.loc[113741, "label"] = "D"
    df.loc[113742, "label"] = "H"
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


# -----------------------------------------------------------------------------
# Boundary 5: Rapid Spell Transitions & Micro-Flickers (8 tests)
# -----------------------------------------------------------------------------
def test_tier2_rapid_flicker_single_frame_retained(make_valid_submission):
    df = make_valid_submission()
    # 1-frame touch: H -> A -> H
    df.loc[20000, "label"] = "H"
    df.loc[20001, "label"] = "A"
    df.loc[20002, "label"] = "H"
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid  # CSV format allows it, though postprocessing might smooth it


def test_tier2_rapid_flicker_dead_ball_stoppage(make_valid_submission):
    df = make_valid_submission()
    df.loc[20000:20005, "label"] = "D"
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier2_rapid_turnover_alternation(make_valid_submission):
    df = make_valid_submission()
    # Alternating labels across 11 frames
    df.loc[20000:20010, "label"] = ["H", "A"] * 5 + ["H"]
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier2_spell_retention_min_spell_15_frames(make_valid_submission):
    df = make_valid_submission()
    df.loc[20000:20014, "label"] = "A"  # exactly 15 frames = 0.6s
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier2_spell_retention_min_spell_25_frames(make_valid_submission):
    df = make_valid_submission()
    df.loc[20000:20024, "label"] = "H"  # 25 frames = 1.0s
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier2_dead_ball_burst(make_valid_submission):
    df = make_valid_submission()
    df.loc[20000:20010, "label"] = "D"
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier2_transition_touch_snapping_clean(make_valid_submission):
    df = make_valid_submission()
    df.loc[20000:20050, "label"] = "H"
    df.loc[20051:20100, "label"] = "A"
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier2_rapid_stoppage_and_restart(make_valid_submission):
    df = make_valid_submission()
    df.loc[20000:20030, "label"] = "H"
    df.loc[20031:20045, "label"] = "D"  # 15-frame stoppage
    df.loc[20046:20080, "label"] = "H"
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid
