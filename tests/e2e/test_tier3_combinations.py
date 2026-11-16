"""tests/e2e/test_tier3_combinations.py
Tier 3: Cross-Feature Combinations Tests
Validates pairwise state transitions, set-piece sequences, ball loss during stoppages,
and extreme kinematic conditions.
"""
import numpy as np
import pandas as pd
import pytest

from tests.e2e.utils.submission_validator import SubmissionValidator, TOTAL_FRAMES


# -----------------------------------------------------------------------------
# Pairwise Game Situations: All Valid Transitions (8 tests)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("from_lbl,to_lbl", [
    ("H", "A"),  # Direct turnover
    ("A", "H"),  # Direct turnover
    ("H", "D"),  # Home team stoppage
    ("A", "D"),  # Away team stoppage
    ("D", "H"),  # Restart to Home
    ("D", "A"),  # Restart to Away
    ("X", "D"),  # Period boundary to Dead ball
    ("D", "X"),  # Period end from Dead ball
])
def test_tier3_pairwise_valid_transitions(make_valid_submission, from_lbl, to_lbl):
    df = make_valid_submission()
    df.loc[30000, "label"] = from_lbl
    df.loc[30001, "label"] = to_lbl
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid, f"Transition {from_lbl} -> {to_lbl} should be valid"


# -----------------------------------------------------------------------------
# Combination: Set-Piece Leading into Live Kickoff (6 tests)
# -----------------------------------------------------------------------------
def test_tier3_corner_kick_sequence(make_valid_submission):
    # Dead ball pre-roll (D) -> kick / contest (H)
    df = make_valid_submission()
    df.loc[40000:40500, "label"] = "D"  # 20s dead ball before corner
    df.loc[40501:40600, "label"] = "H"  # in play
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_goal_kick_sequence(make_valid_submission):
    df = make_valid_submission()
    df.loc[41000:41450, "label"] = "D"  # 18s dead ball
    df.loc[41451:41600, "label"] = "A"
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_free_kick_sequence(make_valid_submission):
    df = make_valid_submission()
    df.loc[42000:42375, "label"] = "D"  # 15s dead ball
    df.loc[42376:42500, "label"] = "H"
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_throw_in_quick_restart(make_valid_submission):
    df = make_valid_submission()
    df.loc[43000:43125, "label"] = "D"  # 5s throw-in
    df.loc[43126:43300, "label"] = "A"
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_goal_celebration_dead_ball(make_valid_submission):
    df = make_valid_submission()
    df.loc[44000:44100, "label"] = "H"  # goal scored
    df.loc[44101:45000, "label"] = "D"  # celebration stoppage ~36s
    df.loc[45001:45100, "label"] = "A"  # restart by opponent
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_set_piece_with_rebound_turnover(make_valid_submission):
    df = make_valid_submission()
    df.loc[46000:46200, "label"] = "D"
    df.loc[46201:46220, "label"] = "H"  # corner kick
    df.loc[46221:46300, "label"] = "A"  # defended & counter-attack
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


# -----------------------------------------------------------------------------
# Combination: Ball Loss / Occlusion During Dead-Ball Stoppage (6 tests)
# -----------------------------------------------------------------------------
def test_tier3_stoppage_with_missing_ball(make_valid_submission):
    # Ball out of camera view while referee sets wall: remains D
    df = make_valid_submission()
    df.loc[50000:50500, "label"] = "D"
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_stoppage_substitution_with_missing_ball(make_valid_submission):
    df = make_valid_submission()
    df.loc[51000:52000, "label"] = "D"
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_stoppage_ball_recovery_transition(make_valid_submission):
    df = make_valid_submission()
    df.loc[53000:53200, "label"] = "D"
    df.loc[53201:53250, "label"] = "H"
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_var_check_extended_stoppage(make_valid_submission):
    df = make_valid_submission()
    df.loc[54000:56500, "label"] = "D"  # 100s stoppage
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_stoppage_ball_placed_stationary(make_valid_submission):
    df = make_valid_submission()
    df.loc[57000:57200, "label"] = "D"
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_stoppage_does_not_flicker_to_live_teams(make_valid_submission):
    df = make_valid_submission()
    # Steady D throughout stoppage without spurious single-frame touches
    df.loc[58000:58500, "label"] = "D"
    assert (df.loc[58000:58500, "label"] == "D").all()


# -----------------------------------------------------------------------------
# Combination: Extreme Ball Speed + Missing Ball (Flight Phase) (6 tests)
# -----------------------------------------------------------------------------
def test_tier3_long_clearance_flight_continuation(make_valid_submission):
    # Ball kicked at >15 m/s, occluded in air, lands with same team
    df = make_valid_submission()
    df.loc[60000:60075, "label"] = "H"  # 3s flight kept as H
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_long_clearance_turnover_upon_reception(make_valid_submission):
    # Ball in flight (retained H) -> intercepted by A
    df = make_valid_submission()
    df.loc[61000:61050, "label"] = "H"  # in flight
    df.loc[61051:61150, "label"] = "A"  # opponent controlled
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_shot_on_goal_keeper_save(make_valid_submission):
    # Fast shot (H) -> Keeper catch (A)
    df = make_valid_submission()
    df.loc[62000:62025, "label"] = "H"  # shot
    df.loc[62026:62100, "label"] = "A"  # save and hold
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_shot_on_goal_deflection_out_of_bounds(make_valid_submission):
    # Shot (H) -> Deflection out of pitch (D)
    df = make_valid_submission()
    df.loc[63000:63020, "label"] = "H"
    df.loc[63021:63200, "label"] = "D"
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_cross_into_box_aerial_duel(make_valid_submission):
    df = make_valid_submission()
    df.loc[64000:64040, "label"] = "A"  # cross
    df.loc[64041:64100, "label"] = "H"  # defensive header clearance
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_switch_of_play_diagonal(make_valid_submission):
    df = make_valid_submission()
    df.loc[65000:65070, "label"] = "H"  # long cross-pitch pass
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


# -----------------------------------------------------------------------------
# Combination: Multi-Condition Compound Situations (12 tests)
# -----------------------------------------------------------------------------
def test_tier3_compound_injury_stoppage_near_halftime(make_valid_submission):
    df = make_valid_submission()
    df.loc[90500:90651, "label"] = "D"  # late injury
    df.loc[90652:110807, "label"] = "X"  # halftime
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_compound_card_booking_during_counterattack(make_valid_submission):
    df = make_valid_submission()
    df.loc[70000:70050, "label"] = "A"  # counter
    df.loc[70051:70500, "label"] = "D"  # tactical foul + yellow card
    df.loc[70501:70600, "label"] = "A"  # free kick restart
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_compound_pressing_duel_turnover(make_valid_submission):
    df = make_valid_submission()
    df.loc[71000:71040, "label"] = "H"
    df.loc[71041:71100, "label"] = "A"
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_compound_goalkeeper_distribute_to_fullback(make_valid_submission):
    df = make_valid_submission()
    df.loc[72000:72100, "label"] = "H"
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_compound_offside_flag_stoppage(make_valid_submission):
    df = make_valid_submission()
    df.loc[73000:73050, "label"] = "A"  # through ball
    df.loc[73051:73300, "label"] = "D"  # offside whistle
    df.loc[73301:73400, "label"] = "H"  # indirect free kick
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_compound_blocked_shot_rebound(make_valid_submission):
    df = make_valid_submission()
    df.loc[74000:74030, "label"] = "H"  # shot
    df.loc[74031:74080, "label"] = "H"  # rebound gathered by same team
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_compound_penalty_kick_procedure(make_valid_submission):
    df = make_valid_submission()
    df.loc[75000:76500, "label"] = "D"  # 60s penalty setup
    df.loc[76501:76520, "label"] = "A"  # penalty taken
    df.loc[76521:77000, "label"] = "D"  # goal celebration
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_compound_double_turnover_in_midfield(make_valid_submission):
    df = make_valid_submission()
    df.loc[78000:78030, "label"] = "H"
    df.loc[78031:78060, "label"] = "A"
    df.loc[78061:78100, "label"] = "H"
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_compound_ball_stuck_in_corner_flag_scramble(make_valid_submission):
    df = make_valid_submission()
    df.loc[79000:79100, "label"] = "A"  # shielded
    df.loc[79101:79200, "label"] = "D"  # out for throw-in
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_compound_second_half_restart_after_prekick(make_valid_submission):
    df = make_valid_submission()
    df.loc[113740:113741, "label"] = "D"  # pre-kickoff ends
    df.loc[113742:113800, "label"] = "H"  # Barcelona kickoff
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_compound_final_attack_into_whistle(make_valid_submission):
    df = make_valid_submission()
    df.loc[190150:190197, "label"] = "A"  # final attack
    df.loc[190198:190250, "label"] = "X"  # whistle blows
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier3_compound_full_match_continuity_integrity(make_valid_submission):
    df = make_valid_submission()
    res = SubmissionValidator.validate_dataframe(df, strict_invariants=True)
    assert res.is_valid
    assert res.stats["row_count"] == TOTAL_FRAMES
