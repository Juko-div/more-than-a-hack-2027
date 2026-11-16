"""tests/e2e/test_tier1_features.py
Tier 1: Feature Coverage Tests
Validates core specification contracts: CSV format, row counts, continuity, label alphabet,
null handling, and match structural invariants.
"""
import numpy as np
import pandas as pd
import pytest

from tests.e2e.utils.submission_validator import SubmissionValidator, TOTAL_FRAMES


# -----------------------------------------------------------------------------
# Feature 1: CSV Format & Header (6 tests)
# -----------------------------------------------------------------------------
def test_tier1_csv_valid_header(make_valid_submission):
    df = make_valid_submission()
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid, f"Valid dataframe failed validation: {res.errors}"


def test_tier1_csv_case_insensitive_header(make_valid_submission):
    df = make_valid_submission().rename(columns={"frame": "FRAME", "label": "LABEL"})
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid, "Case-insensitive headers FRAME, LABEL should be accepted"


def test_tier1_csv_whitespace_padded_header(make_valid_submission):
    df = make_valid_submission().rename(columns={"frame": "  frame  ", "label": " label "})
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid, "Whitespace-padded header columns should be accepted"


def test_tier1_csv_missing_frame_column_rejected(make_corrupted_submission):
    df = make_corrupted_submission("missing_frame_col")
    res = SubmissionValidator.validate_dataframe(df)
    assert not res.is_valid
    assert any("frame" in e.lower() for e in res.errors)


def test_tier1_csv_missing_label_column_rejected(make_corrupted_submission):
    df = make_corrupted_submission("missing_label_col")
    res = SubmissionValidator.validate_dataframe(df)
    assert not res.is_valid
    assert any("label" in e.lower() for e in res.errors)


def test_tier1_csv_reversed_column_order_accepted(make_valid_submission):
    df = make_valid_submission()[["label", "frame"]]
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid, "Reversed column order (label, frame) should be accepted if names match"


# -----------------------------------------------------------------------------
# Feature 2: Exact Row Count 199,319 (6 tests)
# -----------------------------------------------------------------------------
def test_tier1_row_count_exact(make_valid_submission):
    df = make_valid_submission()
    assert len(df) == TOTAL_FRAMES
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier1_row_count_short_rejected(make_corrupted_submission):
    df = make_corrupted_submission("row_count_short")
    res = SubmissionValidator.validate_dataframe(df)
    assert not res.is_valid
    assert any("199318" in e or "199319" in e for e in res.errors)


def test_tier1_row_count_long_rejected(make_corrupted_submission):
    df = make_corrupted_submission("row_count_long")
    res = SubmissionValidator.validate_dataframe(df)
    assert not res.is_valid
    assert any("199320" in e or "199319" in e for e in res.errors)


def test_tier1_row_count_empty_dataframe_rejected():
    df = pd.DataFrame({"frame": [], "label": []})
    res = SubmissionValidator.validate_dataframe(df)
    assert not res.is_valid
    assert any("0" in e and "199319" in e or "mismatch" in e.lower() for e in res.errors)


def test_tier1_row_count_single_row_rejected():
    df = pd.DataFrame({"frame": [0], "label": ["X"]})
    res = SubmissionValidator.validate_dataframe(df)
    assert not res.is_valid
    assert any("1" in e and "199319" in e or "mismatch" in e.lower() for e in res.errors)


def test_tier1_row_count_broadcast_length_rejected():
    # Broadcast camera has 164,488 frames -- common mistake noted in FAQ
    frames = np.arange(164488)
    df = pd.DataFrame({"frame": frames, "label": np.full(164488, "X")})
    res = SubmissionValidator.validate_dataframe(df)
    assert not res.is_valid
    assert any("164488" in e for e in res.errors)


# -----------------------------------------------------------------------------
# Feature 3: Integer Frame Continuity 0..199318 (6 tests)
# -----------------------------------------------------------------------------
def test_tier1_continuity_perfect_sequence(make_valid_submission):
    df = make_valid_submission()
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid


def test_tier1_continuity_float_type_rejected(make_corrupted_submission):
    df = make_corrupted_submission("float_frames")
    res = SubmissionValidator.validate_dataframe(df)
    assert not res.is_valid
    assert any("not integer" in e.lower() for e in res.errors)


def test_tier1_continuity_frame_gap_rejected(make_corrupted_submission):
    df = make_corrupted_submission("frame_gap")
    res = SubmissionValidator.validate_dataframe(df)
    assert not res.is_valid
    assert any("contiguous" in e.lower() or "in order" in e.lower() for e in res.errors)


def test_tier1_continuity_frame_duplicate_rejected(make_corrupted_submission):
    df = make_corrupted_submission("frame_duplicate")
    res = SubmissionValidator.validate_dataframe(df)
    assert not res.is_valid
    assert any("contiguous" in e.lower() or "in order" in e.lower() for e in res.errors)


def test_tier1_continuity_frame_shuffled_rejected(make_corrupted_submission):
    df = make_corrupted_submission("frame_unordered")
    res = SubmissionValidator.validate_dataframe(df)
    assert not res.is_valid
    assert any("in order" in e.lower() or "contiguous" in e.lower() for e in res.errors)


def test_tier1_continuity_one_indexed_start_rejected(make_valid_submission):
    df = make_valid_submission()
    df["frame"] = df["frame"] + 1  # 1..199319
    res = SubmissionValidator.validate_dataframe(df)
    assert not res.is_valid
    assert any("in order" in e.lower() or "contiguous" in e.lower() for e in res.errors)


# -----------------------------------------------------------------------------
# Feature 4: Valid Label Alphabet {H, A, D, X} (6 tests)
# -----------------------------------------------------------------------------
def test_tier1_alphabet_valid_distribution(make_valid_submission):
    df = make_valid_submission()
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid
    for lbl in ["H", "A", "D", "X"]:
        assert res.stats[f"count_{lbl}"] >= 0


def test_tier1_alphabet_invalid_single_character(make_corrupted_submission):
    df = make_corrupted_submission("invalid_label_z")
    res = SubmissionValidator.validate_dataframe(df)
    assert not res.is_valid
    assert any("Z" in e for e in res.errors)


def test_tier1_alphabet_invalid_multi_character(make_corrupted_submission):
    df = make_corrupted_submission("invalid_label_word")
    res = SubmissionValidator.validate_dataframe(df)
    assert not res.is_valid
    assert any("HOME" in e for e in res.errors)


def test_tier1_alphabet_lowercase_coerced_cleanly(make_valid_submission):
    df = make_valid_submission()
    df.loc[12000, "label"] = "h"
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid, f"Lowercase 'h' should be normalized cleanly to 'H', errors: {res.errors}"


def test_tier1_alphabet_surrounding_whitespace_trimmed(make_valid_submission):
    df = make_valid_submission()
    df.loc[12000, "label"] = "  H  "
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid, "Whitespace surrounding label should be stripped cleanly"


def test_tier1_alphabet_numeric_labels_rejected(make_valid_submission):
    df = make_valid_submission()
    df.loc[12000, "label"] = "0"
    res = SubmissionValidator.validate_dataframe(df)
    assert not res.is_valid
    assert any("0" in e for e in res.errors)


# -----------------------------------------------------------------------------
# Feature 5: Null / NaN / Empty Cell Checks (6 tests)
# -----------------------------------------------------------------------------
def test_tier1_null_np_nan_rejected(make_corrupted_submission):
    df = make_corrupted_submission("nan_label")
    res = SubmissionValidator.validate_dataframe(df)
    assert not res.is_valid
    assert any("null/nan/empty" in e.lower() for e in res.errors)


def test_tier1_null_empty_string_rejected(make_corrupted_submission):
    df = make_corrupted_submission("empty_label")
    res = SubmissionValidator.validate_dataframe(df)
    assert not res.is_valid
    assert any("blank/whitespace-only" in e.lower() or "empty" in e.lower() for e in res.errors)


def test_tier1_null_whitespace_only_rejected(make_corrupted_submission):
    df = make_corrupted_submission("whitespace_only_label")
    res = SubmissionValidator.validate_dataframe(df)
    assert not res.is_valid
    assert any("blank/whitespace-only" in e.lower() or "empty" in e.lower() for e in res.errors)


def test_tier1_null_none_rejected(make_valid_submission):
    df = make_valid_submission()
    df.loc[12000, "label"] = None
    res = SubmissionValidator.validate_dataframe(df)
    assert not res.is_valid
    assert any("null/nan/empty" in e.lower() for e in res.errors)


def test_tier1_null_multiple_scattered_nans(make_valid_submission):
    df = make_valid_submission()
    df.loc[[100, 50000, 120000], "label"] = np.nan
    res = SubmissionValidator.validate_dataframe(df)
    assert not res.is_valid
    assert any("3" in e for e in res.errors)


def test_tier1_null_zero_nan_passes(make_valid_submission):
    df = make_valid_submission()
    res = SubmissionValidator.validate_dataframe(df)
    assert res.is_valid
    assert not any("nan" in e.lower() for e in res.errors)


# -----------------------------------------------------------------------------
# Feature 6: Structural Invariant Zones (8 tests)
# -----------------------------------------------------------------------------
def test_tier1_invariant_pre_match_must_be_x(make_corrupted_submission):
    df = make_corrupted_submission("pre_match_invariant_broken")
    res = SubmissionValidator.validate_dataframe(df, strict_invariants=True)
    assert not res.is_valid
    assert any("pre-match invariant" in e.lower() for e in res.errors)


def test_tier1_invariant_halftime_must_be_x(make_corrupted_submission):
    df = make_corrupted_submission("halftime_invariant_broken")
    res = SubmissionValidator.validate_dataframe(df, strict_invariants=True)
    assert not res.is_valid
    assert any("halftime invariant" in e.lower() for e in res.errors)


def test_tier1_invariant_h2_prekick_must_be_d(make_corrupted_submission):
    df = make_corrupted_submission("h2_prekick_invariant_broken")
    res = SubmissionValidator.validate_dataframe(df, strict_invariants=True)
    assert not res.is_valid
    assert any("h2 pre-kickoff invariant" in e.lower() for e in res.errors)


def test_tier1_invariant_post_match_must_be_x(make_corrupted_submission):
    df = make_corrupted_submission("post_match_invariant_broken")
    res = SubmissionValidator.validate_dataframe(df, strict_invariants=True)
    assert not res.is_valid
    assert any("post-match invariant" in e.lower() for e in res.errors)


def test_tier1_invariant_valid_submission_passes_all(make_valid_submission):
    df = make_valid_submission()
    res = SubmissionValidator.validate_dataframe(df, strict_invariants=True)
    assert res.is_valid
    assert len(res.errors) == 0


def test_tier1_invariant_h1_span_allows_live_labels(make_valid_submission):
    df = make_valid_submission()
    assert set(df.loc[11185:90651, "label"].unique()).issubset({"H", "A", "D"})


def test_tier1_invariant_h2_live_span_allows_live_labels(make_valid_submission):
    df = make_valid_submission()
    assert set(df.loc[113742:190197, "label"].unique()).issubset({"H", "A", "D"})


def test_tier1_invariant_warning_mode_when_not_strict(make_corrupted_submission):
    df = make_corrupted_submission("pre_match_invariant_broken")
    res = SubmissionValidator.validate_dataframe(df, strict_invariants=False)
    assert res.is_valid  # passes in non-strict mode
    assert len(res.warnings) > 0  # but records warning
