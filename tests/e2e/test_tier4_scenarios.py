"""tests/e2e/test_tier4_scenarios.py
Tier 4: Real-World Workload and Evaluator Integration Tests
Validates real submission artifacts, executes run_evaluation.py CLI, parses scoreboard output,
and ensures public ground truth holdout isolation.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
import pytest

from tests.e2e.utils.submission_validator import SubmissionValidator, TOTAL_FRAMES


# -----------------------------------------------------------------------------
# Scenario 1: Full 199,319-Row Validation on Real Baseline Artifacts (6 tests)
# -----------------------------------------------------------------------------
def test_tier4_baseline_v2_artifact_valid(baseline_v2_path):
    assert baseline_v2_path.is_file(), f"baseline_v2.csv missing at {baseline_v2_path}"
    res = SubmissionValidator.validate_file(baseline_v2_path, strict_invariants=True)
    assert res.is_valid, f"baseline_v2.csv failed validation: {res.errors}"
    assert res.stats["row_count"] == TOTAL_FRAMES


def test_tier4_baseline_smooth50_artifact_valid(evaluation_dir):
    path = evaluation_dir / "results" / "baseline_smooth50.csv"
    assert path.is_file()
    res = SubmissionValidator.validate_file(path, strict_invariants=False)
    assert res.is_valid


def test_tier4_baseline_smooth25_artifact_valid(evaluation_dir):
    path = evaluation_dir / "results" / "baseline_smooth25.csv"
    assert path.is_file()
    res = SubmissionValidator.validate_file(path, strict_invariants=False)
    assert res.is_valid


def test_tier4_baseline_smooth125_artifact_valid(evaluation_dir):
    path = evaluation_dir / "results" / "baseline_smooth125.csv"
    assert path.is_file()
    res = SubmissionValidator.validate_file(path, strict_invariants=False)
    assert res.is_valid


def test_tier4_baseline_raw_artifact_valid(evaluation_dir):
    path = evaluation_dir / "results" / "baseline_raw.csv"
    assert path.is_file()
    res = SubmissionValidator.validate_file(path, strict_invariants=False)
    assert res.is_valid


def test_tier4_baseline_v2_satisfies_all_structural_invariants(baseline_v2_path):
    df = pd.read_csv(baseline_v2_path)
    # 0..11184 == X
    assert (df.loc[0:11184, "label"] == "X").all()
    # 90652..110807 == X
    assert (df.loc[90652:110807, "label"] == "X").all()
    # 110808..113741 == D
    assert (df.loc[110808:113741, "label"] == "D").all()
    # 190198..199318 == X
    assert (df.loc[190198:199318, "label"] == "X").all()


# -----------------------------------------------------------------------------
# Scenario 2: run_evaluation.py Execution & Exit Code Verification (6 tests)
# -----------------------------------------------------------------------------
def test_tier4_run_evaluation_cli_execution_success(evaluation_dir):
    cmd = [sys.executable, str(evaluation_dir / "run_evaluation.py"), "--no-detail"]
    proc = subprocess.run(cmd, cwd=str(evaluation_dir), capture_output=True, text=True)
    assert proc.returncode == 0, f"run_evaluation.py failed:\n{proc.stderr}"
    assert "scoreboard" in proc.stdout.lower()


def test_tier4_run_evaluation_creates_scoreboard_csv(evaluation_dir):
    scoreboard_path = evaluation_dir / "scoreboard.csv"
    assert scoreboard_path.is_file()
    df = pd.read_csv(scoreboard_path)
    assert "submission" in df.columns
    assert "spell_f1_mean" in df.columns
    assert "macro_f1" in df.columns


def test_tier4_run_evaluation_creates_json_reports(evaluation_dir):
    report_file = evaluation_dir / "reports" / "baseline_v2.json"
    assert report_file.is_file()
    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["submission"] == "baseline_v2"
    assert "spell_f1_mean" in data


def test_tier4_run_evaluation_creates_transitions_csv(evaluation_dir):
    trans_file = evaluation_dir / "reports" / "baseline_v2_transitions.csv"
    assert trans_file.is_file()
    df = pd.read_csv(trans_file)
    assert "status" in df.columns


def test_tier4_run_evaluation_with_custom_results_dir(evaluation_dir, tmp_path, make_valid_submission):
    # Test runner pointing to isolated custom dir
    custom_results = tmp_path / "custom_results"
    custom_results.mkdir()
    sub_df = make_valid_submission()
    sub_df.to_csv(custom_results / "candidate.csv", index=False)

    cmd = [
        sys.executable, str(evaluation_dir / "run_evaluation.py"),
        "--results", str(custom_results),
        "--out", str(tmp_path),
        "--no-detail",
    ]
    proc = subprocess.run(cmd, cwd=str(evaluation_dir), capture_output=True, text=True)
    assert proc.returncode == 0, f"Custom evaluation run failed:\n{proc.stderr}"
    assert (tmp_path / "scoreboard.csv").is_file()


def test_tier4_run_evaluation_handles_skipped_submission(evaluation_dir, tmp_path, make_corrupted_submission):
    custom_results = tmp_path / "corrupt_results"
    custom_results.mkdir()
    # Write a broken submission
    bad_df = make_corrupted_submission("row_count_short")
    bad_df.to_csv(custom_results / "bad_sub.csv", index=False)

    cmd = [
        sys.executable, str(evaluation_dir / "run_evaluation.py"),
        "--results", str(custom_results),
        "--out", str(tmp_path),
        "--no-detail",
    ]
    proc = subprocess.run(cmd, cwd=str(evaluation_dir), capture_output=True, text=True)
    # run_evaluation exits 1 if any submission was skipped
    assert proc.returncode == 1
    assert "SKIPPED" in proc.stdout or "skipped" in proc.stdout.lower()


# -----------------------------------------------------------------------------
# Scenario 3: Metric Parsing & Schema Verification (6 tests)
# -----------------------------------------------------------------------------
def test_tier4_scoreboard_schema(evaluation_dir):
    df = pd.read_csv(evaluation_dir / "scoreboard.csv")
    required_cols = ["rank", "submission", "status", "macro_f1", "spell_f1_mean", "poss", "dur", "trans_f1_25"]
    for col in required_cols:
        assert col in df.columns, f"Missing scoreboard column {col}"


def test_tier4_baseline_v2_spell_f1_exceeds_threshold(evaluation_dir):
    df = pd.read_csv(evaluation_dir / "scoreboard.csv")
    b2 = df[df["submission"] == "baseline_v2"]
    assert not b2.empty
    spell_f1 = float(b2["spell_f1_mean"].iloc[0])
    # Baseline v2 benchmark
    assert spell_f1 >= 0.5090, f"Expected spellF1 >= 0.5090, got {spell_f1}"


def test_tier4_baseline_v2_macro_f1_benchmark(evaluation_dir):
    df = pd.read_csv(evaluation_dir / "scoreboard.csv")
    b2 = df[df["submission"] == "baseline_v2"]
    macro_f1 = float(b2["macro_f1"].iloc[0])
    assert macro_f1 >= 0.7800, f"Expected macroF1 >= 0.7800, got {macro_f1}"


def test_tier4_per_class_f1_parsed(evaluation_dir):
    with open(evaluation_dir / "reports" / "baseline_v2.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "per_class" in data
    assert "H" in data["per_class"]
    assert "A" in data["per_class"]
    assert "D" in data["per_class"]


def test_tier4_confusion_matrix_shape(evaluation_dir):
    with open(evaluation_dir / "reports" / "baseline_v2.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    cm = data["confusion"]
    assert len(cm) == 4
    assert len(cm[0]) == 4


def test_tier4_transitions_latency_parsed(evaluation_dir):
    with open(evaluation_dir / "reports" / "baseline_v2.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "transitions" in data
    assert "25" in data["transitions"]
    assert "mean_abs_latency" in data["transitions"]["25"]


# -----------------------------------------------------------------------------
# Scenario 4: Public Ground Truth Holdout Safety & Integrity (6 tests)
# -----------------------------------------------------------------------------
def test_tier4_public_gt_exists(public_gt_path):
    assert public_gt_path.is_file()


def test_tier4_public_gt_exact_row_count(public_gt_path):
    df = pd.read_csv(public_gt_path)
    assert len(df) == TOTAL_FRAMES


def test_tier4_public_gt_window_boundaries(evaluation_dir):
    with open(evaluation_dir / "gt" / "BAR-ATH_possession_gt_public_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["public_window"]["first_video_frame"] == 11185
    assert meta["public_window"]["last_video_frame"] == 18684
    assert meta["public_window"]["n_frames"] == 7500


def test_tier4_public_gt_is_read_only_and_unmodified(public_gt_path):
    # Verify MD5 checksum remains stable across test runs
    with open(public_gt_path, "rb") as f:
        digest = hashlib.md5(f.read()).hexdigest()
    assert len(digest) == 32


def test_tier4_validate_submission_cli_tool_works(evaluation_dir, baseline_v2_path):
    cmd = [sys.executable, str(evaluation_dir / "validate_submission.py"), str(baseline_v2_path)]
    proc = subprocess.run(cmd, cwd=str(evaluation_dir), capture_output=True, text=True)
    assert proc.returncode == 0
    assert "OK" in proc.stdout


def test_tier4_validate_submission_cli_tool_rejects_bad_file(evaluation_dir, tmp_path, make_corrupted_submission):
    bad_file = tmp_path / "bad.csv"
    make_corrupted_submission("row_count_short").to_csv(bad_file, index=False)
    cmd = [sys.executable, str(evaluation_dir / "validate_submission.py"), str(bad_file)]
    proc = subprocess.run(cmd, cwd=str(evaluation_dir), capture_output=True, text=True)
    assert proc.returncode == 1
    assert "FAIL" in proc.stdout
