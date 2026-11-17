"""tests/run_adversarial_suite.py
Adversarial Stress Harness and Empirical Mutation Verification for More than a Hack 2027.
Tests SubmissionValidator and possession_metrics against a comprehensive matrix of
subtly corrupt submissions, boundary edge cases, and genuine valid submissions.
"""
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

# Add repo root and evaluation to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from tests.e2e.utils.submission_validator import (
    SubmissionValidator,
    ValidationResult,
    TOTAL_FRAMES,
    PRE_MATCH_SPAN,
    H1_SPAN,
    HALFTIME_SPAN,
    H2_PREKICK_SPAN,
    H2_SPAN,
    POST_MATCH_SPAN,
)
import possession_metrics as pm


def generate_clean_submission() -> pd.DataFrame:
    """Build a completely valid 199,319-row submission DataFrame."""
    frames = np.arange(TOTAL_FRAMES, dtype=np.int64)
    labels = np.full(TOTAL_FRAMES, "X", dtype=object)

    # H1 Live (11185..90651)
    h1_len = H1_SPAN[1] - H1_SPAN[0] + 1
    pattern = np.array(["H"] * 80 + ["D"] * 20 + ["A"] * 80 + ["D"] * 20, dtype=object)
    tiled_h1 = np.tile(pattern, int(np.ceil(h1_len / len(pattern))))[:h1_len]
    labels[H1_SPAN[0]:H1_SPAN[1] + 1] = tiled_h1

    # H2 Pre-kickoff (110808..113741) -> strictly D
    labels[H2_PREKICK_SPAN[0]:H2_PREKICK_SPAN[1] + 1] = "D"

    # H2 Live (113742..190197)
    h2_len = H2_SPAN[1] - H2_SPAN[0] + 1
    tiled_h2 = np.tile(pattern, int(np.ceil(h2_len / len(pattern))))[:h2_len]
    labels[H2_SPAN[0]:H2_SPAN[1] + 1] = tiled_h2

    return pd.DataFrame({"frame": frames, "label": labels})


def build_mutation_suite() -> Dict[str, Tuple[pd.DataFrame, str, bool]]:
    """
    Returns a dict mapping mutation_id -> (DataFrame, expected_error_substring, is_valid_expected).
    is_valid_expected is False for mutations that MUST be rejected.
    """
    suite: Dict[str, Tuple[pd.DataFrame, str, bool]] = {}

    clean = generate_clean_submission()

    # =========================================================================
    # Group 1: Row Count & Length Mutations
    # =========================================================================
    # M_ROW_SHORT_1: exactly 199,318 rows (1 short)
    df_short1 = clean.iloc[:-1].copy()
    suite["M_ROW_SHORT_199318"] = (df_short1, "Row count mismatch: expected 199319, got 199318", False)

    # M_ROW_LONG_1: exactly 199,320 rows (1 too many)
    extra_row = pd.DataFrame({"frame": [TOTAL_FRAMES], "label": ["X"]})
    df_long1 = pd.concat([clean, extra_row], ignore_index=True)
    suite["M_ROW_LONG_199320"] = (df_long1, "Row count mismatch: expected 199319, got 199320", False)

    # M_ROW_BROADCAST: 164,488 rows
    df_bcast = pd.DataFrame({"frame": np.arange(164488, dtype=np.int64), "label": np.full(164488, "X", dtype=object)})
    suite["M_ROW_BROADCAST_164488"] = (df_bcast, "Row count mismatch: expected 199319, got 164488", False)

    # M_ROW_EMPTY: 0 rows
    df_empty = pd.DataFrame({"frame": np.array([], dtype=np.int64), "label": np.array([], dtype=object)})
    suite["M_ROW_EMPTY_0"] = (df_empty, "Row count mismatch: expected 199319, got 0", False)

    # M_ROW_SINGLE: 1 row
    df_single = pd.DataFrame({"frame": [0], "label": ["X"]})
    suite["M_ROW_SINGLE_1"] = (df_single, "Row count mismatch: expected 199319, got 1", False)

    # =========================================================================
    # Group 2: Frame Index, Types & Continuity Mutations
    # =========================================================================
    # M_FRAME_FLOAT: float dtype frames
    df_float = clean.copy()
    df_float["frame"] = df_float["frame"].astype(float)
    suite["M_FRAME_FLOAT"] = (df_float, "not integer", False)

    # M_FRAME_STRING: string/object dtype frames
    df_str_frame = clean.copy()
    df_str_frame["frame"] = df_str_frame["frame"].astype(object)
    suite["M_FRAME_STRING"] = (df_str_frame, "not integer", False)

    # M_FRAME_GAP_START: starts at 1 instead of 0 (1..199319)
    df_start1 = clean.copy()
    df_start1["frame"] = df_start1["frame"] + 1
    suite["M_FRAME_GAP_START_1"] = (df_start1, "not contiguous 0..199318 in order", False)

    # M_FRAME_GAP_MID: skips frame 50000 (frame 50000 -> 50001, row 199318 dropped)
    df_gap_mid = clean.copy()
    df_gap_mid.loc[50000:, "frame"] += 1
    df_gap_mid = df_gap_mid.iloc[:-1].copy()
    suite["M_FRAME_GAP_MID"] = (df_gap_mid, "Row count mismatch", False)

    # M_FRAME_GAP_SAME_LEN: jump at 50000 but same length (50000 missing, 199319 extra)
    df_jump = clean.copy()
    df_jump.loc[50000:, "frame"] += 1
    suite["M_FRAME_GAP_SAME_LEN"] = (df_jump, "not contiguous 0..199318 in order", False)

    # M_FRAME_DUPLICATE: duplicate frame 50000
    df_dup = clean.copy()
    df_dup.loc[50001, "frame"] = 50000
    suite["M_FRAME_DUPLICATE_MID"] = (df_dup, "not contiguous 0..199318 in order", False)

    # M_FRAME_DUPLICATE_START: duplicate frame 0
    df_dup0 = clean.copy()
    df_dup0.loc[1, "frame"] = 0
    suite["M_FRAME_DUPLICATE_START"] = (df_dup0, "not contiguous 0..199318 in order", False)

    # M_FRAME_REVERSED: completely inverted order
    df_rev = clean.iloc[::-1].reset_index(drop=True).copy()
    suite["M_FRAME_REVERSED"] = (df_rev, "not contiguous 0..199318 in order", False)

    # M_FRAME_SWAP_2: swap frame 50000 and 50001
    df_swap = clean.copy()
    df_swap.loc[50000, "frame"] = 50001
    df_swap.loc[50001, "frame"] = 50000
    suite["M_FRAME_SWAP_ADJACENT"] = (df_swap, "not contiguous 0..199318 in order", False)

    # M_FRAME_NEGATIVE: frame -1 at row 0
    df_neg = clean.copy()
    df_neg.loc[0, "frame"] = -1
    suite["M_FRAME_NEGATIVE"] = (df_neg, "not contiguous 0..199318 in order", False)

    # =========================================================================
    # Group 3: Label Alphabet & Corruption Mutations
    # =========================================================================
    # M_LABEL_Z: invalid label 'Z'
    df_z = clean.copy()
    df_z.loc[25000, "label"] = "Z"
    suite["M_LABEL_INVALID_Z"] = (df_z, "Invalid labels outside {'H', 'A', 'D', 'X'}", False)

    # M_LABEL_WORD: invalid label 'HOME'
    df_word = clean.copy()
    df_word.loc[25000, "label"] = "HOME"
    suite["M_LABEL_INVALID_WORD"] = (df_word, "Invalid labels outside {'H', 'A', 'D', 'X'}", False)

    # M_LABEL_NUMERIC_0: int 0
    df_num0 = clean.copy()
    df_num0["label"] = df_num0["label"].astype(object)
    df_num0.loc[25000, "label"] = 0
    suite["M_LABEL_NUMERIC_0"] = (df_num0, "Invalid labels outside {'H', 'A', 'D', 'X'}", False)

    # M_LABEL_NUMERIC_1: int 1
    df_num1 = clean.copy()
    df_num1["label"] = df_num1["label"].astype(object)
    df_num1.loc[25000, "label"] = 1
    suite["M_LABEL_NUMERIC_1"] = (df_num1, "Invalid labels outside {'H', 'A', 'D', 'X'}", False)

    # M_LABEL_BOOL: boolean True
    df_bool = clean.copy()
    df_bool["label"] = df_bool["label"].astype(object)
    df_bool.loc[25000, "label"] = True
    suite["M_LABEL_BOOL"] = (df_bool, "Invalid labels outside {'H', 'A', 'D', 'X'}", False)

    # M_LABEL_SPECIAL: '@'
    df_spec = clean.copy()
    df_spec.loc[25000, "label"] = "@"
    suite["M_LABEL_SPECIAL_AT"] = (df_spec, "Invalid labels outside {'H', 'A', 'D', 'X'}", False)

    # M_LABEL_NAN_NP: np.nan
    df_nan = clean.copy()
    df_nan.loc[25000, "label"] = np.nan
    suite["M_LABEL_NAN_NP"] = (df_nan, "null/NaN/empty label entries", False)

    # M_LABEL_NONE: None
    df_none = clean.copy()
    df_none.loc[25000, "label"] = None
    suite["M_LABEL_NONE"] = (df_none, "null/NaN/empty label entries", False)

    # M_LABEL_EMPTY_STR: empty string ""
    df_empty_str = clean.copy()
    df_empty_str.loc[25000, "label"] = ""
    suite["M_LABEL_EMPTY_STRING"] = (df_empty_str, "blank/whitespace-only label entries", False)

    # M_LABEL_SPACES: "   "
    df_spaces = clean.copy()
    df_spaces.loc[25000, "label"] = "   "
    suite["M_LABEL_WHITESPACE_SPACES"] = (df_spaces, "blank/whitespace-only label entries", False)

    # M_LABEL_TAB: "\t"
    df_tab = clean.copy()
    df_tab.loc[25000, "label"] = "\t"
    suite["M_LABEL_WHITESPACE_TAB"] = (df_tab, "blank/whitespace-only label entries", False)

    # M_LABEL_NEWLINE: "\n"
    df_nl = clean.copy()
    df_nl.loc[25000, "label"] = "\n"
    suite["M_LABEL_WHITESPACE_NEWLINE"] = (df_nl, "blank/whitespace-only label entries", False)

    # =========================================================================
    # Group 4: Structural Invariant Mutations (Boundary Frames & Intervals)
    # =========================================================================
    # Pre-match: 0..11184 == X
    # Boundary 0: frame 0 is 'H'
    df_pre_0 = clean.copy()
    df_pre_0.loc[0, "label"] = "H"
    suite["M_INVAR_PRE_FRAME_0"] = (df_pre_0, "Pre-match invariant violated", False)

    # Boundary 11184: frame 11184 is 'D'
    df_pre_11184 = clean.copy()
    df_pre_11184.loc[11184, "label"] = "D"
    suite["M_INVAR_PRE_FRAME_11184"] = (df_pre_11184, "Pre-match invariant violated", False)

    # Interior pre-match: frame 5000 is 'A'
    df_pre_5000 = clean.copy()
    df_pre_5000.loc[5000, "label"] = "A"
    suite["M_INVAR_PRE_FRAME_5000"] = (df_pre_5000, "Pre-match invariant violated", False)

    # Halftime: 90652..110807 == X
    # Boundary 90652: frame 90652 is 'H'
    df_ht_90652 = clean.copy()
    df_ht_90652.loc[90652, "label"] = "H"
    suite["M_INVAR_HT_FRAME_90652"] = (df_ht_90652, "Halftime invariant violated", False)

    # Boundary 110807: frame 110807 is 'A'
    df_ht_110807 = clean.copy()
    df_ht_110807.loc[110807, "label"] = "A"
    suite["M_INVAR_HT_FRAME_110807"] = (df_ht_110807, "Halftime invariant violated", False)

    # Interior halftime: frame 100000 is 'D'
    df_ht_100000 = clean.copy()
    df_ht_100000.loc[100000, "label"] = "D"
    suite["M_INVAR_HT_FRAME_100000"] = (df_ht_100000, "Halftime invariant violated", False)

    # H2 Pre-kickoff: 110808..113741 == D
    # Boundary 110808: frame 110808 is 'X'
    df_h2pk_110808_x = clean.copy()
    df_h2pk_110808_x.loc[110808, "label"] = "X"
    suite["M_INVAR_H2PK_FRAME_110808_X"] = (df_h2pk_110808_x, "H2 pre-kickoff invariant violated", False)

    # Boundary 110808: frame 110808 is 'H'
    df_h2pk_110808_h = clean.copy()
    df_h2pk_110808_h.loc[110808, "label"] = "H"
    suite["M_INVAR_H2PK_FRAME_110808_H"] = (df_h2pk_110808_h, "H2 pre-kickoff invariant violated", False)

    # Boundary 113741: frame 113741 is 'X'
    df_h2pk_113741_x = clean.copy()
    df_h2pk_113741_x.loc[113741, "label"] = "X"
    suite["M_INVAR_H2PK_FRAME_113741_X"] = (df_h2pk_113741_x, "H2 pre-kickoff invariant violated", False)

    # Boundary 113741: frame 113741 is 'A'
    df_h2pk_113741_a = clean.copy()
    df_h2pk_113741_a.loc[113741, "label"] = "A"
    suite["M_INVAR_H2PK_FRAME_113741_A"] = (df_h2pk_113741_a, "H2 pre-kickoff invariant violated", False)

    # Interior H2 pre-kickoff: frame 112000 is 'H'
    df_h2pk_112000 = clean.copy()
    df_h2pk_112000.loc[112000, "label"] = "H"
    suite["M_INVAR_H2PK_FRAME_112000"] = (df_h2pk_112000, "H2 pre-kickoff invariant violated", False)

    # Post-match: 190198..199318 == X
    # Boundary 190198: frame 190198 is 'D'
    df_post_190198 = clean.copy()
    df_post_190198.loc[190198, "label"] = "D"
    suite["M_INVAR_POST_FRAME_190198"] = (df_post_190198, "Post-match invariant violated", False)

    # Boundary 199318: frame 199318 is 'H'
    df_post_199318 = clean.copy()
    df_post_199318.loc[199318, "label"] = "H"
    suite["M_INVAR_POST_FRAME_199318"] = (df_post_199318, "Post-match invariant violated", False)

    # Interior post-match: frame 195000 is 'A'
    df_post_195000 = clean.copy()
    df_post_195000.loc[195000, "label"] = "A"
    suite["M_INVAR_POST_FRAME_195000"] = (df_post_195000, "Post-match invariant violated", False)

    # =========================================================================
    # Group 5: Edge Normalizations & Legitimate Variations (Must Pass!)
    # =========================================================================
    # V_CLEAN: Pristine valid submission
    suite["V_CLEAN_BASELINE"] = (clean.copy(), "", True)

    # V_LOWERCASE: Lowercase labels 'h', 'a', 'd', 'x' (case-insensitive)
    df_lower = clean.copy()
    df_lower["label"] = df_lower["label"].str.lower()
    suite["V_LOWERCASE_NORMALIZED"] = (df_lower, "", True)

    # V_WHITESPACE_PADDED_LABELS: Surrounding whitespace '  H  '
    df_pad = clean.copy()
    df_pad.loc[15000:15050, "label"] = "  H  "
    suite["V_WHITESPACE_PADDED_LABELS"] = (df_pad, "", True)

    # V_REVERSED_COLS: ['label', 'frame']
    df_rev_cols = clean[["label", "frame"]].copy()
    suite["V_REVERSED_COLUMNS"] = (df_rev_cols, "", True)

    # V_UPPERCASE_HEADER: ['FRAME', 'LABEL']
    df_up_cols = clean.rename(columns={"frame": "FRAME", "label": "LABEL"})
    suite["V_UPPERCASE_HEADER"] = (df_up_cols, "", True)

    # V_PADDED_HEADER: [' frame ', ' label ']
    df_pad_cols = clean.rename(columns={"frame": " frame ", "label": " label "})
    suite["V_PADDED_HEADER"] = (df_pad_cols, "", True)

    # V_EXTRA_COLUMNS: ['frame', 'label', 'confidence']
    df_extra_col = clean.copy()
    df_extra_col["confidence"] = 0.95
    suite["V_EXTRA_COLUMNS"] = (df_extra_col, "", True)

    # V_ABSTENTION_IN_H1_LIVE: Valid X during severe occlusion in H1 live
    df_live_x = clean.copy()
    df_live_x.loc[20000:20020, "label"] = "X"
    suite["V_ABSTENTION_IN_H1_LIVE"] = (df_live_x, "", True)

    # =========================================================================
    # Group 6: Constant Guesses (Must Violate Invariants)
    # =========================================================================
    # M_ALL_X: entire match is X (violates H2 pre-kickoff which must be D)
    df_all_x = pd.DataFrame({"frame": np.arange(TOTAL_FRAMES, dtype=np.int64), "label": np.full(TOTAL_FRAMES, "X", dtype=object)})
    suite["M_ALL_X_CONSTANT"] = (df_all_x, "H2 pre-kickoff invariant violated", False)

    # M_ALL_D: entire match is D (violates pre, ht, post invariants)
    df_all_d = pd.DataFrame({"frame": np.arange(TOTAL_FRAMES, dtype=np.int64), "label": np.full(TOTAL_FRAMES, "D", dtype=object)})
    suite["M_ALL_D_CONSTANT"] = (df_all_d, "Pre-match invariant violated", False)

    # M_ALL_H: entire match is H (violates all 4 invariants)
    df_all_h = pd.DataFrame({"frame": np.arange(TOTAL_FRAMES, dtype=np.int64), "label": np.full(TOTAL_FRAMES, "H", dtype=object)})
    suite["M_ALL_H_CONSTANT"] = (df_all_h, "Pre-match invariant violated", False)

    return suite


def run_adversarial_suite():
    print("=" * 80)
    print("STARTING ADVERSARIAL STRESS TEST & MUTATION BENCHMARK")
    print("=" * 80)

    suite = build_mutation_suite()
    total_mutants = len(suite)
    mutations_tested = 0
    mutations_caught = 0
    valid_tested = 0
    valid_passed = 0
    unexpected_failures = []
    unexpected_passes = []

    print(f"Generated {total_mutants} distinct test scenarios:")
    n_corrupt = sum(1 for _, _, exp in suite.values() if not exp)
    n_valid = sum(1 for _, _, exp in suite.values() if exp)
    print(f"  - Corrupt / Mutant Submissions (Must Fail): {n_corrupt}")
    print(f"  - Valid / Normalized Submissions (Must Pass): {n_valid}\n")

    # Benchmarking latency and memory
    latencies = []
    tracemalloc.start()
    t_start_all = time.perf_counter()

    crashes = []
    for mut_id, (df, expected_err, expected_valid) in suite.items():
        t0 = time.perf_counter()
        try:
            res = SubmissionValidator.validate_dataframe(df, strict_invariants=True)
            t_elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(t_elapsed_ms)

            if expected_valid:
                valid_tested += 1
                if res.is_valid:
                    valid_passed += 1
                    status = "PASS (Valid)"
                else:
                    unexpected_failures.append((mut_id, res.errors))
                    status = f"FAIL (Unexpected rejection: {res.errors[:2]})"
            else:
                mutations_tested += 1
                if not res.is_valid:
                    has_expected = any(expected_err.lower() in err.lower() for err in res.errors) if expected_err else True
                    mutations_caught += 1
                    status = f"CAUGHT ({res.errors[0][:60]}...)"
                else:
                    unexpected_passes.append((mut_id, expected_err))
                    status = f"MISSED (Mutant falsely accepted as valid!)"
        except Exception as exc:
            t_elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(t_elapsed_ms)
            crashes.append((mut_id, f"{type(exc).__name__}: {exc}"))
            if not expected_valid:
                mutations_tested += 1
            else:
                valid_tested += 1
            status = f"CRASH ({type(exc).__name__}: {str(exc)[:40]})"

        print(f"  [{status[:25]:25}] {mut_id:<32} ({t_elapsed_ms:5.1f} ms)")

    t_total_s = time.perf_counter() - t_start_all
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Also test actual files on disk:
    print("\n" + "-" * 80)
    print("TESTING REAL ARTIFACTS ON DISK")
    print("-" * 80)
    real_files = [
        ("baseline_v4.csv", REPO_ROOT / "evaluation" / "results" / "baseline_v4.csv", True),
        ("baseline_v2.csv", REPO_ROOT / "evaluation" / "results" / "baseline_v2.csv", True),
        ("baseline_smooth50.csv", REPO_ROOT / "evaluation" / "results" / "baseline_smooth50.csv", False), # non-strict invariants
        ("baseline_raw.csv", REPO_ROOT / "evaluation" / "results" / "baseline_raw.csv", False),
    ]

    for fname, fpath, strict in real_files:
        if fpath.is_file():
            t0 = time.perf_counter()
            res = SubmissionValidator.validate_file(fpath, strict_invariants=strict)
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"  {fname:<25} (strict={strict:<5}): valid={res.is_valid:<5} time={dt:5.1f} ms errors={len(res.errors)} warnings={len(res.warnings)}")
            if strict and not res.is_valid:
                print(f"    Errors: {res.errors}")
        else:
            print(f"  {fname:<25}: FILE NOT FOUND at {fpath}")

    # Summary
    print("\n" + "=" * 80)
    print("ADVERSARIAL STRESS TEST SUMMARY & VERDICT")
    print("=" * 80)
    print(f"Total Scenarios Evaluated       : {total_mutants}")
    print(f"Mutations Tested (Corrupt)      : {mutations_tested}")
    print(f"Mutations Caught                : {mutations_caught} / {mutations_tested} ({mutations_caught / max(1, mutations_tested) * 100:.1f}%)")
    print(f"Valid Scenarios Tested          : {valid_tested}")
    print(f"Valid Scenarios Passed          : {valid_passed} / {valid_tested} ({valid_passed / max(1, valid_tested) * 100:.1f}%)")
    print(f"Crashes / Unhandled Exceptions  : {len(crashes)}")
    print(f"False Positives (Valid Rejected): {len(unexpected_failures)}")
    print(f"False Negatives (Mutant Passed) : {len(unexpected_passes)}")
    print(f"Total Execution Time            : {t_total_s:.2f} s")
    print(f"Mean Validation Latency         : {np.mean(latencies):.2f} ms per 199,319 rows")
    print(f"Median Validation Latency       : {np.median(latencies):.2f} ms")
    print(f"Max Validation Latency          : {np.max(latencies):.2f} ms")
    print(f"Peak Memory Traced              : {peak_mem / (1024 * 1024):.2f} MB")
    print("=" * 80)

    if crashes:
        print("\n[!] CRASHES / UNHANDLED EXCEPTIONS:")
        for mut_id, exc_str in crashes:
            print(f"  - {mut_id}: {exc_str}")

    if unexpected_failures:
        print("\n[!] UNEXPECTED FAILURES (False Positives):")
        for mut_id, errs in unexpected_failures:
            print(f"  - {mut_id}: {errs}")

    if unexpected_passes:
        print("\n[!] UNEXPECTED PASSES (False Negatives - BUGS DETECTED):")
        for mut_id, exp in unexpected_passes:
            print(f"  - {mut_id}: Expected rejection with '{exp}' but test PASSED!")

    success = (len(unexpected_failures) == 0) and (len(unexpected_passes) == 0) and (len(crashes) == 0)
    print(f"\nFinal Verdict: {'PASS - ALL MUTATIONS CAUGHT' if success else 'FAIL - DEFECTS FOUND'}")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(run_adversarial_suite())
