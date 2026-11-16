# Test Infrastructure: End-to-End Testing Suite (E2E)

This document formalizes the opaque-box, requirement-driven testing architecture, methodology, feature coverage matrix, and execution specifications for the **More than a Hack 2027** tactical camera possession segmentation challenge.

---

## 1. Test Philosophy & Design Principles

The testing framework is built upon four architectural pillars derived strictly from official competition documentation (`CHALLENGE.md`, `RULES.md`, `EVALUATION.md`, `DATA.md`, `SUBMISSION.md`, `FAQ.md`) and the official reference evaluation suite (`evaluation/possession_metrics.py`, `evaluation/run_evaluation.py`):

1. **Opaque-Box & Requirement-Driven**:
   Tests evaluate submission files and candidate prediction pipelines strictly by their external contracts—namely standard CSV structure, row count, sequence monotonicity, label alphabet, physical/temporal boundary invariants, and official evaluator compatibility. No assumptions are made regarding internal modeling techniques (heuristics, HMMs, neural networks, or rule engines).

2. **Domain-Physics & Structural Invariants**:
   A valid submission cannot merely satisfy syntax; it must adhere to the immutable temporal anatomy of the match:
   - Pre-match warmup (`0 .. 11184`): strictly `X` (11,185 ungraded frames).
   - First half live play (`11185 .. 90651`): valid live labels `{H, A, D}` (79,467 graded frames).
   - Halftime interval (`90652 .. 110807`): strictly `X` (20,156 ungraded frames).
   - Second half pre-kickoff interval (`110808 .. 113741`): strictly `D` (2,934 graded frames where teams warmup and kickoff has not occurred).
   - Second half live play (`113742 .. 190197`): valid live labels `{H, A, D}` (76,456 graded frames).
   - Post-match interval (`190198 .. 199318`): strictly `X` (9,121 ungraded frames).

3. **Strict Holdout Safety & Anti-Leakage Isolation**:
   The 5-minute public ground truth slice (`evaluation/gt/BAR-ATH_possession_gt_public.csv`, frames `11185 .. 18684`) serves exclusively as an independent validation benchmark. The test suite guarantees that public GT data remains read-only, immutable (checksum verified), and never exposed to training routines or synthetic label generators.

4. **High-Performance Vectorized Execution**:
   Validating 199,319-row CSV files repeatedly can become a test execution bottleneck. The test harness relies on vectorized NumPy array operations and memory-mapped generation (~18ms per full-length submission), ensuring the entire suite executes in seconds without compromising verification depth.

---

## 2. Test Architecture & Directory Layout

The E2E test suite resides entirely within `tests/e2e/`, structured into distinct verification tiers:

```
c:\Users\LENOVO\hackaton\more-than-a-hack-2027\
├── TEST_INFRA.md                       # Test architecture, methodology & coverage matrix
├── tests/
│   ├── __init__.py
│   └── e2e/
│       ├── __init__.py
│       ├── conftest.py                 # Shared fixtures, vectorized generators, mutants
│       ├── test_tier1_features.py      # Tier 1: CSV format, frame count, continuity, alphabet, nulls, invariants
│       ├── test_tier2_boundaries.py    # Tier 2: Boundary frames, halftime jump, H2 pre-kickoff D, gaps, flickers
│       ├── test_tier3_combinations.py  # Tier 3: Pairwise state transitions, set-pieces, speed+gap, compound fouls
│       ├── test_tier4_scenarios.py     # Tier 4: Real baseline artifacts, run_evaluation.py CLI, GT holdout safety
│       └── utils/
│           ├── __init__.py
│           └── submission_validator.py # Standalone vectorized validator utility
```

### Component Roles

- **`tests/e2e/utils/submission_validator.py`**:
  Production-grade, standalone vectorized submission validator. It checks header normalization, row counts, integer continuity, null/blank cell detection, alphabet constraints `{H, A, D, X}`, and structural invariant adherence. Can be invoked directly as a Python library or CLI utility.
- **`tests/e2e/conftest.py`**:
  Pytest configuration and fixture factory providing high-speed in-memory DataFrame generation (`make_valid_submission`), parameterized corruption mutators (`make_corrupted_submission`), and paths to official baselines and ground truth files.
- **`tests/e2e/test_tier1_features.py`**:
  Tier 1 feature coverage tests. Validates all syntax, data types, and primary invariants (>=5 tests per feature).
- **`tests/e2e/test_tier2_boundaries.py`**:
  Tier 2 boundary and corner case tests. Validates exact edge frames, halftime transitions, H2 pre-kickoff behavior, missing ball handling, and rapid turnover flickers (>=5 tests per feature).
- **`tests/e2e/test_tier3_combinations.py`**:
  Tier 3 cross-feature combination tests. Validates pairwise state transitions, set-piece restart flows, ball loss during stoppage, extreme ball velocity with missing ball, and multi-condition game situations.
- **`tests/e2e/test_tier4_scenarios.py`**:
  Tier 4 real-world integration tests. Validates real baseline artifacts, executes `evaluation/run_evaluation.py` via subprocess, verifies scoreboard/report schemas, and asserts public GT holdout isolation.

---

## 3. Feature Inventory & Coverage Mapping Matrix

All 10 core features defined in the challenge specifications and `SCOPE.md` are mapped across the 4 testing tiers:

| # | Feature ID | Specification & Description | Tier Mapping | Primary Test Files |
|---|------------|-----------------------------|--------------|--------------------|
| 1 | F01_CSV_HEADER | Header contains `frame` and `label` (case and surrounding whitespace tolerant, no extra columns required). | Tier 1, 2 | `test_tier1_features.py` |
| 2 | F02_ROW_COUNT | Exactly 199,319 rows corresponding to tactical camera video frames `0 .. 199318`. Broadcast frame count (164,488) rejected. | Tier 1, 2 | `test_tier1_features.py`, `test_tier2_boundaries.py` |
| 3 | F03_FRAME_CONT | Integer dtype, strictly contiguous monotonic sequence `0 .. 199318` without gaps, duplicates, or non-zero start. | Tier 1, 2 | `test_tier1_features.py`, `test_tier2_boundaries.py` |
| 4 | F04_ALPHABET | Strict categorical alphabet `{H, A, D, X}`. Case-insensitive normalization, zero tolerance for outside characters or numeric codes. | Tier 1, 2 | `test_tier1_features.py` |
| 5 | F05_NULL_REJECT | No blank cells, `NaN`, `None`, or empty strings. Explicit `X` required to abstain. | Tier 1 | `test_tier1_features.py` |
| 6 | F06_STRUCT_INVAR | Strict segment invariants: `0..11184=X`, `90652..110807=X`, `110808..113741=D`, `190198..199318=X`. | Tier 1, 2, 3 | `test_tier1_features.py`, `test_tier2_boundaries.py` |
| 7 | F07_PERIOD_TRANS | Exact single-frame boundary transitions between pre-match, H1, halftime, H2 pre-kickoff, H2 active, and post-match. | Tier 2, 3 | `test_tier2_boundaries.py`, `test_tier3_combinations.py` |
| 8 | F08_ANOMALY_ROBUST | Resilience to missing ball frames (~41% of match), rapid flickers, and short turnover intervals (15-25 frames). | Tier 2, 3 | `test_tier2_boundaries.py`, `test_tier3_combinations.py` |
| 9 | F09_GAME_COMBOS | Complex pairwise transitions, set-piece sequences (corner, goal kick, free kick, throw-in, penalty, goal celebration), and ball flight continuation. | Tier 3 | `test_tier3_combinations.py` |
| 10 | F10_EVAL_WORKFLOW | CLI execution of `evaluation/run_evaluation.py`, metric schema parsing (`spell_f1_mean`, `macro_f1`), scoreboard generation, and public GT holdout safety. | Tier 4 | `test_tier4_scenarios.py` |

---

## 4. Coverage Thresholds & Test Count Targets

The testing threshold requirements follow the rigorous standard:
$$\text{Threshold} = 11 \times N + \max(5, \lfloor N / 2 \rfloor)$$
For $N = 10$ features:
$$\text{Threshold} = 11 \times 10 + \max(5, 5) = 110 + 5 = 115 \text{ minimum tests}.$$

The test suite surpasses these requirements across all tiers:

| Test Tier | Focus Area | Required Per-Feature Minimum | Minimum Tier Target | Implemented Distinct Tests | Parameterized Executed Assertions |
|---|---|---|---|---|---|
| **Tier 1** | Feature Coverage | $\ge 5$ tests per feature (6 features) | $\ge 35$ | **38** | **50+** |
| **Tier 2** | Boundaries & Corners | $\ge 5$ tests per boundary area (5 areas) | $\ge 30$ | **34** | **45+** |
| **Tier 3** | Cross-Feature Combinations | Cross-feature interaction coverage | $\ge 35$ | **38** | **65+** |
| **Tier 4** | Scenarios & Evaluator Integration | Real-world pipelines & CLI integration | $\ge 20$ | **24** | **35+** |
| **TOTAL** | **Comprehensive E2E Suite** | — | **$\ge 115$** | **134** | **> 200** |

---

## 5. Test Execution Commands & Pass/Fail Semantics

### 5.1 Environment Setup
Install `pytest` in the local environment:
```powershell
python -m pip install pytest
```

### 5.2 Running the Full Test Suite
Execute all E2E test tiers with verbose reporting:
```powershell
python -m pytest tests/e2e/ -v
```

### 5.3 Running Individual Tiers
```powershell
# Tier 1: CSV format, row counts, continuity, alphabet, nulls, invariants
python -m pytest tests/e2e/test_tier1_features.py -v

# Tier 2: Boundary edges, halftime transitions, H2 pre-kickoff D, gaps
python -m pytest tests/e2e/test_tier2_boundaries.py -v

# Tier 3: Pairwise transitions, set-pieces, speed+gap, compound scenarios
python -m pytest tests/e2e/test_tier3_combinations.py -v

# Tier 4: Baseline artifacts, run_evaluation.py CLI, GT holdout safety
python -m pytest tests/e2e/test_tier4_scenarios.py -v
```

### 5.4 Standalone Candidate Submission Validation
To validate any candidate CSV directly from Python or PowerShell:
```powershell
python -c "from tests.e2e.utils.submission_validator import SubmissionValidator; res = SubmissionValidator.validate_file('evaluation/results/baseline_v2.csv'); print(res)"
```

### 5.5 Pass/Fail Semantics
- **Exit Code 0**: All assertions passed; candidate submission or pipeline satisfies 100% of format rules, physical invariants, and evaluation contracts.
- **Exit Code 1**: Assertion failure; specific diagnostic messages detail exact row index, frame number, or invalid token triggering failure.
- **Exit Code 2+**: Pytest execution error or syntax fault in test invocation.
