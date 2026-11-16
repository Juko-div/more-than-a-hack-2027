"""tests/e2e/utils/submission_validator.py
High-performance standalone submission validator for More than a Hack 2027.
Validates candidate CSV files or DataFrames against all official rules and structural invariants.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import numpy as np
import pandas as pd

ALLOWED_LABELS = frozenset({"H", "A", "D", "X"})
TOTAL_FRAMES = 199319

# Structural Invariant Zones
PRE_MATCH_SPAN = (0, 11184)        # Must be X (11,185 frames)
H1_SPAN = (11185, 90651)           # Graded {H, A, D} (79,467 frames)
HALFTIME_SPAN = (90652, 110807)    # Must be X (20,156 frames)
H2_PREKICK_SPAN = (110808, 113741) # Must be D (2,934 frames)
H2_SPAN = (113742, 190197)         # Graded {H, A, D} (76,456 frames)
POST_MATCH_SPAN = (190198, 199318) # Must be X (9,121 frames)


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.is_valid


class SubmissionValidator:
    """Vectorized validator enforcing official challenge constraints and structural invariants."""

    @classmethod
    def validate_dataframe(
        cls,
        df: pd.DataFrame,
        expected_rows: int = TOTAL_FRAMES,
        strict_invariants: bool = True,
    ) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []
        stats: Dict[str, Any] = {}

        # 1. Header validation
        cols = [str(c).strip().lower() for c in df.columns]
        if "frame" not in cols or "label" not in cols:
            errors.append(f"Missing required columns 'frame' and 'label'. Found: {list(df.columns)}")
            return ValidationResult(is_valid=False, errors=errors)

        frame_col = df.iloc[:, cols.index("frame")]
        label_col = df.iloc[:, cols.index("label")]

        # 2. Row count validation
        actual_rows = len(df)
        stats["row_count"] = actual_rows
        if actual_rows != expected_rows:
            errors.append(f"Row count mismatch: expected {expected_rows}, got {actual_rows}")

        # 3. Frame column integer continuity
        if not np.issubdtype(frame_col.dtype, np.integer):
            errors.append(f"'frame' column is not integer (dtype: {frame_col.dtype})")
        else:
            frames = frame_col.to_numpy()
            if actual_rows == expected_rows:
                if not np.array_equal(frames, np.arange(expected_rows)):
                    errors.append(f"'frame' column is not contiguous 0..{expected_rows - 1} in order")
            elif actual_rows > 0:
                if not np.array_equal(frames, np.arange(actual_rows)):
                    errors.append(f"'frame' column is not contiguous 0..{actual_rows - 1} in order")

        # 4. Null / NaN validation
        raw_labels = label_col.to_numpy()
        nan_mask = pd.isna(raw_labels)
        nan_count = int(nan_mask.sum())
        if nan_count > 0:
            errors.append(f"Found {nan_count} null/NaN/empty label entries. Write 'X' to abstain.")

        # Empty string / whitespace check
        str_empty_mask = np.array(
            [isinstance(v, str) and str(v).strip() == "" for v in raw_labels],
            dtype=bool
        )
        empty_count = int(str_empty_mask.sum())
        if empty_count > 0:
            errors.append(f"Found {empty_count} blank/whitespace-only label entries. Write 'X' to abstain.")

        # 5. Label alphabet validation
        valid_mask = ~nan_mask & ~str_empty_mask
        cleaned_labels = np.array([str(v).strip().upper() for v in raw_labels[valid_mask]], dtype=object)
        unique_labels = set(cleaned_labels)
        disallowed = sorted(unique_labels - ALLOWED_LABELS)
        if disallowed:
            errors.append(f"Invalid labels outside {{H, A, D, X}}: {disallowed[:10]}")

        # Compute label distribution stats
        for lbl in ["H", "A", "D", "X"]:
            stats[f"count_{lbl}"] = int(np.sum(cleaned_labels == lbl)) if len(cleaned_labels) else 0

        # 6. Structural Invariant Checks
        if actual_rows == expected_rows and not errors:
            full_cleaned = np.array([str(v).strip().upper() for v in raw_labels], dtype=object)

            # Invariant 1: Pre-match (0..11184 == X)
            pre_non_x = int(np.sum(full_cleaned[PRE_MATCH_SPAN[0]:PRE_MATCH_SPAN[1] + 1] != "X"))
            if pre_non_x > 0:
                msg = f"Pre-match invariant violated: frames {PRE_MATCH_SPAN[0]}..{PRE_MATCH_SPAN[1]} contains {pre_non_x} non-X labels"
                (errors if strict_invariants else warnings).append(msg)

            # Invariant 2: Halftime (90652..110807 == X)
            ht_non_x = int(np.sum(full_cleaned[HALFTIME_SPAN[0]:HALFTIME_SPAN[1] + 1] != "X"))
            if ht_non_x > 0:
                msg = f"Halftime invariant violated: frames {HALFTIME_SPAN[0]}..{HALFTIME_SPAN[1]} contains {ht_non_x} non-X labels"
                (errors if strict_invariants else warnings).append(msg)

            # Invariant 3: H2 Pre-Kickoff (110808..113741 == D)
            h2_non_d = int(np.sum(full_cleaned[H2_PREKICK_SPAN[0]:H2_PREKICK_SPAN[1] + 1] != "D"))
            if h2_non_d > 0:
                msg = f"H2 pre-kickoff invariant violated: frames {H2_PREKICK_SPAN[0]}..{H2_PREKICK_SPAN[1]} contains {h2_non_d} non-D labels"
                (errors if strict_invariants else warnings).append(msg)

            # Invariant 4: Post-match (190198..199318 == X)
            post_non_x = int(np.sum(full_cleaned[POST_MATCH_SPAN[0]:POST_MATCH_SPAN[1] + 1] != "X"))
            if post_non_x > 0:
                msg = f"Post-match invariant violated: frames {POST_MATCH_SPAN[0]}..{POST_MATCH_SPAN[1]} contains {post_non_x} non-X labels"
                (errors if strict_invariants else warnings).append(msg)

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings, stats=stats)

    @classmethod
    def validate_file(
        cls,
        path: Union[str, Path],
        expected_rows: int = TOTAL_FRAMES,
        strict_invariants: bool = True,
    ) -> ValidationResult:
        p = Path(path)
        if not p.is_file():
            return ValidationResult(is_valid=False, errors=[f"File not found: {p}"])
        try:
            df = pd.read_csv(p, dtype={"label": "object"})
        except Exception as e:
            return ValidationResult(is_valid=False, errors=[f"Could not parse CSV: {e}"])
        return cls.validate_dataframe(df, expected_rows=expected_rows, strict_invariants=strict_invariants)
