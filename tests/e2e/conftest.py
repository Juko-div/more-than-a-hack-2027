"""tests/e2e/conftest.py
Vectorized test fixtures, synthetic generators, and environment hooks.
"""
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
import pytest

from tests.e2e.utils.submission_validator import (
    TOTAL_FRAMES,
    PRE_MATCH_SPAN,
    H1_SPAN,
    HALFTIME_SPAN,
    H2_PREKICK_SPAN,
    H2_SPAN,
    POST_MATCH_SPAN,
)


@pytest.fixture(scope="session")
def workspace_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="session")
def evaluation_dir(workspace_root: Path) -> Path:
    return workspace_root / "evaluation"


@pytest.fixture(scope="session")
def public_gt_path(evaluation_dir: Path) -> Path:
    return evaluation_dir / "gt" / "BAR-ATH_possession_gt_public.csv"


@pytest.fixture(scope="session")
def baseline_v2_path(evaluation_dir: Path) -> Path:
    return evaluation_dir / "results" / "baseline_v2.csv"


@pytest.fixture
def make_valid_submission() -> Callable[..., pd.DataFrame]:
    """Generates a strictly valid 199,319-row submission DataFrame in ~18ms."""
    def _generator(n_rows: int = TOTAL_FRAMES) -> pd.DataFrame:
        frames = np.arange(n_rows, dtype=np.int64)
        labels = np.full(n_rows, "X", dtype=object)

        if n_rows == TOTAL_FRAMES:
            # Graded H1 (11185..90651): alternating H, D, A, D pattern
            h1_len = H1_SPAN[1] - H1_SPAN[0] + 1
            pattern = np.array(["H"] * 100 + ["D"] * 25 + ["A"] * 100 + ["D"] * 25, dtype=object)
            tiled = np.tile(pattern, int(np.ceil(h1_len / len(pattern))))[:h1_len]
            labels[H1_SPAN[0]:H1_SPAN[1] + 1] = tiled

            # Halftime (90652..110807): stays X

            # H2 Pre-kickoff (110808..113741): strictly D
            labels[H2_PREKICK_SPAN[0]:H2_PREKICK_SPAN[1] + 1] = "D"

            # Graded H2 (113742..190197): alternating H, D, A, D pattern
            h2_len = H2_SPAN[1] - H2_SPAN[0] + 1
            tiled_h2 = np.tile(pattern, int(np.ceil(h2_len / len(pattern))))[:h2_len]
            labels[H2_SPAN[0]:H2_SPAN[1] + 1] = tiled_h2

            # Post-match (190198..199318): stays X

        return pd.DataFrame({"frame": frames, "label": labels})
    return _generator


@pytest.fixture
def make_corrupted_submission(make_valid_submission) -> Callable[[str], pd.DataFrame]:
    """Applies specific mutations to a valid submission for boundary and fault testing."""
    def _corrupt(corruption_type: str) -> pd.DataFrame:
        df = make_valid_submission()
        if corruption_type == "missing_frame_col":
            df = df.rename(columns={"frame": "idx"})
        elif corruption_type == "missing_label_col":
            df = df.rename(columns={"label": "possession"})
        elif corruption_type == "row_count_short":
            df = df.iloc[:-1].copy()  # 199,318 rows
        elif corruption_type == "row_count_long":
            extra = pd.DataFrame({"frame": [TOTAL_FRAMES], "label": ["X"]})
            df = pd.concat([df, extra], ignore_index=True)
        elif corruption_type == "float_frames":
            df["frame"] = df["frame"].astype(float)
        elif corruption_type == "frame_gap":
            df = df.copy()
            df.loc[5000:, "frame"] += 1  # frame 5000 missing
        elif corruption_type == "frame_duplicate":
            df = df.copy()
            df.loc[5001, "frame"] = df.loc[5000, "frame"]
        elif corruption_type == "frame_unordered":
            df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        elif corruption_type == "nan_label":
            df = df.copy()
            df.loc[25000, "label"] = np.nan
        elif corruption_type == "empty_label":
            df = df.copy()
            df.loc[25000, "label"] = ""
        elif corruption_type == "whitespace_only_label":
            df = df.copy()
            df.loc[25000, "label"] = "   "
        elif corruption_type == "invalid_label_z":
            df = df.copy()
            df.loc[25000, "label"] = "Z"
        elif corruption_type == "invalid_label_word":
            df = df.copy()
            df.loc[25000, "label"] = "HOME"
        elif corruption_type == "pre_match_invariant_broken":
            df = df.copy()
            df.loc[100, "label"] = "H"
        elif corruption_type == "halftime_invariant_broken":
            df = df.copy()
            df.loc[95000, "label"] = "A"
        elif corruption_type == "h2_prekick_invariant_broken":
            df = df.copy()
            df.loc[111000, "label"] = "H"
        elif corruption_type == "post_match_invariant_broken":
            df = df.copy()
            df.loc[195000, "label"] = "D"
        else:
            raise ValueError(f"Unknown corruption type: {corruption_type}")
        return df
    return _corrupt
