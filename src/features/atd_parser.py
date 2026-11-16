"""src/features/atd_parser.py

High-performance, memory-efficient Automatic Tracking Data (ATD) parser and
coordinate transformer for Metrica FIFA/EPTS tactical tracking data.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import os
import xml.etree.ElementTree as ET
import numpy as np

logger = logging.getLogger(__name__)

# Canonical Challenge Constants
PITCH_LENGTH_METERS: float = 105.3
PITCH_WIDTH_METERS: float = 68.0
TOTAL_VIDEO_FRAMES: int = 199319
FPS: float = 25.0

# Team Slot Definitions
HOME_SLOTS: slice = slice(0, 24)
AWAY_SLOTS: slice = slice(24, 46)
N_HOME_SLOTS: int = 24
N_AWAY_SLOTS: int = 22
TOTAL_PLAYER_SLOTS: int = 46


class ATDParseError(Exception):
    """Raised when ATD metadata or raw tracking format is invalid."""
    pass


@dataclass(frozen=True)
class TeamMetadata:
    """Metadata for a team in the EPTS match export."""
    team_id: str
    name: str
    attack_direction: Optional[str] = None


@dataclass
class ATDFrameData:
    """
    Structured container for tactical tracking data across frames.
    All positional arrays are in meters (float32), with np.nan for untracked objects.
    """
    n_frames: int
    fps: float
    pitch_dim: Tuple[float, float]
    home_team: TeamMetadata
    away_team: TeamMetadata

    # Metric coordinates (meters): shape (n_frames, n_slots, 2)
    home_pos_m: np.ndarray        # (n_frames, 24, 2)
    away_pos_m: np.ndarray        # (n_frames, 22, 2)
    ball_pos_m: np.ndarray        # (n_frames, 2)

    # Speeds (m/s): float32
    home_speed: np.ndarray        # (n_frames, 24)
    away_speed: np.ndarray        # (n_frames, 22)
    ball_speed_raw: np.ndarray    # (n_frames,)

    # Frame indexing
    video_frames: np.ndarray      # (n_frames,), int32: [0, 1, ..., n_frames-1]

    @property
    def all_players_pos_m(self) -> np.ndarray:
        """Concatenated array of all player positions: shape (n_frames, 46, 2)."""
        return np.concatenate([self.home_pos_m, self.away_pos_m], axis=1)

    @property
    def all_players_speed(self) -> np.ndarray:
        """Concatenated array of all player speeds: shape (n_frames, 46)."""
        return np.concatenate([self.home_speed, self.away_speed], axis=1)

    @property
    def home_pos_norm(self) -> np.ndarray:
        """Normalized [0, 1] coordinates for Home players."""
        scale = np.array([self.pitch_dim[0], self.pitch_dim[1]], dtype=np.float32)
        return self.home_pos_m / scale

    @property
    def away_pos_norm(self) -> np.ndarray:
        """Normalized [0, 1] coordinates for Away players."""
        scale = np.array([self.pitch_dim[0], self.pitch_dim[1]], dtype=np.float32)
        return self.away_pos_m / scale

    @property
    def ball_pos_norm(self) -> np.ndarray:
        """Normalized [0, 1] coordinates for the ball."""
        scale = np.array([self.pitch_dim[0], self.pitch_dim[1]], dtype=np.float32)
        return self.ball_pos_m / scale

    @property
    def ball_detected_mask(self) -> np.ndarray:
        """Boolean mask of frames where ball was tracked."""
        return ~np.isnan(self.ball_pos_m[:, 0])

    def is_ball_out_of_bounds(self, margin_m: float = 0.5) -> np.ndarray:
        """
        Boolean mask indicating if ball is beyond pitch perimeter + margin_m.
        Returns False for untracked frames.
        """
        bx = self.ball_pos_m[:, 0]
        by = self.ball_pos_m[:, 1]
        L, W = self.pitch_dim
        seen = self.ball_detected_mask
        out = seen & ((bx < -margin_m) | (bx > L + margin_m) | (by < -margin_m) | (by > W + margin_m))
        return out


class ATDParser:
    """
    Production parser for tactical ATD datasets.
    Implements memory-efficient streaming, metric conversion, and .npz caching.
    """
    def __init__(
        self,
        pitch_length: float = PITCH_LENGTH_METERS,
        pitch_width: float = PITCH_WIDTH_METERS,
        fps: float = FPS,
    ):
        self.pitch_dim = (float(pitch_length), float(pitch_width))
        self.fps = float(fps)

    def parse_metadata(self, xml_path: Union[str, Path]) -> Tuple[TeamMetadata, TeamMetadata, np.ndarray]:
        """
        Parses XML metadata.
        Returns (home_team, away_team, slot_team_ids).
        """
        xml_path = Path(xml_path)
        if not xml_path.is_file():
            raise FileNotFoundError(f"ATD metadata XML not found: {xml_path}")

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            raise ATDParseError(f"XML parse error in {xml_path}: {e}")

        teams_elem = list(root.iter("Team"))
        if len(teams_elem) < 2:
            raise ATDParseError(f"Expected at least 2 teams in XML, found {len(teams_elem)}")

        def _get_direction(team_elem):
            for param in team_elem.iter("ProviderParameter"):
                if param.findtext("Name") == "attack_direction_first_half":
                    return param.findtext("Value")
            return None

        home_team = TeamMetadata(
            team_id=teams_elem[0].get("id", ""),
            name=teams_elem[0].findtext("Name") or "Home",
            attack_direction=_get_direction(teams_elem[0]),
        )
        away_team = TeamMetadata(
            team_id=teams_elem[1].get("id", ""),
            name=teams_elem[1].findtext("Name") or "Away",
            attack_direction=_get_direction(teams_elem[1]),
        )

        players_elem = list(root.iter("Player"))
        if len(players_elem) != TOTAL_PLAYER_SLOTS:
            logger.warning(f"Expected {TOTAL_PLAYER_SLOTS} players, found {len(players_elem)}")

        slot_team_ids = np.array([
            0 if p.get("teamId") == home_team.team_id else 1
            for p in players_elem
        ], dtype=np.int8)

        return home_team, away_team, slot_team_ids

    def parse(
        self,
        raw_path: Union[str, Path],
        xml_path: Optional[Union[str, Path]] = None,
        n_frames: int = TOTAL_VIDEO_FRAMES,
        use_cache: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> ATDFrameData:
        """
        Parses tracking data file and builds metric ATDFrameData.
        Uses binary cache if available.
        """
        raw_path = Path(raw_path)

        if not raw_path.is_file():
            raise FileNotFoundError(f"ATD raw tracking file not found: {raw_path}")

        if xml_path is not None:
            home_team, away_team, slot_team_ids = self.parse_metadata(xml_path)
        else:
            home_team = TeamMetadata("MET_user_team_o4Hossvb8M", "Barcelona", "left_to_right")
            away_team = TeamMetadata("MET_user_team_x7C4FFX2HF", "Atletic Club", "right_to_left")
            slot_team_ids = np.zeros(TOTAL_PLAYER_SLOTS, dtype=np.int8)
            slot_team_ids[AWAY_SLOTS] = 1

        # Check binary cache
        if cache_dir is None:
            cache_dir = raw_path.parent
        cache_path = Path(cache_dir) / f".{raw_path.stem}_cache.npz"

        L, W = self.pitch_dim
        cache_key = np.array([float(n_frames), float(L), float(W), float(self.fps)], dtype=np.float32)

        if use_cache and cache_path.is_file():
            try:
                with np.load(cache_path) as cached:
                    if "key" in cached.files and np.allclose(cached["key"], cache_key):
                        logger.info(f"Loaded ATD tracking from cache: {cache_path}")
                        return ATDFrameData(
                            n_frames=n_frames,
                            fps=self.fps,
                            pitch_dim=self.pitch_dim,
                            home_team=home_team,
                            away_team=away_team,
                            home_pos_m=cached["home_pos_m"],
                            away_pos_m=cached["away_pos_m"],
                            ball_pos_m=cached["ball_pos_m"],
                            home_speed=cached["home_speed"],
                            away_speed=cached["away_speed"],
                            ball_speed_raw=cached["ball_speed_raw"],
                            video_frames=np.arange(n_frames, dtype=np.int32),
                        )
            except Exception as e:
                logger.warning(f"Cache load failed ({e}), falling back to raw file parsing.")

        # Allocate arrays
        px = np.full((n_frames, TOTAL_PLAYER_SLOTS), np.nan, dtype=np.float32)
        py = np.full((n_frames, TOTAL_PLAYER_SLOTS), np.nan, dtype=np.float32)
        ps = np.full((n_frames, TOTAL_PLAYER_SLOTS), np.nan, dtype=np.float32)
        ball = np.full((n_frames, 3), np.nan, dtype=np.float32)

        # Stream lines
        with open(raw_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line_str = line.strip()
                if not line_str:
                    continue
                parts = line_str.split(":")
                if len(parts) != 3:
                    continue
                try:
                    fid = int(parts[0])
                except ValueError:
                    continue
                t = fid - 1  # 1-indexed to 0-indexed
                if not (0 <= t < n_frames):
                    continue

                # Parse players
                try:
                    player_str = parts[1].replace(";", ",")
                    while ",," in player_str:
                        player_str = player_str.replace(",,", ",NaN,")
                    if player_str.startswith(","):
                        player_str = "NaN" + player_str
                    if player_str.endswith(","):
                        player_str = player_str + "NaN"

                    if player_str:
                        vals = np.fromstring(player_str, sep=",", dtype=np.float32)
                        k = min(len(vals) // 3, TOTAL_PLAYER_SLOTS)
                        if k > 0:
                            group = vals[:k * 3].reshape(k, 3)
                            px[t, :k] = group[:, 0]
                            py[t, :k] = group[:, 1]
                            ps[t, :k] = group[:, 2]
                except Exception:
                    pass

                # Parse ball
                try:
                    ball_str = parts[2].strip()
                    while ",," in ball_str:
                        ball_str = ball_str.replace(",,", ",NaN,")
                    if ball_str.startswith(","):
                        ball_str = "NaN" + ball_str
                    if ball_str.endswith(","):
                        ball_str = ball_str + "NaN"

                    if ball_str:
                        b = np.fromstring(ball_str, sep=",", dtype=np.float32)
                        if len(b) >= 2:
                            ball[t, :min(len(b), 3)] = b[:3]
                except Exception:
                    pass

        # Convert to meters
        home_pos_m = np.stack([px[:, HOME_SLOTS] * L, py[:, HOME_SLOTS] * W], axis=-1)
        away_pos_m = np.stack([px[:, AWAY_SLOTS] * L, py[:, AWAY_SLOTS] * W], axis=-1)
        ball_pos_m = np.stack([ball[:, 0] * L, ball[:, 1] * W], axis=-1)

        home_speed = ps[:, HOME_SLOTS]
        away_speed = ps[:, AWAY_SLOTS]
        ball_speed_raw = ball[:, 2]

        data = ATDFrameData(
            n_frames=n_frames,
            fps=self.fps,
            pitch_dim=self.pitch_dim,
            home_team=home_team,
            away_team=away_team,
            home_pos_m=home_pos_m,
            away_pos_m=away_pos_m,
            ball_pos_m=ball_pos_m,
            home_speed=home_speed,
            away_speed=away_speed,
            ball_speed_raw=ball_speed_raw,
            video_frames=np.arange(n_frames, dtype=np.int32),
        )

        # Save cache
        if use_cache:
            try:
                np.savez_compressed(
                    cache_path,
                    key=cache_key,
                    home_pos_m=home_pos_m,
                    away_pos_m=away_pos_m,
                    ball_pos_m=ball_pos_m,
                    home_speed=home_speed,
                    away_speed=away_speed,
                    ball_speed_raw=ball_speed_raw,
                )
                logger.info(f"Saved ATD tracking cache to: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to write cache {cache_path}: {e}")

        return data
