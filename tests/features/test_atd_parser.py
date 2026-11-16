"""tests/features/test_atd_parser.py

Unit tests for ATD parsing, coordinate transformation, and team slot separation.
"""

from pathlib import Path
import numpy as np
import pytest

from src.features.atd_parser import (
    ATDFrameData,
    ATDParseError,
    ATDParser,
    AWAY_SLOTS,
    HOME_SLOTS,
    N_AWAY_SLOTS,
    N_HOME_SLOTS,
    PITCH_LENGTH_METERS,
    PITCH_WIDTH_METERS,
    TOTAL_PLAYER_SLOTS,
    TeamMetadata,
)


def _create_mock_xml(tmp_path: Path) -> Path:
    """Helper to generate a mock Metrica EPTS XML file."""
    xml_content = """<?xml version="1.0" encoding="utf-8"?>
<Metadata>
  <Teams>
    <Team id="MET_user_team_o4Hossvb8M">
      <Name>Barcelona</Name>
      <ProviderParameter>
        <Name>attack_direction_first_half</Name>
        <Value>left_to_right</Value>
      </ProviderParameter>
    </Team>
    <Team id="MET_user_team_x7C4FFX2HF">
      <Name>Atletic Club</Name>
      <ProviderParameter>
        <Name>attack_direction_first_half</Name>
        <Value>right_to_left</Value>
      </ProviderParameter>
    </Team>
  </Teams>
  <Players>
"""
    # 24 home players
    for i in range(1, 25):
        xml_content += f'    <Player id="p_{i}" teamId="MET_user_team_o4Hossvb8M"><ShirtNumber>{i}</ShirtNumber></Player>\n'
    # 22 away players
    for i in range(25, 47):
        xml_content += f'    <Player id="p_{i}" teamId="MET_user_team_x7C4FFX2HF"><ShirtNumber>{i}</ShirtNumber></Player>\n'
    xml_content += """  </Players>
</Metadata>
"""
    xml_file = tmp_path / "metadata.xml"
    xml_file.write_text(xml_content, encoding="utf-8")
    return xml_file


def _create_mock_raw(tmp_path: Path, n_lines: int = 10) -> Path:
    """Helper to generate mock raw ATD lines."""
    raw_file = tmp_path / "mock_atd.txt"
    lines = []
    for fid in range(1, n_lines + 1):
        # 46 player slots: x,y,speed
        players = []
        for p in range(46):
            if p == 0:
                players.append("0.5,0.5,2.5")
            elif p == 24:
                players.append("0.8,0.3,3.0")
            else:
                players.append("NaN,NaN,NaN")
        p_str = ";".join(players)
        # ball slot: bx,by,bspeed
        if fid % 2 == 1:
            b_str = "0.5,0.5,12.0"
        else:
            b_str = "NaN,NaN,NaN"
        lines.append(f"{fid}:{p_str}:{b_str}")

    raw_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return raw_file


def test_metadata_parsing(tmp_path):
    """Verifies team metadata parsing and slot assignment."""
    xml_file = _create_mock_xml(tmp_path)
    parser = ATDParser()
    home_team, away_team, slot_ids = parser.parse_metadata(xml_file)

    assert home_team.name == "Barcelona"
    assert home_team.team_id == "MET_user_team_o4Hossvb8M"
    assert home_team.attack_direction == "left_to_right"

    assert away_team.name == "Atletic Club"
    assert away_team.team_id == "MET_user_team_x7C4FFX2HF"
    assert away_team.attack_direction == "right_to_left"

    assert len(slot_ids) == TOTAL_PLAYER_SLOTS
    assert np.all(slot_ids[:N_HOME_SLOTS] == 0)
    assert np.all(slot_ids[N_HOME_SLOTS:] == 1)


def test_coordinate_conversion_exact():
    """Tests exact conversion from normalized coordinates to meters."""
    parser = ATDParser(pitch_length=105.3, pitch_width=68.0)
    L, W = parser.pitch_dim

    # Center of pitch
    assert (0.5 * L, 0.5 * W) == pytest.approx((52.65, 34.00), abs=1e-3)
    # Origin
    assert (0.0 * L, 0.0 * W) == (0.0, 0.0)
    # Top-right corner
    assert (1.0 * L, 1.0 * W) == (105.3, 68.0)


def test_coordinate_inversion(tmp_path):
    """Verifies normalized properties correctly invert metric coordinates."""
    xml_file = _create_mock_xml(tmp_path)
    raw_file = _create_mock_raw(tmp_path, n_lines=5)

    parser = ATDParser()
    data = parser.parse(raw_file, xml_file, n_frames=5, use_cache=False)

    # Frame 0 has player 0 at (0.5, 0.5)
    np.testing.assert_allclose(data.home_pos_norm[0, 0], [0.5, 0.5], atol=1e-5)
    # Frame 0 has player 24 at (0.8, 0.3)
    np.testing.assert_allclose(data.away_pos_norm[0, 0], [0.8, 0.3], atol=1e-5)
    # Frame 0 has ball at (0.5, 0.5)
    np.testing.assert_allclose(data.ball_pos_norm[0], [0.5, 0.5], atol=1e-5)


def test_zero_indexing_mapping(tmp_path):
    """Tests that ATD 1-indexed line IDs map strictly to 0-indexed frame indices."""
    xml_file = _create_mock_xml(tmp_path)
    raw_file = tmp_path / "single_line.txt"
    # Line ID 11185
    raw_file.write_text("11185:0.5,0.5,1.0:0.5,0.5,1.0\n", encoding="utf-8")

    parser = ATDParser()
    data = parser.parse(raw_file, xml_file, n_frames=12000, use_cache=False)

    assert data.video_frames[11184] == 11184
    # Frame 11184 must have ball data
    assert np.isfinite(data.ball_pos_m[11184, 0])
    # Frame 11183 must be NaN
    assert np.isnan(data.ball_pos_m[11183, 0])


def test_team_slot_separation_shapes(tmp_path):
    """Tests array shapes and team slot splitting."""
    xml_file = _create_mock_xml(tmp_path)
    raw_file = _create_mock_raw(tmp_path, n_lines=8)

    parser = ATDParser()
    data = parser.parse(raw_file, xml_file, n_frames=8, use_cache=False)

    assert data.home_pos_m.shape == (8, 24, 2)
    assert data.away_pos_m.shape == (8, 22, 2)
    assert data.ball_pos_m.shape == (8, 2)

    assert data.home_speed.shape == (8, 24)
    assert data.away_speed.shape == (8, 22)
    assert data.ball_speed_raw.shape == (8,)

    assert data.all_players_pos_m.shape == (8, 46, 2)
    assert data.all_players_speed.shape == (8, 46)


def test_ball_detection_and_missing_values(tmp_path):
    """Tests detection mask and missing ball NaN handling."""
    xml_file = _create_mock_xml(tmp_path)
    raw_file = _create_mock_raw(tmp_path, n_lines=6)

    parser = ATDParser()
    data = parser.parse(raw_file, xml_file, n_frames=6, use_cache=False)

    # In mock generator, odd lines (frames 0, 2, 4) have ball, even lines (1, 3, 5) do not
    assert data.ball_detected_mask[0] == True
    assert data.ball_detected_mask[1] == False
    assert data.ball_detected_mask[2] == True
    assert data.ball_detected_mask[3] == False
    assert np.isnan(data.ball_pos_m[1, 0])


def test_out_of_bounds_detection():
    """Tests is_ball_out_of_bounds helper method on ATDFrameData."""
    home_meta = TeamMetadata("h", "Home")
    away_meta = TeamMetadata("a", "Away")

    ball_pos = np.array([
        [52.65, 34.00],   # In-bounds center
        [-1.0, 34.00],    # Out-of-bounds left
        [106.0, 34.00],   # Out-of-bounds right (margin=0.5, L=105.3 -> 105.8 limit)
        [52.65, -0.6],    # Out-of-bounds top
        [np.nan, np.nan], # Missing
    ], dtype=np.float32)

    data = ATDFrameData(
        n_frames=5,
        fps=25.0,
        pitch_dim=(105.3, 68.0),
        home_team=home_meta,
        away_team=away_meta,
        home_pos_m=np.zeros((5, 24, 2), dtype=np.float32),
        away_pos_m=np.zeros((5, 22, 2), dtype=np.float32),
        ball_pos_m=ball_pos,
        home_speed=np.zeros((5, 24), dtype=np.float32),
        away_speed=np.zeros((5, 22), dtype=np.float32),
        ball_speed_raw=np.zeros(5, dtype=np.float32),
        video_frames=np.arange(5, dtype=np.int32),
    )

    out = data.is_ball_out_of_bounds(margin_m=0.5)
    assert out[0] == False
    assert out[1] == True
    assert out[2] == True
    assert out[3] == True
    assert out[4] == False  # untracked frames return False


def test_synthetic_stream_parsing(tmp_path):
    """Verifies end-to-end parsing accuracy from mock text file."""
    xml_file = _create_mock_xml(tmp_path)
    raw_file = _create_mock_raw(tmp_path, n_lines=10)

    parser = ATDParser()
    data = parser.parse(raw_file, xml_file, n_frames=10, use_cache=False)

    assert data.n_frames == 10
    assert data.fps == 25.0
    assert np.isclose(data.home_pos_m[0, 0, 0], 0.5 * 105.3)
    assert np.isclose(data.home_pos_m[0, 0, 1], 0.5 * 68.0)
    assert np.isclose(data.home_speed[0, 0], 2.5)


def test_cache_serialization_roundtrip(tmp_path):
    """Verifies saving and loading from .npz binary cache."""
    xml_file = _create_mock_xml(tmp_path)
    raw_file = _create_mock_raw(tmp_path, n_lines=10)

    parser = ATDParser()
    # 1. Parse and save cache
    data1 = parser.parse(raw_file, xml_file, n_frames=10, use_cache=True, cache_dir=tmp_path)

    cache_file = tmp_path / f".{raw_file.stem}_cache.npz"
    assert cache_file.is_file()

    # 2. Parse again with use_cache=True
    data2 = parser.parse(raw_file, xml_file, n_frames=10, use_cache=True, cache_dir=tmp_path)

    np.testing.assert_allclose(data1.home_pos_m, data2.home_pos_m, equal_nan=True)
    np.testing.assert_allclose(data1.away_pos_m, data2.away_pos_m, equal_nan=True)
    np.testing.assert_allclose(data1.ball_pos_m, data2.ball_pos_m, equal_nan=True)
    np.testing.assert_allclose(data1.home_speed, data2.home_speed, equal_nan=True)
    np.testing.assert_allclose(data1.away_speed, data2.away_speed, equal_nan=True)
    np.testing.assert_allclose(data1.ball_speed_raw, data2.ball_speed_raw, equal_nan=True)


def test_error_handling_missing_files(tmp_path):
    """Verifies FileNotFoundError on non-existent files."""
    parser = ATDParser()
    with pytest.raises(FileNotFoundError):
        parser.parse_metadata(tmp_path / "non_existent.xml")

    with pytest.raises(FileNotFoundError):
        parser.parse(tmp_path / "non_existent.txt", n_frames=10)


def test_error_handling_corrupt_lines(tmp_path):
    """Verifies parser handles lines with malformed colons or empty strings."""
    xml_file = _create_mock_xml(tmp_path)
    raw_file = tmp_path / "corrupt.txt"
    content = (
        "not a line\n"
        "2:0.5,0.5,1.0;0.5,0.5,1.0:0.5,0.5,5.0\n"
        "3:malformed_colon\n"
        "\n"
        "5:invalid_frame_number:invalid_ball\n"
        "6::0.5,0.5,5.0\n"
    )
    raw_file.write_text(content, encoding="utf-8")

    parser = ATDParser()
    data = parser.parse(raw_file, xml_file, n_frames=10, use_cache=False)
    # Line 2 (video frame 1) should be successfully parsed
    assert np.isfinite(data.ball_pos_m[1, 0])
    # Line 6 (video frame 5) should have ball parsed
    assert np.isfinite(data.ball_pos_m[5, 0])


def test_metadata_missing_teams_raises_error(tmp_path):
    """Verifies ATDParseError when XML has fewer than 2 teams."""
    xml_file = tmp_path / "bad_metadata.xml"
    xml_file.write_text("<Metadata><Teams><Team id='1'><Name>OnlyOne</Name></Team></Teams></Metadata>", encoding="utf-8")
    parser = ATDParser()
    with pytest.raises(ATDParseError):
        parser.parse_metadata(xml_file)


def test_real_match_metadata_and_sample_parse(tmp_path):
    """Integration test: validates real XML and parses first 50 frames of real ATD text."""
    xml_path = Path("data/tactical/BAR-ATH_tactical_atd.xml")
    raw_path = Path("data/tactical/BAR-ATH_tactical_atd.txt")

    if not (xml_path.is_file() and raw_path.is_file()):
        pytest.skip("Real match tracking data files not found.")

    parser = ATDParser()
    home_meta, away_meta, slot_ids = parser.parse_metadata(xml_path)

    assert home_meta.name == "Barcelona"
    assert away_meta.name == "Atletic Club"
    assert len(slot_ids) == 46
    assert np.all(slot_ids[:24] == 0)
    assert np.all(slot_ids[24:] == 1)

    # Parse 50 frames without caching to test raw streaming speed & accuracy
    data = parser.parse(raw_path, xml_path, n_frames=50, use_cache=False)
    assert data.n_frames == 50
    assert data.home_pos_m.shape == (50, 24, 2)
    assert data.away_pos_m.shape == (50, 22, 2)
    assert data.ball_pos_m.shape == (50, 2)
    # Check that frame 0 coordinates match line 1 values (0.33411 * 105.3, 0.90565 * 68.0)
    assert np.isclose(data.home_pos_m[0, 0, 0], 0.33411 * 105.3, atol=1e-3)
    assert np.isclose(data.home_pos_m[0, 0, 1], 0.90565 * 68.0, atol=1e-3)

