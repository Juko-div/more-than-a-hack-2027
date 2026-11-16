"""Computer Vision & Slot Re-Identification Module.

Analyzes player jersey colors directly from the tactical match video
(BAR-ATH_tactical-001.mp4) using homography reprojection and ATD tracking data.

Aggregates color evidence across 5-10 clear, isolated frames per player slot to
produce a robust visual consensus mapping: slot_team_consensus[s] in {0, 1}
(0 = HOME / Barcelona, 1 = AWAY / Athletic Club).
"""

from __future__ import division, print_function

import json
import os
import sys
import time
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from games import GAME, GAMES, require, source_frames
from starter.baseline import team_of_each_track
from starter.baseline_v3 import read_atd
from starter.load_homography import load as load_homography, project


def extract_torso_patch(img, u, v):
    """Crops torso patch above the ground contact point (u, v)."""
    # Perspective-aware player height and width in tactical camera
    h_player = max(24.0, 0.085 * v)
    w_player = max(8.0, 0.30 * h_player)

    v_top = int(round(v - 0.85 * h_player))
    v_bot = int(round(v - 0.25 * h_player))
    u_left = int(round(u - w_player))
    u_right = int(round(u + w_player))

    # Clamp to image boundaries
    img_h, img_w = img.shape[:2]
    v_top = max(0, min(img_h - 1, v_top))
    v_bot = max(0, min(img_h, v_bot))
    u_left = max(0, min(img_w - 1, u_left))
    u_right = max(0, min(img_w, u_right))

    if v_bot <= v_top or u_right <= u_left:
        return None
    return img[v_top:v_bot, u_left:u_right]


def score_patch_colors(patch):
    """Computes HSV color histogram / scores for Barça (HOME=0) vs Athletic (AWAY=1)."""
    if patch is None or patch.size == 0:
        return None

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Filter out pitch grass: Hue 35..85, Sat > 40
    grass_mask = (h >= 35) & (h <= 85) & (s > 40)
    player_mask = ~grass_mask

    n_player = int(np.sum(player_mask))
    if n_player < 10:
        return None

    hp = h[player_mask]
    sp = s[player_mask]
    vp = v[player_mask]

    # Barça (HOME=0): Blaugrana (Deep Blue: Hue 100-135, Crimson/Red: Hue 0-10 or 160-180 with Sat > 30)
    blue_count = int(np.sum((hp >= 100) & (hp <= 135) & (sp > 30)))
    crimson_count = int(np.sum(((hp <= 10) | (hp >= 160)) & (sp > 30)))
    barca_px = blue_count + crimson_count
    barca_score = float(barca_px) / float(n_player)

    # Athletic (AWAY=1): Red-and-White stripes / White kit (White: Val > 180, Sat < 50, plus Red)
    white_count = int(np.sum((vp > 180) & (sp < 50)))
    athletic_px = white_count + crimson_count
    athletic_score = float(athletic_px) / float(n_player)

    blue_pct = (float(blue_count) / float(n_player)) * 100.0
    crimson_pct = (float(crimson_count) / float(n_player)) * 100.0
    white_pct = (float(white_count) / float(n_player)) * 100.0

    # Decision rule:
    # Athletic's away kit is dominated by bright white (white_pct > 25%).
    # Barça's Blaugrana kit has dark blue & crimson tones (white_pct < 10%).
    is_barca = (white_pct < 20.0) or (barca_score > athletic_score and white_pct < 30.0)
    pred_team = 0 if is_barca else 1

    return {
        "n_player_px": n_player,
        "blue_pct": blue_pct,
        "crimson_pct": crimson_pct,
        "white_pct": white_pct,
        "barca_score": barca_score,
        "athletic_score": athletic_score,
        "pred_team": pred_team,
    }


def find_isolated_slot_frames(px, py, length, width, H, reproj, valid_H, n_slots=46, target_samples=8):
    """Finds 5 to 10 isolated, clear frames on the pitch for each player slot."""
    samples_per_slot = {s: [] for s in range(n_slots)}

    # Scan across match play in regular intervals
    scan_step = 100
    for f in range(11200, len(px) - 1000, scan_step):
        if not valid_H[f] or reproj[f] > 1.0:
            continue

        valid_slots = [s for s in range(n_slots) if not np.isnan(px[f, s])]
        if len(valid_slots) < 10:
            continue

        xs = px[f, valid_slots] * length
        ys = py[f, valid_slots] * width

        for idx, s in enumerate(valid_slots):
            if len(samples_per_slot[s]) >= target_samples:
                continue

            x_norm, y_norm = px[f, s], py[f, s]
            if not (0.05 < x_norm < 0.95 and 0.05 < y_norm < 0.95):
                continue

            u, v = project(H[f], x_norm, y_norm)
            if not (50 <= u <= 1870 and 50 <= v <= 1030):
                continue

            # Check distance to nearest other player
            dx = xs - xs[idx]
            dy = ys - ys[idx]
            dists = np.sqrt(dx * dx + dy * dy)
            dists[idx] = np.inf
            min_d = np.min(dists)

            if min_d >= 2.5:
                samples_per_slot[s].append((f, float(u), float(v), float(min_d)))

    return samples_per_slot


def build_slot_jersey_consensus(game=GAME, verbose=True):
    """Computes the visual jersey consensus for all 46 player slots."""
    t0 = time.time()
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    video_path = require(["tactical_video"], game)[0] if "tactical_video" in GAMES[game] \
        else os.path.join(ROOT, "data", "tactical", f"{game}_tactical-001.mp4")

    n_frames = source_frames("tactical", game)
    meta_teams = team_of_each_track(meta_path)
    length, width = GAMES[game]["pitch"]
    H, reproj, valid_H = load_homography(game, source="tactical")

    if verbose:
        print(f"Reading ATD coordinates for {n_frames} frames...")
    px, py, ps, bx, by = read_atd(raw_path, n_frames, len(meta_teams))

    if verbose:
        print("Selecting clear, isolated frames for each slot (min_dist >= 2.5m)...")
    samples_per_slot = find_isolated_slot_frames(
        px, py, length, width, H, reproj, valid_H, n_slots=len(meta_teams), target_samples=8
    )

    # Invert mapping: unique_frame -> list of (slot, u, v)
    frame_to_slots = {}
    for s in range(len(meta_teams)):
        for f, u, v, _ in samples_per_slot[s]:
            frame_to_slots.setdefault(f, []).append((s, u, v))

    unique_frames = sorted(list(frame_to_slots.keys()))
    if verbose:
        print(f"Reading {len(unique_frames)} frames from video: {os.path.basename(video_path)}...")

    slot_evaluations = {s: [] for s in range(len(meta_teams))}
    cap = cv2.VideoCapture(video_path)
    try:
        for f in unique_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, img = cap.read()
            if not ret or img is None:
                continue

            for s, u, v in frame_to_slots[f]:
                patch = extract_torso_patch(img, u, v)
                score = score_patch_colors(patch)
                if score is not None:
                    score["frame"] = f
                    slot_evaluations[s].append(score)
    finally:
        cap.release()

    if verbose:
        print(f"Video processing finished in {time.time() - t0:.2f}s")

    # Aggregate consensus
    consensus = {}
    consensus_details = {}

    for s in range(len(meta_teams)):
        evals = slot_evaluations[s]
        if not evals:
            # Fallback to metadata
            pred = int(meta_teams[s])
            conf = 0.50
            mean_white = 0.0
            mean_blue = 0.0
            mean_crimson = 0.0
        else:
            votes = [ev["pred_team"] for ev in evals]
            barca_votes = sum(1 for v in votes if v == 0)
            athletic_votes = len(votes) - barca_votes
            pred = 0 if barca_votes > athletic_votes else 1
            conf = max(barca_votes, athletic_votes) / float(len(votes))
            mean_white = float(np.mean([ev["white_pct"] for ev in evals]))
            mean_blue = float(np.mean([ev["blue_pct"] for ev in evals]))
            mean_crimson = float(np.mean([ev["crimson_pct"] for ev in evals]))

        consensus[s] = pred
        consensus_details[s] = {
            "slot": s,
            "meta_team": int(meta_teams[s]),
            "consensus_team": pred,
            "confidence": float(conf),
            "samples_count": len(evals),
            "mean_white_pct": mean_white,
            "mean_blue_pct": mean_blue,
            "mean_crimson_pct": mean_crimson,
            "team_label": "HOME (Barça)" if pred == 0 else "AWAY (Athletic)",
        }

        if verbose:
            tag = " [CHANGED!]" if pred != meta_teams[s] else ""
            print(f"Slot {s:2d} (Meta={meta_teams[s]}): Pred={pred} (Conf={conf:.2f}, N={len(evals):2d}, "
                  f"White={mean_white:4.1f}%, Blue={mean_blue:4.1f}%, Crimson={mean_crimson:4.1f}%){tag}")

    # Save output files
    data_dir = os.path.join(ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    json_path = os.path.join(data_dir, ".slot_jersey_consensus.json")
    with open(json_path, "w") as f:
        json.dump(consensus_details, f, indent=2)

    npy_path = os.path.join(data_dir, ".slot_jersey_consensus.npy")
    consensus_array = np.array([consensus[s] for s in range(len(meta_teams))], dtype=np.int8)
    np.save(npy_path, consensus_array)

    if verbose:
        print(f"\n[OK] Consensus saved to:")
        print(f"     - JSON: {os.path.relpath(json_path, ROOT)}")
        print(f"     - NPY:  {os.path.relpath(npy_path, ROOT)}")

    return consensus_array, consensus_details


if __name__ == "__main__":
    print("=" * 70)
    print("SLOT JERSEY CONSENSUS - COMPUTER VISION RE-IDENTIFICATION")
    print("=" * 70)
    arr, details = build_slot_jersey_consensus(verbose=True)
    
    print("\n" + "=" * 70)
    print("SLOT 44 SPECIFIC VERIFICATION:")
    print("=" * 70)
    s44 = details[44]
    print(f"Slot 44 Metadata Team:    {s44['meta_team']} (AWAY)")
    print(f"Slot 44 Consensus Team:   {s44['consensus_team']} ({s44['team_label']})")
    print(f"Slot 44 Visual Confidence:{s44['confidence']:.2f}")
    print(f"Slot 44 Sample Count:     {s44['samples_count']}")
    print(f"Slot 44 White Pct:        {s44['mean_white_pct']:.1f}%")
    print(f"Slot 44 Blue Pct:         {s44['mean_blue_pct']:.1f}%")
    print(f"Slot 44 Crimson Pct:      {s44['mean_crimson_pct']:.1f}%")
    print("=" * 70)
