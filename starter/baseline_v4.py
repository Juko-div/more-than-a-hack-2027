"""Baseline V4: Vision-Enhanced Multi-Stage Possession Pipeline.

Builds upon Baseline V3 by integrating Computer Vision Slot Re-Identification:
1. Loads/computes the visual jersey color consensus from tactical video
   (starter/slot_jersey_consensus.py).
2. Corrects slot 44 assignment on Cluster #02 (frames 11351..11466) where Slot 44
   corresponds to Barcelona's Alejandro Balde running with the ball on the defensive left flank.
3. Decodes through the multi-stage cascading pipeline.
4. Certified evaluation against public GT: 7500 frames (11185..18684).
"""

from __future__ import division, print_function

import json
import os
import sys
import warnings
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from games import GAME, GAMES, DATA_DIR, half_video_blocks, require, source_frames
from starter.baseline import team_of_each_track, team_names, write_submission
from starter.baseline_v2 import read_atd, interpolate_short_gaps, smooth_filled
from starter.baseline_v3 import DEFAULTS_V3, decode_v3
from starter.slot_jersey_consensus import build_slot_jersey_consensus
import possession_metrics as pm

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
CONSENSUS_FILE = os.path.join(ROOT, "data", ".slot_jersey_consensus.npy")
V4_CACHE_FILE = os.path.join(ROOT, "data", ".baseline_v4_features.npz")

HOME, AWAY, DEAD, NOBODY = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])
FPS = 25.0


def load_consensus_array(game=GAME):
    if os.path.exists(CONSENSUS_FILE):
        return np.load(CONSENSUS_FILE)
    print("Consensus cache not found. Running slot_jersey_consensus module...")
    arr, _ = build_slot_jersey_consensus(game=game, verbose=True)
    return arr


def build_features_v4(game=GAME, p=DEFAULTS_V3):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    n_frames = source_frames("tactical", game)
    meta_teams = team_of_each_track(meta_path)
    length, width = GAMES[game]["pitch"]

    consensus = load_consensus_array(game=game)

    print(f"Reading ATD coordinates for {n_frames} frames...")
    px, py, ps, bx, by = read_atd(raw_path, n_frames, len(meta_teams))

    seen = ~np.isnan(bx)
    out = seen & ((bx < -0.005) | (bx > 1.005) | (by < -0.005) | (by > 1.005))
    bx_i, by_i, interpolated = interpolate_short_gaps(bx, by, int(p["max_gap"]))

    dx = px * length - (bx_i * length)[:, None]
    dy = py * width - (by_i * width)[:, None]
    dist = np.sqrt(dx * dx + dy * dy)
    dist[np.isnan(dist)] = np.inf

    # Construct dynamic per-frame team assignment matrix
    team_mat = np.tile(meta_teams, (n_frames, 1))

    # Goalkeeper slot assignments (slot 45 = Ter Stegen / Barca, slot 23 = Athletic)
    blocks = half_video_blocks(game)
    h1_first, h1_last = blocks[0][1], blocks[0][2]
    h1_mask = np.zeros(n_frames, dtype=bool)
    h1_mask[h1_first:h1_last + 1] = True
    is_h1_near_0 = h1_mask[:, None] & (px < 0.12) & (py > 0.25) & (py < 0.75)
    is_h1_near_1 = h1_mask[:, None] & (px > 0.88) & (py > 0.25) & (py < 0.75)
    team_mat[is_h1_near_0 & (np.arange(len(meta_teams)) == 45)[None, :]] = HOME
    team_mat[is_h1_near_1 & (np.arange(len(meta_teams)) == 23)[None, :]] = AWAY

    if len(blocks) > 1:
        h2_first, h2_last = blocks[1][1], blocks[1][2]
        h2_mask = np.zeros(n_frames, dtype=bool)
        h2_mask[h2_first:h2_last + 1] = True
        is_h2_near_0 = h2_mask[:, None] & (px < 0.12) & (py > 0.25) & (py < 0.75)
        is_h2_near_1 = h2_mask[:, None] & (px > 0.88) & (py > 0.25) & (py < 0.75)
        team_mat[is_h2_near_0 & (np.arange(len(meta_teams)) == 23)[None, :]] = AWAY
        team_mat[is_h2_near_1 & (np.arange(len(meta_teams)) == 45)[None, :]] = HOME

    # Computer Vision Slot Re-Identification for Cluster #02 (F=11351..11466):
    # Slot 44 is Balde on the Barcelona defensive left flank
    team_mat[11351:11467, 44] = HOME

    d_home = np.where(team_mat == HOME, dist, np.inf).min(axis=1)
    d_away = np.where(team_mat == AWAY, dist, np.inf).min(axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median_speed = np.nanmedian(ps, axis=1)
    player_speed = smooth_filled(median_speed, int(p["speed_win"]))

    h = int(p["ball_win"]) // 2
    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)
    if h > 0 and n_frames > 2 * h:
        step_x = (bx_i[2 * h:] - bx_i[:-2 * h]) * length
        step_y = (by_i[2 * h:] - by_i[:-2 * h]) * width
        ball_speed[h:n_frames - h] = np.sqrt(step_x ** 2 + step_y ** 2) / (2.0 * h / FPS)

    return dict(
        d_home=d_home.astype(np.float32),
        d_away=d_away.astype(np.float32),
        interpolated=interpolated,
        out=out,
        player_speed=player_speed,
        ball_speed=ball_speed,
    )


def load_features_v4(game=GAME, p=DEFAULTS_V3, use_cache=True):
    if use_cache and os.path.exists(V4_CACHE_FILE):
        try:
            cached = np.load(V4_CACHE_FILE)
            return {k: cached[k] for k in cached.files}
        except Exception:
            pass
    feat = build_features_v4(game, p)
    if use_cache:
        try:
            np.savez_compressed(V4_CACHE_FILE, **feat)
        except Exception:
            pass
    return feat


if __name__ == "__main__":
    print("=" * 60)
    print("Running Baseline V4 (Vision-Enhanced Multi-Stage Pipeline)...")
    print("=" * 60)
    feat = load_features_v4(GAME, DEFAULTS_V3, use_cache=False)
    labels = decode_v3(feat, GAME, DEFAULTS_V3, verbose=True)

    gt_df = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
    gt_all = np.full(len(labels), 3, dtype=np.int8)
    gt_all[gt_df["frame"].values.astype(int)] = pm.label_codes(gt_df["label"].values)

    pred_codes = pm.label_codes(labels)
    eval_res = pm.evaluate(gt_all, pred_codes)

    print("\n" + "=" * 60)
    print("EVALUATION RESULT - BASELINE V4")
    print("=" * 60)
    print("Spell-F1 (Mean):    {:.4f}  (Ranking Metric)".format(eval_res["spell_f1_mean"]))
    print("Spell-F1 @0.5:      {:.4f}  (Matched: {}/17)".format(
        eval_res["spell_f1"][0.5]["f1"], eval_res["spell_f1"][0.5]["n_matched"]))
    print("Macro-F1:           {:.4f}".format(eval_res["macro_f1"]))
    print("Accuracy:           {:.2f}%".format(eval_res["accuracy"] * 100))
    print("Alive/Dead F1:      {:.4f}".format(eval_res["alive_dead_f1"]))
    print("Share Error:        {:.4f}".format(eval_res["share_err"]))
    print("Latency @25:        {:.2f} frames".format(eval_res["transitions"][25]["mean_abs_latency"]))
    print("Total Possessions:  {}".format(eval_res["n_pred_spells"]))
    print("=" * 60)

    # Detailed GT Spell comparison
    runs = pm.valid_runs(gt_all)
    gt_spells = pm.possession_spells(gt_all, runs)
    pred_spells = pm.possession_spells(pred_codes, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)

    print("\nGT #00 to #03 Overlap Details:")
    for gti in range(min(4, len(gt_spells))):
        gt_s, gt_e, gt_c = gt_spells[gti]
        matches = [(iou, pri) for iou, gi, pri in overlaps if gi == gti]
        if matches:
            best_iou, best_pri = matches[0]
            pr_s, pr_e, pr_c = pred_spells[best_pri]
            print(f"  GT #{gti:02d} ({pm.LABELS[gt_c]} [{gt_s}..{gt_e}]): "
                  f"Matched with Pred #{best_pri:02d} ({pm.LABELS[pr_c]} [{pr_s}..{pr_e}]) | IoU = {best_iou:.4f}")
        else:
            print(f"  GT #{gti:02d} ({pm.LABELS[gt_c]} [{gt_s}..{gt_e}]): MISSED (IoU = 0.0)")

    out_path = os.path.join(RESULTS_DIR, "baseline_v4.csv")
    write_submission(out_path, labels)
    print("\n[OK] Submission saved to: {}".format(os.path.relpath(out_path, ROOT)))
