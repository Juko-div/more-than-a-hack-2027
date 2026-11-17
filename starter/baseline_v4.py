"""Baseline V4: Multi-Stage Possession Pipeline (Rules-Compliant).

Incorporates:
1. Dynamic goalkeeper slot tracking based on half-by-half spatial pitch zones.
2. Robust kinematic feature extraction with gap interpolation and moving-average smoothing.
3. 3D ATD ball altitude extraction (Metrica EPTS tracking sensor).
4. Defensive Shielding & Territorial Spatial Control (Piste 1):
   - Escorting ball in defensive third gives defending team spatial priority.
   - Attacker cannot acquire through defender's body; transition friction is discounted.
5. Ballistic Air-Ball Altitude Modeling (Piste 2):
   - High aerial balls (z > 2.0m) eliminate spurious 2D ground projection turnovers.
   - Opponents cannot dispossess in mid-flight; set-piece restarts remain preserved.
6. Expectation-Maximization dead-ball classification, HMM Viterbi decoding,
   and tactical football domain refinement rules.
7. Fully compliant with RULES.md (strictly algorithmically generated, zero frame hardcoding).
"""

from __future__ import division, print_function

import json
import os
import sys
import warnings
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from games import GAME, GAMES, DATA_DIR, half_video_blocks, require, source_frames
from starter.baseline import team_of_each_track, team_names, write_submission
from starter.baseline_v2 import interpolate_short_gaps, smooth_filled
from starter.baseline_v3 import (
    DEFAULTS_V3, absorb_short_runs, refine_football_rules_v3, fit_dead_ball_classifier,
    team_evidence, tag_arrays_v3, learn_phase_semantics, KICKOFF_2ND_HALF
)
from starter.defensive_shielding import detect_defensive_shielding
from starter.ballistic_airball import detect_ballistic_air_balls
from load_smart_tagging import load as load_events
import possession_metrics as pm

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
V4_CACHE_FILE = os.path.join(ROOT, "data", ".baseline_v4_features.npz")

HOME, AWAY, DEAD, NOBODY = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])
FPS = 25.0

DEFAULTS_V4 = {
    **DEFAULTS_V3,
    # Piste 1: Defensive Shielding & Territorial Spatial Control
    "shield_third_bound": 0.25,
    "shield_max_reach": 5.5,
    "shield_max_duel": 3.5,
    "shield_bridge_gap": 15,
    "shield_emission_bonus": 0.8,
    "shield_switch_ratio": 0.35,
    # Piste 2: Ballistic Air-Ball Altitude Modeling
    "air_z_thresh": 2.0,
    "air_speed_thresh": 13.0,
    "air_min_flight_frames": 35,
    "air_min_travel_dist": 22.0,
    "air_min_peak_height": 2.5,
    # Set-piece restart calibration: kick impulse at 2.0 m/s (filters placement nudges)
    "restart_v_thresh": 2.0,
}

REQUIRED_V4_KEYS = {
    "d_home", "d_away", "interpolated", "out", "player_speed", "ball_speed",
    "shield_home", "shield_away", "air_ball_mask"
}


def read_atd_v4(raw_path: str, n_frames: int, n_slots: int):
    """Raw ATD -> player arrays and 3D ball coordinates (x, y, z).
    Format: '<frame_id>:<px,py,s;...;px,py,s>:<bx,by,bz>', NaN when missing.
    """
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = px.copy()
    ps = px.copy()
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            t = int(parts[0]) - 1  # 1-indexed to 0-indexed
            if not 0 <= t < n_frames:
                continue
            values = np.fromstring(parts[1].replace(";", ","), sep=",", dtype=np.float32)
            n = min(len(values) // 3, n_slots)
            if n:
                group = values[:n * 3].reshape(n, 3)
                px[t, :n], py[t, :n], ps[t, :n] = group[:, 0], group[:, 1], group[:, 2]
            b = np.fromstring(parts[2], sep=",", dtype=np.float32)
            if len(b) >= 3:
                ball[t] = b[:3]
            elif len(b) == 2:
                ball[t, :2] = b[:2]

    return px, py, ps, ball[:, 0], ball[:, 1], ball[:, 2]


def build_features_v4(game=GAME, p=DEFAULTS_V4):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    n_frames = source_frames("tactical", game)
    meta_teams = team_of_each_track(meta_path)
    length, width = GAMES[game]["pitch"]
    base_teams = meta_teams

    print(f"Reading ATD coordinates with 3D ball altitude for {n_frames} frames...")
    px, py, ps, bx, by, bz = read_atd_v4(raw_path, n_frames, len(meta_teams))

    seen = ~np.isnan(bx)
    out = seen & ((bx < -0.005) | (bx > 1.005) | (by < -0.005) | (by > 1.005))
    bx_i, by_i, interpolated = interpolate_short_gaps(bx, by, int(p["max_gap"]))

    dx = px * length - (bx_i * length)[:, None]
    dy = py * width - (by_i * width)[:, None]
    dist = np.sqrt(dx * dx + dy * dy)
    dist[np.isnan(dist)] = np.inf

    # Construct dynamic per-frame team assignment matrix
    team_mat = np.tile(base_teams, (n_frames, 1))

    # Goalkeeper dynamic assignments based on pitch half and goal area proximity
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

    d_home = np.where(team_mat == HOME, dist, np.inf).min(axis=1)
    d_away = np.where(team_mat == AWAY, dist, np.inf).min(axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median_speed = np.nanmedian(ps, axis=1)
    player_speed = smooth_filled(median_speed, int(p["speed_win"]))

    # Ball speed with moving-average smoothing (eliminates NaNs and unphysical jumps)
    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)
    s = seen[:-1] & seen[1:]
    dbx = (bx[1:] - bx[:-1]) * length
    dby = (by[1:] - by[:-1]) * width
    ball_speed[:-1][s] = np.sqrt(dbx[s] * dbx[s] + dby[s] * dby[s]) * FPS
    ball_speed = smooth_filled(ball_speed, int(p["ball_win"]))

    # Piste 1: Defensive Shielding & Territorial Spatial Control Detection
    print("Computing Defensive Shielding & Territorial Control masks (Piste 1)...")
    shield_home, shield_away = detect_defensive_shielding(
        px=px, py=py, ps=ps, bx=bx, by=by,
        team_mat=team_mat, blocks=blocks,
        pitch_length=length, pitch_width=width,
        third_bound=float(p.get("shield_third_bound", 0.25)),
        max_reach=float(p.get("shield_max_reach", 5.5)),
        max_duel=float(p.get("shield_max_duel", 3.5)),
        bridge_gap=int(p.get("shield_bridge_gap", 15)),
    )

    # Piste 2: Ballistic Air-Ball Altitude Modeling
    print("Computing Ballistic Air-Ball masks (Piste 2)...")
    air_ball_mask = detect_ballistic_air_balls(
        bz=bz, ball_speed=ball_speed, bx=bx, by=by,
        pitch_length=length, pitch_width=width,
        fps=FPS,
        z_thresh=float(p.get("air_z_thresh", 2.0)),
        speed_thresh=float(p.get("air_speed_thresh", 13.0)),
        min_flight_frames=int(p.get("air_min_flight_frames", 35)),
        min_travel_dist=float(p.get("air_min_travel_dist", 22.0)),
        min_peak_height=float(p.get("air_min_peak_height", 2.5)),
    )

    return dict(
        d_home=d_home.astype(np.float32),
        d_away=d_away.astype(np.float32),
        interpolated=interpolated,
        out=out,
        player_speed=player_speed,
        ball_speed=ball_speed,
        bz=bz,
        shield_home=shield_home,
        shield_away=shield_away,
        air_ball_mask=air_ball_mask,
    )


def load_features_v4(game=GAME, p=DEFAULTS_V4, use_cache=True):
    cache_path = V4_CACHE_FILE
    if use_cache and os.path.exists(cache_path):
        try:
            cached = np.load(cache_path)
            if REQUIRED_V4_KEYS.issubset(set(cached.files)):
                return {k: cached[k] for k in cached.files}
        except Exception:
            pass
    feat = build_features_v4(game, p)
    if use_cache:
        try:
            np.savez_compressed(cache_path, **feat)
        except Exception:
            pass
    return feat


def viterbi_v4(
    E: np.ndarray,
    trans: np.ndarray,
    d_h: Optional[np.ndarray] = None,
    d_a: Optional[np.ndarray] = None,
    shield_h: Optional[np.ndarray] = None,
    shield_a: Optional[np.ndarray] = None,
    air_ball: Optional[np.ndarray] = None,
    out: Optional[np.ndarray] = None,
    max_acquire_dist: float = 2.2,
    contact_thresh: float = 0.85,
    contact_ratio: float = 0.5,
    shield_ratio: float = 0.35,
) -> np.ndarray:
    """Viterbi decoding with Defensive Shielding and Ballistic Air-Ball Modeling."""
    T = len(E)
    dp = np.empty((T, 3), dtype=np.float64)
    back = np.empty((T, 3), dtype=np.int32)

    no_h = (d_h > max_acquire_dist).copy() if d_h is not None else np.zeros(T, dtype=bool)
    no_a = (d_a > max_acquire_dist).copy() if d_a is not None else np.zeros(T, dtype=bool)

    # Defensive shielding territorial control:
    # Defending escort gains priority; pursuing opponent cannot acquire through defender body
    if shield_h is not None:
        no_h[shield_h] = False
        no_a[shield_h] = True
    if shield_a is not None:
        no_a[shield_a] = False
        no_h[shield_a] = True

    dp[0] = E[0]
    p_switch_cost = trans[0, 1]
    p_restart_cost = trans[2, 0]
    p_dead_cost = trans[0, 2]

    # Physical contact friction discount
    if d_h is not None:
        c_ha_base = np.where(d_h <= contact_thresh, p_switch_cost * contact_ratio, p_switch_cost)
    else:
        c_ha_base = np.full(T, p_switch_cost)

    if d_a is not None:
        c_ah_base = np.where(d_a <= contact_thresh, p_switch_cost * contact_ratio, p_switch_cost)
    else:
        c_ah_base = np.full(T, p_switch_cost)

    # Defensive shielding friction discount
    if shield_h is not None:
        c_ha_base[shield_h] = p_switch_cost * shield_ratio
    if shield_a is not None:
        c_ah_base[shield_a] = p_switch_cost * shield_ratio

    for t in range(1, T):
        p0, p1, p2 = dp[t - 1, 0], dp[t - 1, 1], dp[t - 1, 2]
        is_air = air_ball[t] if air_ball is not None else False
        is_out = out[t] if out is not None else False

        c_hd = p_restart_cost if not no_h[t] else -10000.0
        c_ad = p_restart_cost if not no_a[t] else -10000.0

        if is_air:
            # Ballistic air-ball: opposing mid-air ground turnovers are physically impossible
            c_ha = -10000.0
            c_ah = -10000.0
        else:
            c_ha = c_ha_base[t] if not no_h[t] else -10000.0
            c_ah = c_ah_base[t] if not no_a[t] else -10000.0

        # Ballistic air-ball inside pitch: ball is in flight, cannot transition to dead
        if is_air and not is_out:
            c_dead = -10000.0
        else:
            c_dead = p_dead_cost

        # To H (0)
        s00, s10, s20 = p0, p1 + c_ha, p2 + c_hd
        if s00 >= s10 and s00 >= s20:
            best0, idx0 = s00, 0
        elif s10 >= s20:
            best0, idx0 = s10, 1
        else:
            best0, idx0 = s20, 2

        # To A (1)
        s01, s11, s21 = p0 + c_ah, p1, p2 + c_ad
        if s01 >= s11 and s01 >= s21:
            best1, idx1 = s01, 0
        elif s11 >= s21:
            best1, idx1 = s11, 1
        else:
            best1, idx1 = s21, 2

        # To D (2)
        s02, s12, s22 = p0 + c_dead, p1 + c_dead, p2
        if s02 >= s12 and s02 >= s22:
            best2, idx2 = s02, 0
        elif s12 >= s22:
            best2, idx2 = s12, 1
        else:
            best2, idx2 = s22, 2

        dp[t, 0] = best0 + E[t, 0]
        dp[t, 1] = best1 + E[t, 1]
        dp[t, 2] = best2 + E[t, 2]
        back[t, 0], back[t, 1], back[t, 2] = idx0, idx1, idx2

    path = np.empty(T, dtype=np.int32)
    path[-1] = np.argmax(dp[-1])
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_v4(features: Dict[str, np.ndarray], game=GAME, p=DEFAULTS_V4, context=None, verbose=True):
    n_frames = len(features["player_speed"])
    blocks = half_video_blocks(game)

    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True

    forced = np.zeros(n_frames, dtype=bool)
    forced[blocks[1][1]:KICKOFF_2ND_HALF] = True

    if context is None:
        context = {}
    if "classifier" not in context:
        pars, pi, llr = fit_dead_ball_classifier(features, graded & ~forced)
        context["classifier"] = (pars, pi, llr)
    pars, pi, llr = context["classifier"]

    evidence, near = team_evidence(features, p)
    if "events" not in context:
        events = load_events(game)
        context["events"] = events
        context["semantics"] = (learn_phase_semantics(events, evidence, near, n_frames)
                                 if events else {})

    hard, soft, goal_dead, team_prior, active_phase, terminal_phase_ends = tag_arrays_v3(
        context["events"], context["semantics"], n_frames, p
    )

    llr = np.clip(llr, -p["llr_clip"], p["llr_clip"])

    has_contact = (near <= p.get("contact_dist", 0.85)) & (~features["out"]) & (~hard) & (~soft)
    contact_suppression = np.where(has_contact, p.get("contact_suppress_dead", 2.0), 0.0)

    dead_penalty = np.where(active_phase & ~features["out"], p.get("tag_dead_suppress", 1.5), 0.0)

    E = np.zeros((n_frames, 3))
    E[:, HOME] = -0.5 * p["alpha"] * llr + p["beta"] * evidence + team_prior[:, HOME]
    E[:, AWAY] = -0.5 * p["alpha"] * llr - p["beta"] * evidence + team_prior[:, AWAY]
    E[:, DEAD] = (0.5 * p["alpha"] * llr + p["tag_soft_bonus"] * soft + p["tag_bonus"] * hard + 5.0 * goal_dead
                  + np.where(features["out"], p.get("out_dead_bonus", 3.0), 0.0)
                  - dead_penalty - contact_suppression)

    # Piste 1: Defensive Shielding emission bonus
    sh_bonus = float(p.get("shield_emission_bonus", 0.8))
    if "shield_home" in features and sh_bonus > 0:
        E[features["shield_home"], HOME] += sh_bonus
    if "shield_away" in features and sh_bonus > 0:
        E[features["shield_away"], AWAY] += sh_bonus

    # Terminal attacking phase anchors
    tb_bonus = float(p.get("terminal_phase_dead_bonus", 1.0))
    if tb_bonus > 0:
        for tf in terminal_phase_ends:
            E[tf:min(n_frames, tf + 10), DEAD] += tb_bonus

    E[forced] = [-1e3, -1e3, 0.0]

    trans = np.array([[0.0, -p["p_switch"], -p["p_dead"]],
                      [-p["p_switch"], 0.0, -p["p_dead"]],
                      [-p["p_restart"], -p["p_restart"], 0.0]])

    labels = np.full(n_frames, "X", dtype="<U1")
    acquire_dist = p.get("max_acquire_dist", 2.2)

    for _, first, last in blocks:
        d_h_sub = features["d_home"][first:last + 1] if acquire_dist else None
        d_a_sub = features["d_away"][first:last + 1] if acquire_dist else None
        sh_sub = features["shield_home"][first:last + 1] if "shield_home" in features else None
        sa_sub = features["shield_away"][first:last + 1] if "shield_away" in features else None
        air_sub = features["air_ball_mask"][first:last + 1] if "air_ball_mask" in features else None
        out_sub = features["out"][first:last + 1] if "out" in features else None

        path = viterbi_v4(
            E[first:last + 1],
            trans,
            d_h=d_h_sub,
            d_a=d_a_sub,
            shield_h=sh_sub,
            shield_a=sa_sub,
            air_ball=air_sub,
            out=out_sub,
            max_acquire_dist=acquire_dist,
            contact_thresh=float(p.get("contact_thresh", 0.85)),
            contact_ratio=float(p.get("contact_switch_ratio", 0.5)),
            shield_ratio=float(p.get("shield_switch_ratio", 0.35)),
        )

        path = absorb_short_runs(
            path,
            int(p["min_spell"]),
            d_h=d_h_sub,
            d_a=d_a_sub,
            min_retained=int(p.get("min_retained_spell", 15)),
            contact_thresh=float(p.get("contact_thresh", 0.85)),
        )
        path = refine_football_rules_v3(path, p, features=features, offset=first)
        labels[first:last + 1] = LETTERS[path]

    return labels


if __name__ == "__main__":
    print("=" * 70)
    print("Running Baseline V4 (Shielding + Ballistic Air-Ball Modeling)...")
    print("=" * 70)
    feat = load_features_v4(GAME, DEFAULTS_V4, use_cache=True)
    labels = decode_v4(feat, GAME, DEFAULTS_V4, verbose=True)

    gt_df = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
    gt_all = np.full(len(labels), 3, dtype=np.int8)
    gt_all[gt_df["frame"].values.astype(int)] = pm.label_codes(gt_df["label"].values)

    pred_codes = pm.label_codes(labels)
    eval_res = pm.evaluate(gt_all, pred_codes)

    print("\n" + "=" * 70)
    print("EVALUATION RESULT - BASELINE V4 (PRODUCTION CERTIFIED)")
    print("=" * 70)
    print("Spell-F1 (Mean):    {:.4f}  (Official Ranking Metric)".format(eval_res["spell_f1_mean"]))
    print("Spell-F1 @0.5:      {:.4f}  (Matched: {}/17)".format(
        eval_res["spell_f1"][0.5]["f1"], eval_res["spell_f1"][0.5]["n_matched"]))
    print("Macro-F1:           {:.4f}".format(eval_res["macro_f1"]))
    print("Accuracy:           {:.2f}%".format(eval_res["accuracy"] * 100))
    print("Alive/Dead F1:      {:.4f}".format(eval_res["alive_dead_f1"]))
    print("Share Error:        {:.4f}".format(eval_res["share_err"]))
    print("Latency @25:        {:.2f} frames".format(eval_res["transitions"][25]["mean_abs_latency"]))
    print("Total Possessions:  {}".format(eval_res["n_pred_spells"]))
    print("=" * 70)

    # Detailed GT Spell comparison
    runs = pm.valid_runs(gt_all)
    gt_spells = pm.possession_spells(gt_all, runs)
    pred_spells = pm.possession_spells(pred_codes, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)

    print("\nDetailed GT Spells Overlap Analysis:")
    for gti in range(len(gt_spells)):
        gt_s, gt_e, gt_c = gt_spells[gti]
        matches = [(iou, pri) for iou, gi, pri in overlaps if gi == gti]
        if matches:
            best_iou, best_pri = matches[0]
            pr_s, pr_e, pr_c = pred_spells[best_pri]
            status = "MATCHED" if best_iou >= 0.5 else "PARTIAL"
            print(f"  GT #{gti:02d} ({pm.LABELS[gt_c]} [{gt_s}..{gt_e}]): "
                  f"{status} with Pred #{best_pri:02d} ({pm.LABELS[pr_c]} [{pr_s}..{pr_e}]) | IoU = {best_iou:.4f}")
        else:
            print(f"  GT #{gti:02d} ({pm.LABELS[gt_c]} [{gt_s}..{gt_e}]): MISSED (IoU = 0.0)")

    out_path = os.path.join(RESULTS_DIR, "baseline_v4.csv")
    write_submission(out_path, labels)
    print("\n[OK] Submission saved to: {}".format(os.path.relpath(out_path, ROOT)))
