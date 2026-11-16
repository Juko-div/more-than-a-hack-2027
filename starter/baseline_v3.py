"""Baseline V3: Multi-Stage Cascading Possession Pipeline.

Architecture in 4 sequential stages (balayage étagé):
  Stage 1: Hard structural & pitch-boundary masking (X and certain out-of-bounds D).
  Stage 2: Event-anchored dead-ball regime with Contact Override (prohibiting false D during shielding).
  Stage 3: Live-play possession decoding with bounded phase priors (tags guide, never lock out transitions).
  Stage 4: Domain boundary refinement (calibrated restarts, reflex clearance absorption).

Certified evaluation against public GT: 7500 frames (11185..18684).
Match-wide inference across 199,319 frames.
"""

from __future__ import division, print_function

import argparse
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from games import GAME, GAMES, DATA_DIR, half_video_blocks, require, source_frames  # noqa: E402
from baseline import team_of_each_track, team_names, write_submission  # noqa: E402
from load_smart_tagging import load as load_events  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
CACHE_FILE = os.path.join(ROOT, "data", ".baseline_v3_features.npz")

HOME, AWAY, DEAD, NOBODY = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])
FPS = 25.0

KICKOFF_2ND_HALF = 113742
SIDE_OF_CLUB = (("barcel", HOME), ("atl", AWAY), ("ath", AWAY))

DEFAULTS_V3 = dict(
    # Stage 1 & 2: Structural & Dead-Ball
    max_gap=12,
    speed_win=25,
    ball_win=25,
    alpha=0.35,
    llr_clip=5.0,
    out_dead_bonus=3.0,
    contact_suppress_dead=0.0,   # Set to 0.0 by default to prevent false merging of sequential attacks
    contact_dist=0.85,           # Distance defining player contact shielding the ball
    # Set pieces pre-rolls
    tag_pre=3.0,
    tag_bonus=0.5,
    tag_soft_pre=15.0,
    tag_soft_bonus=0.1,
    tag_dead_suppress=1.5,
    terminal_phase_dead_bonus=1.0, # Moderated: marks turnover/stoppage without choking transitions
    # Stage 3: Live Possession Dynamics
    radius=2.5,
    lean=2.0,
    beta=1.4,
    tag_team_max=0.08,           # Upper bound for phase tags: cannot overpower proximity
    p_switch=14.0,               # Transition cost H <-> A
    p_dead=24.0,                 # Transition cost entering D
    p_restart=14.0,              # Transition cost leaving D
    max_acquire_dist=2.2,        # Metres: turnover requires player <= 2.2m (avoids mid-air flips)
    contact_switch_ratio=0.5,    # Transition friction discount when tackling directly (d <= contact_thresh)
    contact_thresh=0.85,         # Direct contact proximity threshold
    min_retained_spell=15,       # Frames: minimum duration for a verified physical contact turnover (0.6s)
    # Stage 4: Refinement
    min_spell=35,                # Minimum frames for isolated possession (1.4s)
    rule1_max=60,                # Pre-stoppage reflex clearance absorption (2.4s)
    rule2_max=90,                # Placement between stoppages (3.6s)
    rule3_max=15,                # Glitch bridging (0.6s)
    restart_v_thresh=1.8,        # Ball speed threshold for set-piece kick (1.8 m/s instead of 2.5)
)


# ---------------------------------------------------------------------------------------------
# 1. Feature Extraction (Reusing validated feature pipeline)
# ---------------------------------------------------------------------------------------------
def read_atd(raw_path, n_frames, n_slots):
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
            p_entries = parts[1].split(";")
            if p_entries and p_entries[0]:
                group = np.fromstring(parts[1].replace(";", ","), sep=",", dtype=np.float32)
                group = group.reshape(-1, 3)
                n = min(len(group), n_slots)
                px[t, :n], py[t, :n], ps[t, :n] = group[:n, 0], group[:n, 1], group[:n, 2]
            b = np.fromstring(parts[2], sep=",", dtype=np.float32)
            if len(b) >= 3:
                ball[t] = b[:3]
    return px, py, ps, ball[:, 0], ball[:, 1]


def interpolate_short_gaps(x, y, max_gap):
    out_x, out_y = x.copy(), y.copy()
    seen = np.flatnonzero(~np.isnan(x))
    filled = np.zeros(len(x), dtype=bool)
    if len(seen) < 2:
        return out_x, out_y, filled
    for a, b in zip(seen[:-1], seen[1:]):
        n = b - a - 1
        if 0 < n <= max_gap:
            w = np.arange(1, n + 1, dtype=np.float32) / (n + 1)
            out_x[a + 1:b] = x[a] + (x[b] - x[a]) * w
            out_y[a + 1:b] = y[a] + (y[b] - y[a]) * w
            filled[a + 1:b] = True
    return out_x, out_y, filled


def smooth_filled(v, window):
    ok = np.flatnonzero(np.isfinite(v))
    if len(ok) == 0:
        return np.zeros(len(v), dtype=np.float32)
    v = np.interp(np.arange(len(v)), ok, v[ok])
    c = np.concatenate(([0.0], np.cumsum(v)))
    idx = np.arange(len(v))
    lo = np.maximum(0, idx - window // 2)
    hi = np.minimum(len(v), idx + window // 2 + 1)
    return ((c[hi] - c[lo]) / (hi - lo)).astype(np.float32)


def build_features(game, p):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    n_frames = source_frames("tactical", game)
    team = team_of_each_track(meta_path)
    length, width = GAMES[game]["pitch"]
    home, away = team_names(meta_path)

    px, py, ps, bx, by = read_atd(raw_path, n_frames, len(team))
    seen = ~np.isnan(bx)
    out = seen & ((bx < -0.005) | (bx > 1.005) | (by < -0.005) | (by > 1.005))
    bx_i, by_i, interpolated = interpolate_short_gaps(bx, by, int(p["max_gap"]))

    dx = px * length - (bx_i * length)[:, None]
    dy = py * width - (by_i * width)[:, None]
    dist = np.sqrt(dx * dx + dy * dy)
    dist[np.isnan(dist)] = np.inf

    team_mat = np.tile(team, (n_frames, 1))
    blocks = half_video_blocks(game)
    h1_first, h1_last = blocks[0][1], blocks[0][2]
    h1_mask = np.zeros(n_frames, dtype=bool)
    h1_mask[h1_first:h1_last + 1] = True
    is_h1_near_0 = h1_mask[:, None] & (px < 0.12) & (py > 0.25) & (py < 0.75)
    is_h1_near_1 = h1_mask[:, None] & (px > 0.88) & (py > 0.25) & (py < 0.75)
    team_mat[is_h1_near_0 & (np.arange(len(team)) == 45)[None, :]] = HOME
    team_mat[is_h1_near_1 & (np.arange(len(team)) == 23)[None, :]] = AWAY

    if len(blocks) > 1:
        h2_first, h2_last = blocks[1][1], blocks[1][2]
        h2_mask = np.zeros(n_frames, dtype=bool)
        h2_mask[h2_first:h2_last + 1] = True
        is_h2_near_0 = h2_mask[:, None] & (px < 0.12) & (py > 0.25) & (py < 0.75)
        is_h2_near_1 = h2_mask[:, None] & (px > 0.88) & (py > 0.25) & (py < 0.75)
        team_mat[is_h2_near_0 & (np.arange(len(team)) == 23)[None, :]] = AWAY
        team_mat[is_h2_near_1 & (np.arange(len(team)) == 45)[None, :]] = HOME

    d_home = np.where(team_mat == HOME, dist, np.inf).min(axis=1)
    d_away = np.where(team_mat == AWAY, dist, np.inf).min(axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median_speed = np.nanmedian(ps, axis=1)

    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)
    s = seen[:-1] & seen[1:]
    dbx = (bx[1:] - bx[:-1]) * length
    dby = (by[1:] - by[:-1]) * width
    ball_speed[:-1][s] = np.sqrt(dbx[s] * dbx[s] + dby[s] * dby[s]) * FPS

    return dict(
        key="{}_maxgap{}_sw{}_bw{}".format(game, p["max_gap"], p["speed_win"], p["ball_win"]),
        d_home=d_home.astype(np.float32),
        d_away=d_away.astype(np.float32),
        interpolated=interpolated,
        out=out,
        player_speed=smooth_filled(median_speed, int(p["speed_win"])),
        ball_speed=smooth_filled(ball_speed, int(p["ball_win"])),
    )


def load_features(game=GAME, p=DEFAULTS_V3, use_cache=True):
    if use_cache and os.path.exists(CACHE_FILE):
        try:
            cached = np.load(CACHE_FILE)
            return {k: cached[k] for k in cached.files}
        except Exception:
            pass
    feat = build_features(game, p)
    try:
        np.savez_compressed(CACHE_FILE, **feat)
    except Exception:
        pass
    return feat


# ---------------------------------------------------------------------------------------------
# 2. Dead-Ball Classifier (EM) with Contact Safeguard
# ---------------------------------------------------------------------------------------------
def _gauss(x, mu, sd):
    return -0.5 * ((x - mu) / sd) ** 2 - np.log(sd)


def _class_loglik(par, player, lball, out):
    ll = _gauss(player, par["mu_p"], par["sd_p"])
    seen = np.isfinite(lball)
    ll = ll + np.where(seen, _gauss(np.where(seen, lball, 0.0), par["mu_b"], par["sd_b"]), 0.0)
    return ll + np.where(out, np.log(par["q_out"]), np.log(1.0 - par["q_out"]))


def fit_dead_ball_classifier(features, fit_mask, iterations=60):
    player = features["player_speed"].astype(np.float64)
    lball = np.log1p(features["ball_speed"].astype(np.float64))
    out = features["out"]
    idx = np.flatnonzero(fit_mask)
    P, B, O = player[idx], lball[idx], out[idx]

    r = np.where(P <= np.percentile(P, 35.0), 0.9, 0.1)
    pars = [None, None]
    for _ in range(iterations):
        for k, w in enumerate((r, 1.0 - r)):
            mu_p = (w * P).sum() / w.sum()
            sd_p = np.sqrt((w * (P - mu_p) ** 2).sum() / w.sum()).clip(1e-3)
            ws = w[np.isfinite(B)]
            mu_b = (ws * B[np.isfinite(B)]).sum() / ws.sum()
            sd_b = np.sqrt((ws * (B[np.isfinite(B)] - mu_b) ** 2).sum() / ws.sum()).clip(1e-3)
            q = ((w * O).sum() + 0.5) / (w.sum() + 1.0)
            pars[k] = dict(mu_p=mu_p, sd_p=sd_p, mu_b=mu_b, sd_b=sd_b, q_out=float(np.clip(q, 1e-4, 0.5)))
        pi = float(r.mean())
        ll_d = _class_loglik(pars[0], P, B, O) + np.log(pi)
        ll_p = _class_loglik(pars[1], P, B, O) + np.log(1.0 - pi)
        r = 1.0 / (1.0 + np.exp(np.clip(ll_p - ll_d, -50, 50)))

    if pars[0]["mu_p"] > pars[1]["mu_p"]:
        pars.reverse()
        pi = 1.0 - pi
    llr = (_class_loglik(pars[0], player, lball, out) - _class_loglik(pars[1], player, lball, out)
           + np.log(pi / (1.0 - pi)))
    return pars, pi, llr


# ---------------------------------------------------------------------------------------------
# 3. Smart Tagging & Team Evidence with Decoupled Priority
# ---------------------------------------------------------------------------------------------
def _side(team_tag):
    team_tag = (team_tag or "").strip().lower()
    for prefix, side in SIDE_OF_CLUB:
        if team_tag.startswith(prefix):
            return side
    return None


def team_evidence(features, p):
    d_home = features["d_home"].astype(np.float64)
    d_away = features["d_away"].astype(np.float64)
    near = np.minimum(d_home, d_away)
    none = np.isinf(near)
    lean = np.clip(np.where(np.isinf(d_away), 50.0, d_away) - np.where(np.isinf(d_home), 50.0, d_home), -50.0, 50.0)

    weight = np.exp(-0.5 * (np.where(none, 1e3, near) / p["radius"]) ** 2)
    weight = np.where(features["interpolated"], 0.5 * weight, weight)
    weight = np.where(features["out"], 0.0, weight)

    bs = features["ball_speed"]
    has_bs = np.isfinite(bs)
    fast_damping = np.ones_like(weight)
    fast_damping[has_bs] = np.clip(1.0 - (bs[has_bs] - 4.0) / 6.0, 0.35, 1.0)
    weight = weight * fast_damping

    return np.where(none, 0.0, weight * np.tanh(lean / p["lean"])), near


def learn_phase_semantics(events, evidence, near, n_frames):
    by_code = {}
    for e in events:
        code = (e["code"] or "").strip().lower()
        side = _side(e["tags"].get("Team"))
        if code in ("set pieces", "goals") or side is None:
            continue
        a, b = max(0, e["start_frame"]), min(n_frames, e["end_frame"])
        if b > a:
            by_code.setdefault(code, []).append((a, b, side))

    semantics = {}
    for code, spans in by_code.items():
        votes = []
        for a, b, side in spans:
            ev = evidence[a:b]
            nr = near[a:b]
            valid = (nr < 3.0) & (np.abs(ev) > 0.05)
            if np.any(valid):
                mean_ev = ev[valid].mean()
                votes.append(mean_ev > 0 if side == HOME else mean_ev < 0)
        if votes:
            agreement = np.mean(votes)
            sign = 1 if agreement > 0.58 else (-1 if agreement < 0.42 else 0)
            semantics[code] = (sign, spans)
    return semantics


def tag_arrays_v3(events, semantics, n_frames, p):
    hard = np.zeros(n_frames, dtype=bool)
    soft = np.zeros(n_frames, dtype=bool)
    goal_dead = np.zeros(n_frames, dtype=bool)
    team_prior = np.zeros((n_frames, 2), dtype=np.float64)
    active_phase = np.zeros(n_frames, dtype=bool)
    terminal_phase_ends = set()

    for e in events:
        code = (e["code"] or "").strip().lower()
        s = e["start_frame"]
        end = e["end_frame"]
        sp_type = (e.get("tags", {}).get("Type") or "").strip().lower()

        if code == "set pieces":
            if "goal kick" in sp_type:
                pre_s = 6.0
            elif "free kick" in sp_type:
                pre_s = 5.0
            elif "throw-in" in sp_type:
                pre_s = 3.0
            else:
                pre_s = p.get("tag_pre", 3.0)

            hard[max(0, s - int(pre_s * FPS)):min(n_frames, s + int(1.0 * FPS))] = True
            soft[max(0, s - int(p["tag_soft_pre"] * FPS)):max(0, s)] = True
        elif code == "goals":
            goal_dead[max(0, s):min(n_frames, end)] = True
        elif code in ("progression", "build up", "attacking transition", "long ball"):
            active_phase[max(0, s):min(n_frames, end)] = True

        if code in ("attacking transition", "progression", "build up"):
            if end < n_frames:
                terminal_phase_ends.add(end)

    # Bounded phase prior: cannot exceed tag_team_max
    for sign, spans in semantics.values():
        for a, b, side in spans:
            if sign != 0:
                possessor = side if sign > 0 else 1 - side
                team_prior[a:b, possessor] = np.clip(
                    team_prior[a:b, possessor] + 0.05, 0.0, p.get("tag_team_max", 0.08)
                )

    return hard, soft, goal_dead, team_prior, active_phase, terminal_phase_ends


# ---------------------------------------------------------------------------------------------
# 4. Multi-Stage Decoding (Balayage Étagé)
# ---------------------------------------------------------------------------------------------
def viterbi(E, trans, d_h=None, d_a=None, max_acquire_dist=None, contact_thresh=0.85, contact_ratio=0.5):
    T = len(E)
    if d_h is None or d_a is None or max_acquire_dist is None:
        back = np.zeros((T, 3), dtype=np.int8)
        score = E[0].copy()
        for t in range(1, T):
            cand = score[:, None] + trans
            back[t] = cand.argmax(axis=0)
            score = cand[back[t], np.arange(3)] + E[t]
        path = np.zeros(T, dtype=np.int8)
        path[-1] = score.argmax()
        for t in range(T - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path

    dp = np.empty((T, 3), dtype=np.float64)
    back = np.empty((T, 3), dtype=np.int32)
    no_h = (d_h > max_acquire_dist)
    no_a = (d_a > max_acquire_dist)
    dp[0] = E[0]
    p_switch_cost = trans[0, 1]
    p_restart_cost = trans[2, 0]
    p_dead_cost = trans[0, 2]

    # Physical contact friction discount: when a player tackles directly at close range,
    # the turnover barrier is halved
    c_ha_base = np.where(d_h <= contact_thresh, p_switch_cost * contact_ratio, p_switch_cost)
    c_ah_base = np.where(d_a <= contact_thresh, p_switch_cost * contact_ratio, p_switch_cost)

    for t in range(1, T):
        p0, p1, p2 = dp[t - 1, 0], dp[t - 1, 1], dp[t - 1, 2]
        c_ha = c_ha_base[t] if not no_h[t] else -10000.0
        c_hd = p_restart_cost if not no_h[t] else -10000.0
        c_ah = c_ah_base[t] if not no_a[t] else -10000.0
        c_ad = p_restart_cost if not no_a[t] else -10000.0

        # To H (0)
        s00, s10, s20 = p0, p1 + c_ha, p2 + c_hd
        if s00 >= s10 and s00 >= s20: best0, idx0 = s00, 0
        elif s10 >= s20: best0, idx0 = s10, 1
        else: best0, idx0 = s20, 2

        # To A (1)
        s01, s11, s21 = p0 + c_ah, p1, p2 + c_ad
        if s01 >= s11 and s01 >= s21: best1, idx1 = s01, 0
        elif s11 >= s21: best1, idx1 = s11, 1
        else: best1, idx1 = s21, 2

        # To D (2)
        s02, s12, s22 = p0 + p_dead_cost, p1 + p_dead_cost, p2
        if s02 >= s12 and s02 >= s22: best2, idx2 = s02, 0
        elif s12 >= s22: best2, idx2 = s12, 1
        else: best2, idx2 = s22, 2

        dp[t, 0] = best0 + E[t, 0]
        dp[t, 1] = best1 + E[t, 1]
        dp[t, 2] = best2 + E[t, 2]
        back[t, 0], back[t, 1], back[t, 2] = idx0, idx1, idx2

    path = np.empty(T, dtype=np.int32)
    path[-1] = np.argmax(dp[-1])
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def absorb_short_runs(path, min_len, d_h=None, d_a=None, min_retained=15, contact_thresh=0.85):
    if min_len <= 1:
        return path
    path = path.copy()
    for _ in range(3):
        edges = np.flatnonzero(np.diff(path)) + 1
        starts = np.concatenate(([0], edges))
        ends = np.concatenate((edges, [len(path)]))
        changed = False
        for i in range(len(starts)):
            a, b = starts[i], ends[i]
            dur = b - a
            team_code = path[a]
            if team_code == DEAD:
                continue
            if dur >= min_len:
                continue

            # Verified physical contact retention guard:
            # If the tackling team touched the ball directly (<= contact_thresh)
            # and retained it for >= min_retained frames (e.g. 15f = 0.6s),
            # this is a genuine controlled turnover! KEEP IT!
            if d_h is not None and d_a is not None and dur >= min_retained:
                d_team = d_h[a:b] if team_code == HOME else d_a[a:b]
                if np.nanmin(d_team) <= contact_thresh:
                    continue

            left = (starts[i - 1], ends[i - 1]) if i > 0 and path[starts[i - 1]] != DEAD else None
            right = (starts[i + 1], ends[i + 1]) if i + 1 < len(starts) and path[starts[i + 1]] != DEAD else None
            options = [r for r in (left, right) if r is not None]
            if options:
                keep = max(options, key=lambda r: r[1] - r[0])
                path[a:b] = path[keep[0]]
                changed = True
        if not changed:
            break
    return path


def refine_football_rules_v3(path, p, features=None, offset=0):
    path = path.copy()
    edges = np.flatnonzero(np.diff(path)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [len(path)]))

    rule1_max = int(p.get("rule1_max", 60))
    rule2_max = int(p.get("rule2_max", 90))
    rule3_max = int(p.get("rule3_max", 15))
    restart_v = float(p.get("restart_v_thresh", 1.8))

    for i in range(1, len(starts) - 1):
        s_prev, e_prev = starts[i - 1], ends[i - 1]
        s_cur, e_cur   = starts[i], ends[i]
        s_next, e_next = starts[i + 1], ends[i + 1]

        c_prev, c_cur, c_next = path[s_prev], path[s_cur], path[s_next]
        dur = e_cur - s_cur

        # 1. Reflex defensive clearance directly out: A -> H (< rule1_max) -> D => D
        if (c_prev in (HOME, AWAY)) and (c_cur in (HOME, AWAY)) and (c_cur != c_prev) and (c_next == DEAD):
            if dur < rule1_max:
                path[s_cur:e_cur] = DEAD

        # 2. Deflection or placement between stoppages: D -> H/A (< rule2_max) -> D => D
        if (c_prev == DEAD) and (c_cur in (HOME, AWAY)) and (c_next == DEAD):
            if dur < rule2_max:
                path[s_cur:e_cur] = DEAD

        # 3. Micro-interruption during active possession: H -> D (< rule3_max) -> H => H
        if (c_prev in (HOME, AWAY)) and (c_cur == DEAD) and (c_next == c_prev):
            if dur < rule3_max:
                path[s_cur:e_cur] = c_prev

    # 4. Calibrated set-piece restart:
    # A dead ball exits into live play as soon as ball moves (v_ball > restart_v) or player sprints
    if features is not None and "ball_speed" in features and "player_speed" in features:
        bs = features["ball_speed"]
        ps = features["player_speed"]
        restart_starts = np.flatnonzero((path[:-1] == DEAD) & (path[1:] != DEAD)) + 1
        for s_idx in restart_starts:
            abs_f = offset + s_idx
            k = 0
            max_k = min(125, len(path) - s_idx)
            while k < max_k:
                f_now = abs_f + k
                curr_bs = bs[f_now]
                if np.isfinite(curr_bs) and curr_bs > restart_v:
                    break
                if ps[f_now] > 2.0:
                    break
                k += 1
            if k > 0 and k < max_k:
                path[s_idx:s_idx + k] = DEAD

    return path


def decode_v3(features, game=GAME, p=DEFAULTS_V3, context=None, verbose=True):
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

    # Stage 2: Contact Override on DEAD (Open Play Only)
    # If a player has ball at d < contact_dist (0.85m) inside pitch limits (not out)
    # AND not during a tagged set piece preparation (~hard & ~soft), suppress false dead ball!
    has_contact = (near <= p.get("contact_dist", 0.85)) & (~features["out"]) & (~hard) & (~soft)
    contact_suppression = np.where(has_contact, p.get("contact_suppress_dead", 2.0), 0.0)

    dead_penalty = np.where(active_phase & ~features["out"], p.get("tag_dead_suppress", 1.5), 0.0)

    E = np.zeros((n_frames, 3))
    # Live team emissions: proximity evidence + bounded phase prior
    E[:, HOME] = -0.5 * p["alpha"] * llr + p["beta"] * evidence + team_prior[:, HOME]
    E[:, AWAY] = -0.5 * p["alpha"] * llr - p["beta"] * evidence + team_prior[:, AWAY]
    # Dead-ball emissions
    E[:, DEAD] = (0.5 * p["alpha"] * llr + p["tag_soft_bonus"] * soft + p["tag_bonus"] * hard + 5.0 * goal_dead
                  + np.where(features["out"], p.get("out_dead_bonus", 3.0), 0.0)
                  - dead_penalty - contact_suppression)

    # Terminal attacking phase anchors
    tb_bonus = float(p.get("terminal_phase_dead_bonus", 1.0))
    if tb_bonus > 0:
        for tf in terminal_phase_ends:
            E[max(0, tf - 5):min(n_frames, tf + 10), DEAD] += tb_bonus

    E[forced] = [-1e3, -1e3, 0.0]

    trans = np.array([[0.0, -p["p_switch"], -p["p_dead"]],
                      [-p["p_switch"], 0.0, -p["p_dead"]],
                      [-p["p_restart"], -p["p_restart"], 0.0]])

    labels = np.full(n_frames, "X", dtype="<U1")
    acquire_dist = p.get("max_acquire_dist", 2.2)
    for _, first, last in blocks:
        d_h_sub = features["d_home"][first:last + 1] if acquire_dist else None
        d_a_sub = features["d_away"][first:last + 1] if acquire_dist else None
        path = viterbi(E[first:last + 1], trans, d_h=d_h_sub, d_a=d_a_sub, max_acquire_dist=acquire_dist,
                       contact_thresh=float(p.get("contact_thresh", 0.85)),
                       contact_ratio=float(p.get("contact_switch_ratio", 0.5)))
        path = absorb_short_runs(path, int(p["min_spell"]), d_h=d_h_sub, d_a=d_a_sub,
                                 min_retained=int(p.get("min_retained_spell", 15)),
                                 contact_thresh=float(p.get("contact_thresh", 0.85)))
        path = refine_football_rules_v3(path, p, features=features, offset=first)
        labels[first:last + 1] = LETTERS[path]
    return labels


if __name__ == "__main__":
    import possession_metrics as pm
    import pandas as pd

    print("Running Baseline V3 (Multi-Stage Cascading Pipeline)...")
    feat = load_features(GAME, DEFAULTS_V3, use_cache=True)
    labels = decode_v3(feat, GAME, DEFAULTS_V3, verbose=True)

    gt_df = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
    gt_all = np.full(len(labels), 3, dtype=np.int8)
    gt_all[gt_df["frame"].values.astype(int)] = pm.label_codes(gt_df["label"].values)

    pred_codes = pm.label_codes(labels)
    eval_res = pm.evaluate(gt_all, pred_codes)

    print("\n" + "=" * 60)
    print("EVALUATION RESULT - BASELINE V3")
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

    out_path = os.path.join(RESULTS_DIR, "baseline_v3.csv")
    write_submission(out_path, labels)
    print("\n[OK] Submission saved to: {}".format(os.path.relpath(out_path, ROOT)))
