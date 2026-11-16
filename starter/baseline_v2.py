"""baseline_v2: Hidden Markov Model with multimodal dead-ball detection and football possession logic.

This model treats possession as continuous temporal intervals (spells) rather than individual frames:
  1. HMM / Viterbi decoding over three states (Home, Away, Dead ball).
  2. Unsupervised 3-signal dead-ball classifier (collective player speed, ball speed, pitch boundary).
  3. Smart Tagging integration (set-pieces, goal celebrations, learned phase-possession semantics).
  4. Kinematic pass damping: fast-moving balls in flight do not trigger false turnovers.
  5. Football-aware spell refinement: removes spurious dead-ball placements, pre-stoppage touches, and micro-flickers.
  6. Pitch boundary & out-of-bounds handling: suppresses false possession and asserts dead-ball prior outside pitch lines.
  7. Official FAQ rules: frames 110,808 to 113,741 (pre-2nd-half kickoff) forced to Dead ball (D).

Usage:
    python starter/baseline_v2.py                 # writes evaluation/results/baseline_v2.csv
    python starter/baseline_v2.py --diagnose      # label-free inspection of results CSVs
    python starter/baseline_v2.py --sweep         # parameter sweep
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, DATA_DIR, half_video_blocks, require, source_frames  # noqa: E402
from baseline import team_of_each_track, team_names, write_submission  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
CACHE_PATH = os.path.join(DATA_DIR, ".baseline_v2_features.npz")

HOME, AWAY, DEAD = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])
FPS = 25

# Second-half kickoff: graded frames before it are forced dead ball (DATA.md / FAQ.md)
KICKOFF_2ND_HALF = 113742

SIDE_OF_CLUB = (("barcel", HOME), ("atl", AWAY), ("ath", AWAY))

# Priors and football-scale parameters
DEFAULTS = dict(
    max_gap=12,         # interpolate ball gaps up to 0.5 s
    speed_win=25,       # collective player speed smoothing (1.0 s)
    ball_win=25,        # ball speed window (1.0 s)
    radius=2.5,         # control radius (metres)
    lean=2.0,           # distance margin saturating team evidence (metres)
    beta=1.4,           # weight of ball-proximity evidence
    alpha=0.35,         # weight of dead-ball classifier
    llr_clip=5.0,       # log-likelihood ratio clipping
    p_switch=15.0,      # transition cost for H <-> A turnover
    p_dead=25.0,        # transition cost for entering D (calibrated for exact spell transition timing)
    p_restart=15.0,     # transition cost for leaving D
    tag_pre=3.0,        # seconds before set-piece tag start with strong D bonus
    tag_bonus=0.5,      # bonus weight for set piece pre-roll
    tag_soft_pre=15.0,  # seconds before set-piece tag start with faint D bonus
    tag_soft_bonus=0.1,
    tag_team=0.05,      # weight of learned tactical phase priors
    tag_dead_suppress=1.0, # dead-ball penalty during active tactical phases
    out_dead_bonus=2.0, # prior bonus for ball detected outside pitch boundaries
    terminal_phase_dead_bonus=2.0, # dead-ball bonus at terminal boundary of attacking phases
    min_spell=50,       # frames: minimum length for isolated possession runs (2.0 s)
    max_acquire_dist=1.8, # metres: turnover requires acquiring player <= 1.8m from ball (flight persistence guard)
    rule1_max=45,       # frames: max duration for pre-stoppage touch absorption (1.8 s)
    rule2_max=90,       # frames: max duration for inter-stoppage placement absorption (3.6 s)
    rule3_max=15,       # frames: max duration for brief glitch bridge (0.6 s)
)


# ---------------------------------------------------------------------------------------------
# 1. ATD Reading & Feature Extraction
# ---------------------------------------------------------------------------------------------
def read_atd(raw_path, n_frames, n_slots):
    """Raw ATD -> arrays. Format: '<frame_id>:<px,py,s;...;px,py,s>:<bx,by,bs>', NaN when missing."""
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
    return px, py, ps, ball[:, 0], ball[:, 1]


def interpolate_short_gaps(x, y, max_gap):
    """Linear interpolation of missing ball positions across gaps <= max_gap frames."""
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
    """Interpolate NaNs, then compute centred moving average over window frames."""
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
    """Compute and return per-frame tracking features."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    n_frames = source_frames("tactical", game)
    team = team_of_each_track(meta_path)
    length, width = GAMES[game]["pitch"]
    home, away = team_names(meta_path)
    print("teams: H = {}, A = {}".format(home, away))
    print("reading {} ...".format(os.path.basename(raw_path)))

    px, py, ps, bx, by = read_atd(raw_path, n_frames, len(team))
    seen = ~np.isnan(bx)
    out = seen & ((bx < -0.005) | (bx > 1.005) | (by < -0.005) | (by > 1.005))
    bx_i, by_i, interpolated = interpolate_short_gaps(bx, by, int(p["max_gap"]))

    dx = px * length - (bx_i * length)[:, None]
    dy = py * width - (by_i * width)[:, None]
    dist = np.sqrt(dx * dx + dy * dy)
    dist[np.isnan(dist)] = np.inf

    # Dynamic Goalkeeper identity swap correction across halves:
    # Metrica ATD tracks are tracklets rather than fixed player IDs.
    # In 1st half: Barcelona (HOME) defends x < 0.15; Athletic Club (AWAY) defends x > 0.85.
    # In 2nd half: Barcelona (HOME) defends x > 0.85; Athletic Club (AWAY) defends x < 0.15.
    # Goalkeeper tracklets (slots 23 and 45) appearing in the goal box they defend are assigned to the defending team.
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
    player_speed = smooth_filled(median_speed, int(p["speed_win"]))

    h = int(p["ball_win"]) // 2
    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)
    if h > 0 and n_frames > 2 * h:
        step_x = (bx_i[2 * h:] - bx_i[:-2 * h]) * length
        step_y = (by_i[2 * h:] - by_i[:-2 * h]) * width
        ball_speed[h:n_frames - h] = np.sqrt(step_x ** 2 + step_y ** 2) / (2.0 * h / FPS)

    print("  ball seen on {:.1f}% of frames, ball speed measurable on {:.1f}%".format(
        100.0 * seen.mean(), 100.0 * np.isfinite(ball_speed).mean()))
    return dict(d_home=d_home.astype(np.float32), d_away=d_away.astype(np.float32),
                interpolated=interpolated, out=out, player_speed=player_speed,
                ball_speed=ball_speed)


def load_features(game, p, use_cache=True):
    """Load cached features if settings match, otherwise build and cache."""
    key = np.array([p["max_gap"], p["speed_win"], p["ball_win"], 3.0], dtype=float)
    if use_cache and os.path.isfile(CACHE_PATH):
        cached = np.load(CACHE_PATH)
        if "key" in cached.files and np.array_equal(cached["key"], key):
            print("features from cache {}".format(os.path.relpath(CACHE_PATH, ROOT)))
            return {k: cached[k] for k in cached.files if k != "key"}
    features = build_features(game, p)
    if use_cache:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        np.savez_compressed(CACHE_PATH, key=key, **features)
    return features


# ---------------------------------------------------------------------------------------------
# 2. Multimodal Dead-Ball Classifier (Unsupervised EM)
# ---------------------------------------------------------------------------------------------
def _gauss(x, mu, sd):
    return -0.5 * ((x - mu) / sd) ** 2 - np.log(sd)


def _class_loglik(par, player, lball, out):
    ll = _gauss(player, par["mu_p"], par["sd_p"])
    seen = np.isfinite(lball)
    ll = ll + np.where(seen, _gauss(np.where(seen, lball, 0.0), par["mu_b"], par["sd_b"]), 0.0)
    return ll + np.where(out, np.log(par["q_out"]), np.log(1.0 - par["q_out"]))


def fit_dead_ball_classifier(features, fit_mask, iterations=60):
    """Learns two speed/pitch regimes (stopped / playing) via EM without any ground truth labels."""
    player = features["player_speed"].astype(np.float64)
    lball = np.log1p(features["ball_speed"].astype(np.float64))
    out = features["out"]
    idx = np.flatnonzero(fit_mask)
    P, B, O = player[idx], lball[idx], out[idx]
    seen = np.isfinite(B)

    r = np.where(P <= np.percentile(P, 35.0), 0.9, 0.1)
    pars = [None, None]
    for _ in range(iterations):
        for k, w in enumerate((r, 1.0 - r)):
            mu_p = (w * P).sum() / w.sum()
            sd_p = np.sqrt((w * (P - mu_p) ** 2).sum() / w.sum()).clip(1e-3)
            ws = w[seen]
            mu_b = (ws * B[seen]).sum() / ws.sum()
            sd_b = np.sqrt((ws * (B[seen] - mu_b) ** 2).sum() / ws.sum()).clip(1e-3)
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
# 3. Smart Tagging Priors & Team Evidence
# ---------------------------------------------------------------------------------------------
def _side(team_tag):
    team_tag = (team_tag or "").strip().lower()
    for prefix, side in SIDE_OF_CLUB:
        if team_tag.startswith(prefix):
            return side
    return None


def load_events(game):
    try:
        from load_smart_tagging import load
        return load(game)
    except (IOError, ImportError):
        return None


def team_evidence(features, p):
    """Computes proximity-based H/A evidence with kinematic pass damping."""
    d_home = features["d_home"].astype(np.float64)
    d_away = features["d_away"].astype(np.float64)
    near = np.minimum(d_home, d_away)
    none = np.isinf(near)
    lean = np.clip(np.where(np.isinf(d_away), 50.0, d_away) - np.where(np.isinf(d_home), 50.0, d_home), -50.0, 50.0)

    # Proximity weight: falls off as distance exceeds control radius
    weight = np.exp(-0.5 * (np.where(none, 1e3, near) / p["radius"]) ** 2)
    weight = np.where(features["interpolated"], 0.5 * weight, weight)
    weight = np.where(features["out"], 0.0, weight)  # ball outside pitch cannot be in live possession

    # Kinematic pass damping: fast-moving balls in flight (> 4 m/s) do not trigger false turnovers
    bs = features["ball_speed"]
    has_bs = np.isfinite(bs)
    fast_damping = np.ones_like(weight)
    fast_damping[has_bs] = np.clip(1.0 - (bs[has_bs] - 4.0) / 6.0, 0.35, 1.0)
    weight = weight * fast_damping

    return np.where(none, 0.0, weight * np.tanh(lean / p["lean"])), near


def learn_phase_semantics(events, evidence, near, n_frames, verbose=True):
    """Learns whether phase Team tag indicates current possessor or defending opponent."""
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
    if verbose:
        print("\nphase tags: does Team name the team on the ball? (learned from tracking, no labels)")
        print("  {:<30} {:>8} {:>10}   decision".format("code", "frames", "agreement"))

    for code, spans in sorted(by_code.items()):
        hit = tot = 0
        for a, b, side in spans:
            ok = (near[a:b] < 4.0) & (np.abs(evidence[a:b]) > 0.2)
            tracked = np.where(evidence[a:b] > 0, HOME, AWAY)[ok]
            hit += int((tracked == side).sum())
            tot += int(ok.sum())
        agreement = hit / tot if tot else float("nan")
        sign = 0 if not np.isfinite(agreement) or tot < 500 else (1 if agreement > 0.58 else (-1 if agreement < 0.42 else 0))
        semantics[code] = (sign, spans)
        if verbose:
            print("  {:<30} {:>8} {:>10.3f}   {}".format(
                code, tot, agreement, {1: "tag = team on the ball", -1: "tag = the opponent", 0: "ignored"}[sign]))
    return semantics


def tag_arrays(events, semantics, n_frames, p):
    """Builds dead-ball priors (set pieces + goals), active play indicators, team direction priors,
    and terminal phase stoppage anchors from Smart Tagging."""
    hard = np.zeros(n_frames, dtype=bool)
    soft = np.zeros(n_frames, dtype=bool)
    goal_dead = np.zeros(n_frames, dtype=bool)
    active_phase = np.zeros(n_frames, dtype=bool)
    team_prior = np.zeros((n_frames, 2))
    terminal_stoppage_frames = set()

    if events is None:
        return hard, soft, goal_dead, team_prior, active_phase, terminal_stoppage_frames

    for e in events:
        code = (e["code"] or "").strip().lower()
        s = e["start_frame"]
        end = e["end_frame"]
        sp_type = (e.get("tags", {}).get("Type") or "").strip().lower()

        if code == "set pieces":
            # Type-differentiated pre-roll duration:
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
            # Goals: dead ball during entire celebration up to kickoff after goal
            goal_dead[max(0, s):min(n_frames, end)] = True
        elif code in ("progression", "build up", "attacking transition", "long ball"):
            active_phase[max(0, s):min(n_frames, end)] = True

        # Terminal boundary of attacking phases: when an attack ends, it frequently marks stoppage or turnover
        if code in ("attacking transition", "progression", "build up"):
            if end < n_frames:
                terminal_stoppage_frames.add(end)

    for sign, spans in semantics.values():
        for a, b, side in spans:
            if sign != 0:
                possessor = side if sign > 0 else 1 - side
                team_prior[a:b, possessor] += p["tag_team"]

    return hard, soft, goal_dead, team_prior, active_phase, terminal_stoppage_frames


# ---------------------------------------------------------------------------------------------
# 4. Decoding & Football Rule Refinements
# ---------------------------------------------------------------------------------------------
def viterbi(E, trans, d_h=None, d_a=None, max_acquire_dist=None):
    """Viterbi decoding for emission scores E[T, 3] and transition log-penalties trans[from, to].
    
    When d_h and d_a are provided with max_acquire_dist, enforces football physics:
    a team cannot acquire live possession (turnover or restart) while its nearest player
    is > max_acquire_dist from the ball, protecting flying balls, passes, and clearances
    from spurious mid-air possession flips.
    """
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

    for t in range(1, T):
        p0, p1, p2 = dp[t - 1, 0], dp[t - 1, 1], dp[t - 1, 2]
        c_ha = p_switch_cost if not no_h[t] else -10000.0
        c_hd = p_restart_cost if not no_h[t] else -10000.0
        c_ah = p_switch_cost if not no_a[t] else -10000.0
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


def absorb_short_runs(path, min_len):
    """Absorbs short H/A runs (< min_len frames) touching another H/A possession."""
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
            if path[a] == DEAD or b - a >= min_len:
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


def refine_football_rules(path, p, features=None, offset=0):
    """Refines state sequence using domain rules of association football:

    1. Pre-stoppage spurious turnover: an opponent touching/picking up the dead ball right before
       a stoppage (A -> H -> D with short H) is absorbed into D.
    2. Isolated touch between stoppages (D -> H/A -> D) is absorbed into D.
    3. Micro-interruption during active possession (H -> D -> H) is bridged as continuous play.
    4. Set-piece / goal-kick placement: a dead-ball spell cannot exit into live possession while
       the ball is stationary on the turf (< 2.5 m/s) and player speed is low (< 2.2 m/s).
    """
    path = path.copy()
    edges = np.flatnonzero(np.diff(path)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [len(path)]))

    rule1_max = int(p.get("rule1_max", 45))
    rule2_max = int(p.get("rule2_max", 90))
    rule3_max = int(p.get("rule3_max", 15))

    for i in range(1, len(starts) - 1):
        s_prev, e_prev = starts[i - 1], ends[i - 1]
        s_cur, e_cur   = starts[i], ends[i]
        s_next, e_next = starts[i + 1], ends[i + 1]

        c_prev, c_cur, c_next = path[s_prev], path[s_cur], path[s_next]
        dur = e_cur - s_cur

        # 1. Spurious touch before whistle/dead ball: A -> H (< rule1_max) -> D => D
        if (c_prev in (HOME, AWAY)) and (c_cur in (HOME, AWAY)) and (c_cur != c_prev) and (c_next == DEAD):
            if dur < rule1_max:
                path[s_cur:e_cur] = DEAD

        # 2. Deflection or placement between stoppages: D -> H/A (< rule2_max) -> D => D
        # A player retrieving an out-of-play ball and placing it for a goal kick or free kick is dead-ball handling
        if (c_prev == DEAD) and (c_cur in (HOME, AWAY)) and (c_next == DEAD):
            if dur < rule2_max:
                path[s_cur:e_cur] = DEAD

        # 3. Brief tracking glitch / false stoppage inside possession: H -> D (< rule3_max) -> H => H
        if (c_prev in (HOME, AWAY)) and (c_cur == DEAD) and (c_next == c_prev):
            if dur < rule3_max:
                path[s_cur:e_cur] = c_prev

    # 4. Set-piece / goal kick restart placement:
    # A dead ball cannot exit into live possession while the ball is stationary on the turf
    # (< 2.5 m/s) and player speed is low (< 2.2 m/s). It only begins when the kick is taken.
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
                if np.isfinite(curr_bs) and curr_bs > 2.5:
                    break
                if ps[f_now] > 2.2:
                    break
                k += 1
            if k > 0 and k < max_k:
                path[s_idx:s_idx + k] = DEAD

    return path


def decode(features, game, p, context=None, verbose=True):
    """Full-match decoding producing 'H' / 'A' / 'D' / 'X' array."""
    n_frames = len(features["player_speed"])
    blocks = half_video_blocks(game)
    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    forced = np.zeros(n_frames, dtype=bool)
    forced[blocks[1][1]:KICKOFF_2ND_HALF] = True  # graded frames before 2nd-half kickoff: all D

    if context is None:
        context = {}
    if "classifier" not in context:
        pars, pi, llr = fit_dead_ball_classifier(features, graded & ~forced)
        context["classifier"] = (pars, pi, llr)
        if verbose:
            print("dead-ball classifier (unsupervised): P(stopped) = {:.2f}".format(pi))
            for name, par in zip(("stopped", "playing"), pars):
                print("  {:<8} players {:.2f} +/- {:.2f} m/s | ball {:.2f} log-m/s | outside-pitch rate {:.3f}".format(
                    name, par["mu_p"], par["sd_p"], par["mu_b"], par["q_out"]))
    pars, pi, llr = context["classifier"]

    evidence, near = team_evidence(features, p)
    if "events" not in context:
        events = load_events(game)
        context["events"] = events
        context["semantics"] = (learn_phase_semantics(events, evidence, near, n_frames, verbose)
                                 if events else {})
    hard, soft, goal_dead, team_prior, active_phase, terminal_phase_ends = tag_arrays(
        context["events"], context["semantics"], n_frames, p
    )

    llr = np.clip(llr, -p["llr_clip"], p["llr_clip"])
    dead_penalty = np.where(active_phase & ~features["out"], p.get("tag_dead_suppress", 1.0), 0.0)

    E = np.zeros((n_frames, 3))
    E[:, HOME] = -0.5 * p["alpha"] * llr + p["beta"] * evidence + team_prior[:, HOME]
    E[:, AWAY] = -0.5 * p["alpha"] * llr - p["beta"] * evidence + team_prior[:, AWAY]
    E[:, DEAD] = (0.5 * p["alpha"] * llr + p["tag_soft_bonus"] * soft + p["tag_bonus"] * hard + 5.0 * goal_dead
                  + np.where(features["out"], p.get("out_dead_bonus", 2.0), 0.0) - dead_penalty)

    # Terminal attacking phase anchors: slight prior towards DEAD at the exact transition frame
    tb_bonus = float(p.get("terminal_phase_dead_bonus", 2.0))
    if tb_bonus > 0:
        for tf in terminal_phase_ends:
            E[max(0, tf - 5):min(n_frames, tf + 10), DEAD] += tb_bonus

    E[forced] = [-1e3, -1e3, 0.0]

    trans = np.array([[0.0, -p["p_switch"], -p["p_dead"]],
                      [-p["p_switch"], 0.0, -p["p_dead"]],
                      [-p["p_restart"], -p["p_restart"], 0.0]])

    labels = np.full(n_frames, "X", dtype="<U1")
    acquire_dist = p.get("max_acquire_dist")
    for _, first, last in blocks:
        d_h_sub = features["d_home"][first:last + 1] if acquire_dist else None
        d_a_sub = features["d_away"][first:last + 1] if acquire_dist else None
        path = viterbi(E[first:last + 1], trans, d_h=d_h_sub, d_a=d_a_sub, max_acquire_dist=acquire_dist)
        path = absorb_short_runs(path, int(p["min_spell"]))
        path = refine_football_rules(path, p, features=features, offset=first)
        labels[first:last + 1] = LETTERS[path]
    return labels


# ---------------------------------------------------------------------------------------------
# 5. Diagnostics & CLI
# ---------------------------------------------------------------------------------------------
def possession_stats(labels, game=GAME):
    code = np.full(len(labels), -1)
    for i, letter in enumerate("HAD"):
        code[labels == letter] = i
    spells = []
    for team in (HOME, AWAY):
        m = np.concatenate(([False], code == team, [False]))
        starts = np.flatnonzero(~m[:-1] & m[1:])
        ends = np.flatnonzero(m[:-1] & ~m[1:])
        spells.extend((ends - starts) / FPS)
    sp = np.array(spells) if spells else np.array([0.0])
    graded = np.zeros(len(labels), dtype=bool)
    for _, first, last in half_video_blocks(game):
        graded[first:last + 1] = True
    d_share = float((labels[graded] == "D").mean())
    return dict(possessions=len(spells),
                mean_s=float(sp.mean()),
                median_s=float(np.median(sp)),
                under_2s=int((sp < 2.0).sum()),
                dead_share=d_share)


def summary_header():
    return "{:<22} {:>5} {:>7} {:>7} {:>5} {:>7}".format(
        "run", "poss", "mean s", "med s", "<2 s", "D share")


def summary_row(name, labels, game):
    s = possession_stats(labels, game)
    return "{:<22} {:>5} {:>7.1f} {:>7.1f} {:>5} {:>7.3f}".format(
        name, s["possessions"], s["mean_s"], s["median_s"], s["under_2s"], s["dead_share"])


def diagnose(csv_paths):
    print("\n" + summary_header())
    for path in csv_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            arr = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=1, dtype="<U1")
            print(summary_row(name, arr, GAME))
        except Exception as e:
            print("{:<22} error: {}".format(name, e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--diagnose", nargs="*", metavar="CSV")
    parser.add_argument("--sweep", action="store_true", help="parameter sweep")
    parser.add_argument("--name", default="baseline_v2")
    parser.add_argument("--out", default=RESULTS_DIR)
    parser.add_argument("--no-cache", action="store_true")
    for key, value in DEFAULTS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args()

    if args.diagnose is not None:
        diagnose(args.diagnose or sorted(
            os.path.join(args.out, f) for f in os.listdir(args.out) if f.endswith(".csv")))
        sys.exit(0)

    params = {k: getattr(args, k) for k in DEFAULTS}
    features = load_features(GAME, params, use_cache=not args.no_cache)

    if args.sweep:
        context = {}
        print("\n" + summary_header())
        for alpha in (0.25, 0.35, 0.45):
            for p_dead in (25.0, 30.0, 40.0):
                trial = dict(params, alpha=alpha, p_dead=p_dead)
                labels = decode(features, GAME, trial, context, verbose=(not context))
                print(summary_row("alpha={} p_dead={:g}".format(alpha, p_dead), labels, GAME))
        sys.exit(0)

    labels = decode(features, GAME, params)
    out_path = os.path.join(args.out, args.name + ".csv")
    path = write_submission(out_path, labels)
    print("\n" + summary_header())
    print(summary_row(args.name, labels, GAME))
    print("\n{} -> {}".format(args.name, os.path.relpath(path, ROOT)))
    print("Next:  python evaluation/validate_submission.py {}".format(os.path.relpath(path, ROOT)))
