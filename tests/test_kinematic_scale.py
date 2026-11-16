import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.features.atd_parser import ATDParser
from src.features.ball_trajectory import process_ball_trajectory, PITCH_LENGTH_METERS, PITCH_WIDTH_METERS
from src.features.kinematic_features import extract_kinematic_features

parser = ATDParser()
data = parser.parse('data/tactical/BAR-ATH_tactical_atd.txt', 'data/tactical/BAR-ATH_tactical_atd.xml')
bx = data.ball_pos_norm[:, 0]
by = data.ball_pos_norm[:, 1]
res = process_ball_trajectory(bx, by)

# Option A: Convert bx, by to METERS so both players and ball are in METERS
bx_m = res['bx'] * PITCH_LENGTH_METERS
by_m = res['by'] * PITCH_WIDTH_METERS

px_m = data.all_players_pos_m[:, :, 0]
py_m = data.all_players_pos_m[:, :, 1]

print("Running extract_kinematic_features with BOTH in METERS...")
feat_m = extract_kinematic_features(
    px=px_m,
    py=py_m,
    bx=bx_m,
    by=by_m,
    ball_speed=res['speed'],
    is_interpolated=res['is_interpolated'],
)

hc = feat_m['home_control']
ac = feat_m['away_control']
pt = feat_m['pass_team']

print(f"Home control frames: {hc.sum()} ({hc.mean()*100:.2f}%)")
print(f"Away control frames: {ac.sum()} ({ac.mean()*100:.2f}%)")
print(f"pass_team == 0 (Home): {(pt == 0).sum()} ({(pt == 0).mean()*100:.2f}%)")
print(f"pass_team == 1 (Away): {(pt == 1).sum()} ({(pt == 1).mean()*100:.2f}%)")
print(f"pass_team == -1: {(pt == -1).sum()} ({(pt == -1).mean()*100:.2f}%)")

# Option B: Run with BOTH in NORMALIZED [0, 1]
print("\nRunning extract_kinematic_features with BOTH in NORMALIZED...")
feat_norm = extract_kinematic_features(
    px=data.home_pos_norm[:, :, 0] if False else np.concatenate([data.home_pos_norm[:, :, 0], data.away_pos_norm[:, :, 0]], axis=1),
    py=np.concatenate([data.home_pos_norm[:, :, 1], data.away_pos_norm[:, :, 1]], axis=1),
    bx=res['bx'],
    by=res['by'],
    ball_speed=res['speed'],
    is_interpolated=res['is_interpolated'],
)

hc_n = feat_norm['home_control']
ac_n = feat_norm['away_control']
pt_n = feat_norm['pass_team']

print(f"Home control frames (norm): {hc_n.sum()} ({hc_n.mean()*100:.2f}%)")
print(f"Away control frames (norm): {ac_n.sum()} ({ac_n.mean()*100:.2f}%)")
print(f"pass_team == 0 (Home, norm): {(pt_n == 0).sum()} ({(pt_n == 0).mean()*100:.2f}%)")
print(f"pass_team == 1 (Away, norm): {(pt_n == 1).sum()} ({(pt_n == 1).mean()*100:.2f}%)")
