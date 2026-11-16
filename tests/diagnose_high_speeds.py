import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.features.atd_parser import ATDParser
from src.features.ball_trajectory import process_ball_trajectory, PITCH_LENGTH_METERS, PITCH_WIDTH_METERS

parser = ATDParser()
data = parser.parse('data/tactical/BAR-ATH_tactical_atd.txt', 'data/tactical/BAR-ATH_tactical_atd.xml')
bx = data.ball_pos_norm[:, 0]
by = data.ball_pos_norm[:, 1]
res = process_ball_trajectory(bx, by)
spd = res['speed']

idx_45 = np.where(spd >= 44.9)[0]
print(f"Total frames at or above 45 m/s clamp: {len(idx_45)}")

print("\nFirst 3 cases of >= 45 m/s:")
for idx in idx_45[:3]:
    print(f"\n--- Frame {idx} ---")
    for t in range(max(0, idx - 4), min(len(bx), idx + 5)):
        x_m = bx[t] * PITCH_LENGTH_METERS if np.isfinite(bx[t]) else np.nan
        y_m = by[t] * PITCH_WIDTH_METERS if np.isfinite(by[t]) else np.nan
        print(f"  t={t}: x={x_m:.2f}m, y={y_m:.2f}m, speed={spd[t]:.2f}m/s")
