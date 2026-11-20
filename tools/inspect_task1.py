"""任务一 · 固定样本区测量（替代旧版"整幅画面去掉球"的口径）。

样本区（全部写明行/列范围，避免"地面"里混进鞋面、反光或球）：
  A 球心邻域 : 以检出球心为心、半径 r×0.55 的圆盘
  B 右上暗地面: 行 150–330，列 230–350
  C 下部亮斑 : 行 340–430，列 265–355
  D 左下鞋面 : 行 440–600，列 0–110

判定门槛（vision_lab/detection.py）：H 4–28，S ≥ 105，V ≥ 85。

运行：python tools/inspect_task1.py
"""
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision_lab.detection import BASELINE_FILL, color_mask, find_candidates, select_ball  # noqa: E402

VIDEO = ROOT / "samples" / "basketball.mp4"
FRAME = 100
H_LO, H_HI, S_LO, V_LO = 4, 28, 105, 85


def load_frame(index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(VIDEO))
    frame = None
    for i in range(index + 1):
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise SystemExit(f"读不到第 {i} 帧")
    cap.release()
    return frame


frame = load_frame(FRAME)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
r_chan = frame[:, :, 2]
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = color_mask(frame)
ball = select_ball(find_candidates(mask), frame.shape)
cx, cy, rad = ball.center[0], ball.center[1], ball.radius

height, width = gray.shape
yy, xx = np.mgrid[0:height, 0:width]


def rect(r0, r1, c0, c1):
    m = np.zeros((height, width), bool)
    m[r0:r1, c0:c1] = True
    return m


regions = {
    "A 球心邻域": ((xx - cx) ** 2 + (yy - cy) ** 2) <= (rad * 0.55) ** 2,
    "B 右上暗地面": rect(150, 330, 230, 350),
    "C 下部亮斑": rect(340, 430, 265, 355),
    "D 左下鞋面": rect(440, 600, 0, 110),
}

print(f"frame {FRAME}: {width}x{height}")
print(f"ball: center=({cx:.1f},{cy:.1f}) r={rad:.1f} fill={ball.fill_ratio:.3f} (threshold {BASELINE_FILL})\n")
header = (f"{'样本区':<12}{'像素数':>8}{'灰度均值':>9}{'R均值':>8}{'R最大':>7}"
          f"{'H中位':>7}{'S中位':>7}{'V中位':>7}{'掩膜命中':>9}")
print(header)
print("-" * len(header))
for name, m in regions.items():
    h = hsv[m]
    print(f"{name:<12}{m.sum():>8}{gray[m].mean():>9.1f}{r_chan[m].mean():>8.1f}{r_chan[m].max():>7}"
          f"{np.median(h[:, 0]):>7.0f}{np.median(h[:, 1]):>7.0f}{np.median(h[:, 2]):>7.0f}"
          f"{(mask[m] > 0).mean() * 100:>8.1f}%")

print("\n=== 谁挡掉了干扰区？逐区检查三个门槛 ===")
for name, m in regions.items():
    if name.startswith("A"):
        continue
    h = hsv[m]
    in_h = (h[:, 0] >= H_LO) & (h[:, 0] <= H_HI)
    pass_all = in_h & (h[:, 1] >= S_LO) & (h[:, 2] >= V_LO)
    s_med = np.median(h[in_h][:, 1]) if in_h.any() else float("nan")
    print(f"{name}: H 在 [{H_LO},{H_HI}] 的像素 {int(in_h.sum())} 个（{in_h.mean()*100:.2f}%），"
          f"其中 S 中位 {s_med:.0f}，通过 S/V 的 {int(pass_all.sum())} 个")

print("\n=== 球以外的候选（应为 0 个额外候选）===")
blobs = cv2.connectedComponentsWithStats(mask, 8)
print(f"掩膜前景连通块: {blobs[0] - 1} 个")
for i in range(1, blobs[0]):
    x, y, w, h_, area = blobs[2][i]
    print(f"  块{i}: area={area} bbox=({x},{y},{w},{h_})")
