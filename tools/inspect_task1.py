"""任务一 · 数据核对脚本（不是仓库原有文件，只用于自己查证数字）。

在仓库根目录运行：
    python tools/inspect_task1.py

它做的事：取第 100 帧，把画面上的「球 / 地面 / 鞋面」三块区域分别用
灰度图、R 通道、HSV 掩膜量一遍，并打印每个掩膜连通块的面积和位置。
所有数字都可以在 VS Code 里对着 runs/task1/frame-100.png 逐块核对。
"""
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision_lab.detection import color_mask, find_candidates, select_ball  # noqa: E402

VIDEO = ROOT / "samples" / "basketball.mp4"
FRAME = 100

# ---- 取帧 -------------------------------------------------------------
cap = cv2.VideoCapture(str(VIDEO))
frame = None
for index in range(FRAME + 1):
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"读不到第 {index} 帧")
cap.release()
print(f"frame {FRAME}: shape={frame.shape}")

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
r_chan = frame[:, :, 2]                      # OpenCV 是 BGR，索引 2 才是 R
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = color_mask(frame)

ball = select_ball(find_candidates(mask), frame.shape)
cx, cy, rad = ball.center[0], ball.center[1], ball.radius
print(f"球: center=({cx:.1f},{cy:.1f}) r={rad:.1f} area={ball.area:.0f} fill={ball.fill_ratio:.4f}")

H, W = frame.shape[:2]
yy, xx = np.mgrid[0:H, 0:W]
disk = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (rad * 0.55) ** 2   # 球内部一小块

# ---- 三块区域 ---------------------------------------------------------
shoe = np.zeros((H, W), bool)
shoe[int(H * 0.68):int(H * 0.95), 0:int(W * 0.32)] = True       # 左下鞋面
ground = np.ones((H, W), bool)                                   # 地面＝整幅画面去掉球
ground[int(cy - rad) - 20:int(cy + rad) + 20, int(cx - rad) - 20:int(cx + rad) + 20] = False

print("\n=== 三块区域的亮度（同一帧） ===")
print(f"{'区域':<8}{'灰度':>8}{'R 通道':>8}{'H':>7}{'S':>7}{'V':>7}{'掩膜命中':>10}")
for name, m in (("球", disk), ("地面", ground), ("鞋面", shoe)):
    print(
        f"{name:<8}{gray[m].mean():>8.1f}{r_chan[m].mean():>8.1f}"
        f"{np.median(hsv[m][:, 0]):>7.0f}{np.median(hsv[m][:, 1]):>7.0f}"
        f"{np.median(hsv[m][:, 2]):>7.0f}{(mask[m] > 0).mean() * 100:>9.1f}%"
    )

# ---- 地面为什么会被/不会被选中 ---------------------------------------
print("\n=== 地面的亮度分布（说明为什么灰度、R 通道都不可靠） ===")
g, r = gray[ground], r_chan[ground]
print(f"灰度: 均值={g.mean():.1f}  p95={np.percentile(g, 95):.0f}  最大={g.max()}"
      f"  与球均值(≈133)重叠的比例={(g >= 133).mean() * 100:.1f}%")
print(f"R   : 均值={r.mean():.1f}  p95={np.percentile(r, 95):.0f}  最大={r.max()}"
      f"  达到鞋面亮度(≥85)的比例={(r >= 85).mean() * 100:.1f}%  达到球亮度(≥171)的比例={(r >= 171).mean() * 100:.1f}%")

print("\n=== 掩膜里到底有几个连通块 ===")
count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
print(f"连通块数（含背景）: {count}，即前景 {count - 1} 个")
for i in range(1, count):
    x, y, w, h, area = stats[i]
    print(f"  块{i}: 面积={area}  bbox=({x},{y},{w},{h})")

print("\n=== 程序实际输出的候选 ===")
for i, item in enumerate(find_candidates(mask), 1):
    print(f"  候选{i}: r={item.radius:.1f} fill={item.fill_ratio:.3f}")

print("\n核对方法：打开 runs/task1/frame-100.png，")
print("  第 2 块（灰度）里球和鞋面的亮度是否接近、第 3 块（R）里鞋面是否发亮、")
print("  第 4 块（HSV 掩膜）里是否只剩球一个白块。")
