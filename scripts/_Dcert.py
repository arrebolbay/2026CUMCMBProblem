"""最终安全验证：22 点候选网的超高分辨率定向覆盖证书。"""
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")
sys.path.insert(0, r"G:\C\CUMCM B题\scripts")

from src.coverage import directional_certificate as cert  # noqa: E402
from _Dtest import build  # noqa: E402

for lbl, pts in (("基准 25 点 (950x12+1875x12)", build(950.0, 12, 1875.0, 12, 0.5)),
                 ("候选 22 点 (1000x9+1875x12)", build(1000.0, 9, 1875.0, 12, 0.0))):
    print(f"--- {lbl} ({len(pts)} 点) ---")
    for tag, (nr, na, nb) in (("高分辨  ", (80, 720, 7200)),
                              ("超高分辨", (120, 1440, 14400)),
                              ("极高分辨", (200, 2160, 21600))):
        r = cert(pts, n_radial=nr, n_angular=na, n_boundary=nb)
        print(f"   {tag}: fail={r['fail']:4d}  最大角隙={r['max_gap_deg']:.4f}°  "
              f"样本={r['n_samples']}")
