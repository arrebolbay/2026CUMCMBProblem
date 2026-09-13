"""诊断：22 点网下 channel=12 为何清不掉（1 条示向度 + 极近可见点 86m）。"""
import math
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")
sys.path.insert(0, r"G:\C\CUMCM B题\scripts")

from _Dtest import build, V  # noqa: E402
from src.mock_simulator import MockSimulator, random_case  # noqa: E402

for lbl, pts in (("22点", build(1000.0, 9, 1875.0, 12, 0.0)),
                 ("23点", build(1000.0, 10, 1875.0, 12, 0.5))):
    jam = random_case(n=16, seed=20, directional_ratio=0.5)
    src = [j for j in jam if j.channel == 12][0]
    sim = MockSimulator(jam, seed=20)
    r = V(pts).run(sim)
    acts = [e for e in sim.log if e.get("channel") == 12]
    meas = [e for e in acts if e["path"] == "/measure"]
    clr = [e for e in acts if e["path"] == "/clear"]
    n_dir = sum(1 for e in meas if e["measure_result"] == "direction")
    print(f"\n===== {lbl}: 未清={r.unresolved} =====")
    print(f"  源({src.x:.0f},{src.y:.0f}) dir={src.direction:.1f} r_eff={src.r_eff:.0f}"
          f"  动作: measure={len(meas)} (direction={n_dir}) clear={len(clr)}"
          f" 成功={sum(1 for e in clr if e['clear_result']=='success')}")
    for e in meas[:12]:
        print(f"    measure@({e['position'][0]:.0f},{e['position'][1]:.0f})"
              f" -> {e['measure_result']} svd={e['svd_deg']}")
    for e in clr[:12]:
        print(f"    clear  @({e['position'][0]:.0f},{e['position'][1]:.0f})"
              f" d(源)={math.dist(e['position'], (src.x, src.y)):.1f} ->"
              f" {e['clear_result']}")
