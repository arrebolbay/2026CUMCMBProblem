"""验证 inner_first_strong 全档效果（复用 _p4order 的 V/bench）。"""
import sys

sys.path.insert(0, r"G:\C\CUMCM B题")
sys.path.insert(0, r"G:\C\CUMCM B题\scripts")

from _p4order import bench  # noqa: E402

print("全档对比：tsp(当前) vs inner_first_strong（30 例/档，同种子）")
for ratio, tag in [(0.0, "全向"), (0.5, "半定向"), (1.0, "全定向")]:
    row = [f"{tag:4s}"]
    for order in ("tsp", "inner_first_strong"):
        cells = []
        for n in (10, 16):
            t, m, u = bench(ratio, n, seeds=30, order=order)
            cells.append(f"N{n}:{t:6.1f}/{m:5.2f}")
        row.append(f"{order:18s} " + " ".join(cells))
    print(" | ".join(row))
