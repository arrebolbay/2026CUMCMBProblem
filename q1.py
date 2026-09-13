# ============================================================================
# 【学习注释】问题一：测向交会定位（q1.py，约 300 行）
# ----------------------------------------------------------------------------
# 这个文件解决"已知若干检测点与对应示向度，求干扰源的定位区域及其直径"。
# 三个值得学的点：
#   1) 把"±1° 的示向度误差"化成**两条边界射线**，进而写成两个**半平面**；
#      用二维叉积表示，避免斜率写法在 90°/270° 处发散，也避免 0°/360° 跨界。
#   2) 定位区域 = 全部半平面求交，必为**凸多边形** ⇒ 直径必在**极点对**取得，
#      于是连续优化降为有限枚举（更快可用旋转卡壳 O(n)）。
#   3) "以直径为直径的圆能否覆盖定位区域"给了**完全判定**：
#      覆盖圆圆心只能是最远点对的中点；只要存在三个顶点内角都 < 90°，覆盖必然失败。
#      工程结论：应当用**最小覆盖圆**而非直径圆作为不确定性包络。
# 主要函数：clip_polygon_by_left_halfplane / intersect_bearing_sectors /
#          polygon_diameter / diameter_circle_coverage / solve_localization
# ============================================================================

"""B 题问题 1：测向交会定位。"""
from __future__ import annotations
from dataclasses import dataclass
from math import cos, hypot, isfinite, radians, sin
from typing import Literal, Sequence
Point = tuple[float, float]
Status = Literal["bounded", "empty", "unbounded_or_too_large"]
@dataclass(frozen=True)
class LocalizationResult :
    status: Status
    polygon: tuple[Point, ...]
    diameter: float | None
    diameter_pair: tuple[Point, Point] | None
    diameter_circle_center: Point | None
    diameter_circle_radius: float | None
    covered_by_diameter_circle: bool | None
    search_half_size: float
def _sub(a: Point, b: Point) -> Point:
    return a[0] - b[0], a[1] - b[1]
def _cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]
def _distance(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])
def _deduplicate_polygon(polygon: list[Point], eps: float) -> list[Point]:
    if not polygon:
        return []
    points = [polygon[0]]
    for p in polygon[1:]:
        if _distance(p, points[-1]) > eps:
            points.append(p)
    if len(points) > 1 and _distance(points[0], points[-1]) <= eps:
        points.pop()
    return points
# 【学习注释】半平面裁剪（Sutherland–Hodgman 的关键一步）：
# 用一条有向直线把当前多边形裁掉"外侧"，返回剩下的凸多边形顶点。
# value(p) = cross(方向, p − 线上一点)，符号即"在直线哪一侧"；
# 相邻两点跨侧时按参数 t 求交点插入，保证裁完仍是凸多边形。

def clip_polygon_by_left_halfplane(
    polygon: Sequence[Point],
    line_point: Point,
    line_direction: Point,
    eps: float,
) -> list[Point]:
    if not polygon:
        return []
    def side(p: Point) -> float:
        return _cross(line_direction, _sub(p, line_point))
    result: list[Point] = []
    prev = polygon[-1]
    prev_value = side(prev)
    prev_inside = prev_value >= -eps
    for cur in polygon:
        cur_value = side(cur)
        cur_inside = cur_value >= -eps
        if prev_inside != cur_inside:
            denom = prev_value - cur_value
            if abs(denom) > eps:
                t = prev_value / denom
                t = max(0.0, min(1.0, t))
                result.append(
                    (
                        prev[0] + t * (cur[0] - prev[0]),
                        prev[1] + t * (cur[1] - prev[1]),
                    )
                )
        if cur_inside:
            result.append(cur)
        prev = cur
        prev_value = cur_value
        prev_inside = cur_inside
    return _deduplicate_polygon(result, eps)
# 【学习注释】把"检测点 S 测得示向度 θ ± 1°"化成**两条边界射线**，
# 再用二维叉积写成**两个半平面**（注意：不是用斜率！）。
# 好处：① 90°/270° 附近不会发散；② 0° 与 360° 跨界不会出错；
#       ③ 半平面求交可直接用线性裁剪，几何意义清晰。

def _bearing_halfplanes(
    station: Point, bearing_deg: float, error_deg: float
) -> tuple[tuple[Point, Point], tuple[Point, Point]]:
    low = radians(bearing_deg - error_deg)
    high = radians(bearing_deg + error_deg)
    d1 = cos(low), sin(low)
    d2 = cos(high), sin(high)
    return (station, d1), (station, (-d2[0], -d2[1]))
# 【学习注释】定位区域 = 各检测点给出的角域**逐个求交**。
# 每个角域是两个半平面之交，故整体是"闭半平面之交" ⇒ 必为**闭凸集**；
# 有界的充要条件是 n 段示向弧之交为空（论文用回收锥证明）。

def intersect_bearing_sectors(
    stations: Sequence[Point],
    bearings_deg: Sequence[float],
    error_deg: float = 1.0,
    initial_half_size: float = 1800.0,
    max_expansions: int = 24,
    relative_eps: float = 1e-10,
) -> tuple[Status, list[Point], float]:
    if len(stations) != len(bearings_deg):
        raise ValueError (" 检测点数量与示向度数量必须相同")
    if not stations:
        raise ValueError (" 至少需要一个检测点")
    if not isfinite(error_deg) or not 0.0 < error_deg < 90.0:
        raise ValueError (" 该凸角域算法要求 0 < error_deg < 90")
    if not isfinite(initial_half_size) or initial_half_size <= 0.0:
        raise ValueError ("initial_half_size 必须为有限正数")
    if not isinstance(max_expansions, int) or max_expansions < 0:
        raise ValueError ("max_expansions 必须为非负整数")
    if not isfinite(relative_eps) or relative_eps <= 0.0:
        raise ValueError ("relative_eps 必须为正")
    stations = [(float(x), float(y)) for x, y in stations]
    bearings = [float(x) for x in bearings_deg]
    if any(not isfinite(v) for p in stations for v in p):
        raise ValueError (" 检测点坐标必须为有限数")
    if any(not isfinite(v) for v in bearings):
        raise ValueError (" 示向度必须为有限数")
    bearings = [x % 360.0 for x in bearings]
    limits = [
        hp
        for station, bearing in zip(stations, bearings)
        for hp in _bearing_halfplanes(station, bearing, error_deg)
    ]
    half_size = float(initial_half_size)
    last_polygon: list[Point] = []
    for k in range(max_expansions + 1):
        eps = relative_eps * max(1.0, half_size)
        polygon: list[Point] = [
            (-half_size, -half_size),
            (half_size, -half_size),
            (half_size, half_size),
            (-half_size, half_size),
        ]
        for point, direction in limits:
            polygon = clip_polygon_by_left_halfplane(
                polygon, point, direction, eps
            )
            if not polygon:
                break
        last_polygon = polygon
        if not polygon:
            return "empty", [], half_size
        tol = 20.0 * eps
        on_box = any(
            abs(abs(x) - half_size) <= tol
            or abs(abs(y) - half_size) <= tol
            for x, y in polygon
        )
        if not on_box:
            return "bounded", polygon, half_size
        if k < max_expansions:
            half_size *= 2.0
    return "unbounded_or_too_large", last_polygon, half_size
# 【学习注释】求凸多边形的直径（区域内两点最大距离）。
# 依据"凸集直径必在极点对取得"，把连续优化降为**有限枚举**；
# 点多时可换旋转卡壳在 O(m) 内扫完对踵点对。

def polygon_diameter(polygon: Sequence[Point]) -> tuple[float, tuple[Point, Point]]:
    if not polygon:
        raise ValueError (" 空区域没有直径")
    if len(polygon) == 1:
        return 0.0, (polygon[0], polygon[0])
    max_d2 = -1.0
    pair = polygon[0], polygon[1]
    for i, p in enumerate(polygon):
        for q in polygon[i + 1 :]:
            d2 = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
            if d2 > max_d2:
                max_d2 = d2
                pair = p, q
    return max_d2**0.5, pair
# 【学习注释】**"以直径为直径的圆能否覆盖定位区域"的完全判定**（论文亮点）：
#   ① 覆盖圆的圆心只可能是**最远点对的中点**（否则无法同时容纳这两点）；
#   ② 判据：存在一对顶点，使其余所有顶点对它的张角都 ≥ 90° ⇒ 覆盖；
#      等价推论：只要**有 3 个顶点内角都 < 90°**，覆盖必然失败。
# 反例：交会角 γ = 91° 时有顶点超出直径圆 0.307 m（相对 1.76%）。
# 工程结论：应以**最小覆盖圆**而非直径圆作为不确定性包络。

def diameter_circle_coverage(
    polygon: Sequence[Point],
    diameter: float,
    diameter_pair: tuple[Point, Point],
    relative_eps: float = 1e-10,
) -> tuple[Point, float, bool]:
    if not polygon:
        raise ValueError (" 空区域无法判断直径圆覆盖")
    if not isfinite(diameter) or diameter < 0.0:
        raise ValueError ("diameter 必须为有限非负数")
    if not isfinite(relative_eps) or relative_eps <= 0.0:
        raise ValueError ("relative_eps 必须为正")
    a, b = diameter_pair
    pair_distance = _distance(a, b)
    pair_tol = relative_eps * max(1.0, diameter, pair_distance)
    if abs(pair_distance - diameter) > pair_tol:
        raise ValueError ("diameter_pair 两点距离必须等于 diameter")
    center = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
    radius = diameter / 2.0
    tol = relative_eps * max(1.0, diameter, radius)
    covered = all(_distance(center, p) <= radius + tol for p in polygon)
    return center, radius, covered
def solve_localization(
    stations: Sequence[Point],
    bearings_deg: Sequence[float],
    error_deg: float = 1.0,
    initial_half_size: float = 1800.0,
    max_expansions: int = 24,
    relative_eps: float = 1e-10,
) -> LocalizationResult:
    status, polygon, half_size = intersect_bearing_sectors(
        stations=stations,
        bearings_deg=bearings_deg,
        error_deg=error_deg,
        initial_half_size=initial_half_size,
        max_expansions=max_expansions,
        relative_eps=relative_eps,
    )
    if status != "bounded" or not polygon:
        return LocalizationResult(
            status=status,
            polygon=tuple(polygon),
            diameter=None,
            diameter_pair=None,
            diameter_circle_center=None,
            diameter_circle_radius=None,
            covered_by_diameter_circle=None,
            search_half_size=half_size,
        )
    diameter, pair = polygon_diameter(polygon)
    center, radius, covered = diameter_circle_coverage(
        polygon=polygon,
        diameter=diameter,
        diameter_pair=pair,
        relative_eps=relative_eps,
    )
    return LocalizationResult(
        status=status,
        polygon=tuple(polygon),
        diameter=diameter,
        diameter_pair=pair,
        diameter_circle_center=center,
        diameter_circle_radius=radius,
        covered_by_diameter_circle=covered,
        search_half_size=half_size,
    )
def plot_localization(
    result: LocalizationResult,
    stations: Sequence[Point],
    bearings_deg: Sequence[float],
    error_deg: float = 1.0,
    save_path: str | None = None,
) -> None:
    if result.status != "bounded" or not result.polygon:
        raise ValueError (" 只有非空有限区域可以绘图")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle as CirclePatch
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8, 8))
    closed = [*result.polygon, result.polygon[0]]
    xs = [p[0] for p in closed]
    ys = [p[1] for p in closed]
    ax.fill(xs, ys, color="tab:blue", alpha=0.25, label=" 定位区域")
    ax.plot(xs, ys, color="tab:blue")
    ray_length = max(100.0, 1.2 * result.search_half_size)
    for i, (station, bearing) in enumerate(zip(stations, bearings_deg), 1):
        ax.scatter(*station, marker="^", color="black", zorder=5)
        ax.annotate(f"S{i}", station, xytext=(5, 5), textcoords="offset points")
        for angle in (bearing - error_deg, bearing + error_deg):
            rad = radians(angle)
            end = (
                station[0] + ray_length * cos(rad),
                station[1] + ray_length * sin(rad),
            )
            ax.plot(
                [station[0], end[0]],
                [station[1], end[1]],
                color="gray",
                linestyle="--",
                linewidth=0.8,
            )
    assert result.diameter_pair is not None
    p1, p2 = result.diameter_pair
    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        color="tab:orange",
        linewidth=2.0,
        label=f" 直径 D= {result.diameter:.3f} m",
    )
    assert result.diameter_circle_center is not None
    assert result.diameter_circle_radius is not None
    circle_color = "tab:green" if result.covered_by_diameter_circle else "tab:red"
    ax.add_patch(
        CirclePatch(
            result.diameter_circle_center,
            result.diameter_circle_radius,
            fill=False,
            color=circle_color,
            linewidth=2.0,
            label=f" 直径圆 R=D/2= {result.diameter_circle_radius:.3f} m",
        )
    )
    ax.scatter(*result.diameter_circle_center, marker="x", color=circle_color, zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if save_path is None :
        plt.show()
    else:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
if __name__ == "__main__":
    demo_stations = [
        (-400.0, 0.0),
        (600.0, 0.0),
        (100.0, 500.0),
    ]
    demo_bearings = [0.0, 180.0, 270.0]
    demo_result = solve_localization(demo_stations, demo_bearings)
    print(f" 状态: {demo_result.status}")
    print(f" 顶点数: {len(demo_result.polygon)}")
    if demo_result.diameter is not None :
        print(f" 定位区域直径: {demo_result.diameter:.6f} m")
        print(f" 直径圆半径: {demo_result.diameter_circle_radius:.6f} m")
        print(f" 直径为 D 的圆能否覆盖: {demo_result.covered_by_diameter_circle}")
