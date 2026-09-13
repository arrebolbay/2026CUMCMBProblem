"""计算几何核心库（问题 1 的算法基础，同时服务问题 2/3/4）。

数学模型
--------
单条示向度 beta 带有 ±eps 的**有界误差**（题目给定：同一地点误差固定，
故采用集员/区间模型而非统计平均）。因此每条示向度给出一个张角 2*eps
的**楔形** W(beta) = { X : dir(S->X) ∈ [beta-eps, beta+eps] }，
它是两个半平面的交。多个检测点的楔形相交得到**定位区域**

        P = ∩_i W(beta_i)

P 为凸多边形，真值目标必落在 P 内。

本模块提供：
  * 角度工具（归一化、环绕差）
  * 楔形 -> 半平面；半平面交（顺序裁剪）
  * 凸包、面积、形心
  * 凸多边形直径：旋转卡壳 O(V)（另有 O(V^2) 暴力实现用于交叉验证）
  * 最小覆盖圆：Welzl 随机增量（期望 O(V)）
  * 直径圆是否覆盖定位区域的判定
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from .config import EPS_DEG

Point = Tuple[float, float]

__all__ = [
    "Point",
    "normalize_deg",
    "angle_diff",
    "unit_vector",
    "distance",
    "bearing_between",
    "HalfPlane",
    "wedge_halfplanes",
    "clip_polygon",
    "halfplane_intersection",
    "convex_hull",
    "polygon_area",
    "polygon_centroid",
    "polygon_diameter_bruteforce",
    "polygon_diameter_calipers",
    "min_enclosing_circle",
    "circle_covers_polygon",
    "diameter_circle_covers",
    "localization_region",
    "wedge_polygon",
]


# --------------------------------------------------------------------------- #
# 1. 角度与向量工具
# --------------------------------------------------------------------------- #
def normalize_deg(angle: float) -> float:
    """把角度归一化到 [0, 360)。"""
    return angle % 360.0


def angle_diff(a: float, b: float) -> float:
    """角度环绕差 a-b，结果落在 (-180, 180]。"""
    return (a - b + 180.0) % 360.0 - 180.0


def unit_vector(deg: float) -> Point:
    """方位角(度, 逆时针, x 轴正向为 0) 对应的单位向量。"""
    rad = math.radians(deg)
    return (math.cos(rad), math.sin(rad))


def distance(p: Point, q: Point) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def bearing_between(p: Point, q: Point) -> float:
    """由 p 指向 q 的方位角，范围 [0, 360)。"""
    return normalize_deg(math.degrees(math.atan2(q[1] - p[1], q[0] - p[0])))


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


# --------------------------------------------------------------------------- #
# 2. 半平面与楔形
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HalfPlane:
    """半平面 { X : nx*x + ny*y <= c }，(nx, ny) 为单位外法向。"""

    nx: float
    ny: float
    c: float

    def signed_value(self, p: Point) -> float:
        """<0 表示点在半平面内部，>0 表示在外部。"""
        return self.nx * p[0] + self.ny * p[1] - self.c

    def contains(self, p: Point, tol: float = 1e-9) -> bool:
        return self.signed_value(p) <= tol

    @staticmethod
    def with_normal_through(normal_deg: float, point: Point) -> "HalfPlane":
        """构造法向为 normal_deg、且过给定点的半平面（法向指向外部）。"""
        nx, ny = unit_vector(normal_deg)
        return HalfPlane(nx, ny, nx * point[0] + ny * point[1])


def wedge_halfplanes(station: Point, bearing_deg: float, eps_deg: float = EPS_DEG):
    """示向度楔形 -> 两个半平面。

    楔形 = 从 station 出发、方向角落在 [beta-eps, beta+eps] 内的点的集合。
    两条边界射线的外法向分别为 beta+eps+90 与 beta-eps-90。
    """
    hp_plus = HalfPlane.with_normal_through(bearing_deg + eps_deg + 90.0, station)
    hp_minus = HalfPlane.with_normal_through(bearing_deg - eps_deg - 90.0, station)
    return hp_plus, hp_minus


# --------------------------------------------------------------------------- #
# 3. 半平面交（顺序裁剪）
# --------------------------------------------------------------------------- #
def clip_polygon(poly: Sequence[Point], hp: HalfPlane, tol: float = 1e-9) -> List[Point]:
    """用半平面裁剪多边形（Sutherland–Hodgman），保持顶点顺序。"""
    if not poly:
        return []
    out: List[Point] = []
    n = len(poly)
    for i in range(n):
        cur = poly[i]
        nxt = poly[(i + 1) % n]
        v_cur = hp.signed_value(cur)
        v_nxt = hp.signed_value(nxt)
        if v_cur <= tol:                     # 当前点在内部（含边界）
            out.append(cur)
        if (v_cur > tol and v_nxt < -tol) or (v_cur < -tol and v_nxt > tol):
            t = v_cur / (v_cur - v_nxt)      # 与边界求交
            out.append((cur[0] + t * (nxt[0] - cur[0]),
                        cur[1] + t * (nxt[1] - cur[1])))
    # 去重（裁剪产生的重合点）
    dedup: List[Point] = []
    for p in out:
        if not dedup or distance(dedup[-1], p) > 1e-7:
            dedup.append(p)
    if len(dedup) > 1 and distance(dedup[0], dedup[-1]) <= 1e-7:
        dedup.pop()
    return dedup


def halfplane_intersection(
    hps: Iterable[HalfPlane],
    bbox: float = 1.0e6,
    tol: float = 1e-9,
) -> List[Point]:
    """半平面交：从一个足够大的正方形开始逐次裁剪，返回凸多边形顶点。"""
    poly: List[Point] = [(-bbox, -bbox), (bbox, -bbox), (bbox, bbox), (-bbox, bbox)]
    for hp in hps:
        poly = clip_polygon(poly, hp, tol=tol)
        if len(poly) < 3:
            return []
    return poly


def localization_region(
    stations: Sequence[Point],
    bearings: Sequence[float],
    eps_deg: float = EPS_DEG,
    bbox: float = 1.0e6,
) -> List[Point]:
    """多站交会定位区域：P = ∩ W(beta_i)（凸多边形顶点列表）。"""
    if len(stations) != len(bearings):
        raise ValueError("stations 与 bearings 长度必须一致")
    hps: List[HalfPlane] = []
    for s, b in zip(stations, bearings):
        hps.extend(wedge_halfplanes(s, b, eps_deg))
    return halfplane_intersection(hps, bbox=bbox)


def wedge_polygon(
    station: Point,
    bearing_deg: float,
    eps_deg: float = EPS_DEG,
    radius: float = 1.0e6,
) -> List[Point]:
    """单个楔形（用于作图），返回四边形顶点。"""
    a = unit_vector(bearing_deg - eps_deg)
    b = unit_vector(bearing_deg + eps_deg)
    origin = station
    return [
        origin,
        (origin[0] + radius * a[0], origin[1] + radius * a[1]),
        (origin[0] + radius * b[0], origin[1] + radius * b[1]),
    ]


# --------------------------------------------------------------------------- #
# 4. 凸包 / 多边形基本量
# --------------------------------------------------------------------------- #
def convex_hull(points: Sequence[Point]) -> List[Point]:
    """Andrew 单调链凸包，返回逆时针顶点（不含共线冗余点）。"""
    pts = sorted(set((float(p[0]), float(p[1])) for p in points))
    if len(pts) <= 1:
        return pts

    def build(seq: Sequence[Point]) -> List[Point]:
        stack: List[Point] = []
        for p in seq:
            while len(stack) >= 2 and _cross(stack[-2], stack[-1], p) <= 1e-12:
                stack.pop()
            stack.append(p)
        return stack

    lower = build(pts)
    upper = build(reversed(pts))
    hull = lower[:-1] + upper[:-1]
    return hull if len(hull) >= 3 else pts


def polygon_area(poly: Sequence[Point]) -> float:
    """鞋带公式，返回非负面积。"""
    if len(poly) < 3:
        return 0.0
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def polygon_centroid(poly: Sequence[Point]) -> Point:
    """多边形形心（面积加权）。"""
    a = polygon_area(poly)
    if a <= 1e-12:
        if not poly:
            raise ValueError("空多边形")
        return (sum(p[0] for p in poly) / len(poly),
                sum(p[1] for p in poly) / len(poly))
    cx = cy = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        w = x1 * y2 - x2 * y1
        cx += (x1 + x2) * w
        cy += (y1 + y2) * w
    return (cx / (6.0 * a), cy / (6.0 * a))


# --------------------------------------------------------------------------- #
# 5. 凸多边形直径
# --------------------------------------------------------------------------- #
def polygon_diameter_bruteforce(poly: Sequence[Point]) -> Tuple[float, Point, Point]:
    """O(V^2) 暴力求直径（用于交叉验证）。"""
    best = -1.0
    bp, bq = (0.0, 0.0), (0.0, 0.0)
    for i, p in enumerate(poly):
        for q in poly[i + 1:]:
            d = distance(p, q)
            if d > best:
                best, bp, bq = d, p, q
    if best < 0:
        best = 0.0
    return best, bp, bq


def polygon_diameter_calipers(poly: Sequence[Point]) -> Tuple[float, Point, Point]:
    """旋转卡壳法求凸多边形直径，O(V)（Toussaint 1983）。

    凸多边形直径必在一对对踵顶点处取得；顶点数 <= 3 时直接暴力。
    """
    hull = convex_hull(poly)
    n = len(hull)
    if n <= 3:
        return polygon_diameter_bruteforce(hull)

    best = -1.0
    bp, bq = hull[0], hull[0]
    k = 1
    for i in range(n):
        j = (i + 1) % n
        while True:
            nk = (k + 1) % n
            area_k = abs(_cross(hull[i], hull[j], hull[k]))
            area_nk = abs(_cross(hull[i], hull[j], hull[nk]))
            if area_nk > area_k:
                k = nk
            else:
                break
        for cand in (hull[i], hull[j]):
            d = distance(cand, hull[k])
            if d > best:
                best, bp, bq = d, cand, hull[k]
    return best, bp, bq


# --------------------------------------------------------------------------- #
# 6. 最小覆盖圆（Welzl 随机增量）
# --------------------------------------------------------------------------- #
def _circle_from2(a: Point, b: Point) -> Tuple[Point, float]:
    cx, cy = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
    return (cx, cy), distance(a, b) / 2.0


def _circle_from3(a: Point, b: Point, c: Point) -> Tuple[Point, float]:
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:                       # 三点共线 -> 退化为两点圆
        pairs = [(a, b), (a, c), (b, c)]
        pair = max(pairs, key=lambda pr: distance(pr[0], pr[1]))
        return _circle_from2(*pair)
    ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay)
          + (cx ** 2 + cy ** 2) * (ay - by)) / d
    uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx)
          + (cx ** 2 + cy ** 2) * (bx - ax)) / d
    center = (ux, uy)
    return center, distance(center, a)


def min_enclosing_circle(
    points: Sequence[Point],
    seed: int = 20260911,
) -> Tuple[Point, float]:
    """最小覆盖圆（Welzl 随机增量，期望 O(n)）。

    问题1 第二问的"保证覆盖"方案：当以直径为直径的圆无法覆盖定位区域时，
    改用最小覆盖圆即可保证覆盖，而且半径最小。
    """
    pts = [(float(p[0]), float(p[1])) for p in points]
    if not pts:
        raise ValueError("空点集")
    rng = random.Random(seed)
    rng.shuffle(pts)

    center: Point = pts[0]
    radius = 0.0

    def inside(p: Point) -> bool:
        return distance(p, center) <= radius + 1e-9

    for i, p in enumerate(pts):
        if inside(p):
            continue
        center, radius = p, 0.0
        for j in range(i):
            q = pts[j]
            if inside(q):
                continue
            center, radius = _circle_from2(p, q)
            for k in range(j):
                r = pts[k]
                if inside(r):
                    continue
                center, radius = _circle_from3(p, q, r)
    return center, radius


# --------------------------------------------------------------------------- #
# 7. 覆盖判定（问题1 第二问）
# --------------------------------------------------------------------------- #
def circle_covers_polygon(center: Point, radius: float, poly: Sequence[Point],
                          tol: float = 1e-7) -> bool:
    """圆是否覆盖凸多边形（凸性 => 只需检查顶点）。"""
    return all(distance(center, p) <= radius + tol for p in poly)


def diameter_circle_covers(poly: Sequence[Point]) -> dict:
    """以定位区域"直径为直径"的圆能否覆盖该区域。

    两种等价判定：
      * 圆心为直径 AB 的中点，覆盖凸多边形 <=> 各顶点对 AB 张角 >= 90 度；
      * 直接检验各顶点到 (A+B)/2 的距离 <= |AB|/2。
    """
    if len(poly) < 3:
        raise ValueError("多边形顶点不足")
    diam, pa, pb = polygon_diameter_calipers(poly)
    center = ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)
    radius = diam / 2.0
    covered = circle_covers_polygon(center, radius, poly)
    mec_center, mec_radius = min_enclosing_circle(poly)
    return {
        "diameter": diam,
        "p_a": pa,
        "p_b": pb,
        "circle_center": center,
        "circle_radius": radius,
        "covered": covered,
        "mec_center": mec_center,
        "mec_radius": mec_radius,
        "mec_diameter": 2.0 * mec_radius,
    }
