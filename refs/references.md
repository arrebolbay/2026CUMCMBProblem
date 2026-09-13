# 参考文献（经 OpenAlex / Crossref API 核验，2026-09-11）

## A. 圆覆盖与检测点下界（问题3）

[1] FEJES TÓTH G. Thinnest covering of a circle by eight, nine, or ten congruent circles[M]//
Combinatorial and Computational Geometry. Cambridge: Cambridge University Press, 2005: 361-376.
DOI: 10.1017/9781009701259.019 · https://doi.org/10.1017/9781009701259.019
（OpenAlex: https://openalex.org/W2243611313）

[2] BEZDEK K. Über einige Kreisüberdeckungen[J]. Beiträge zur Algebra und Geometrie, 1983, 14: 7-13.
（n=6 时最优半径比 1.7988… 的原始证明，经 [1] 转引确认；无 DOI）

[3] GÁSPÁR Z, TARNAI T. Partial covering of a circle by equal circles. Part I: The mechanical models[J].
Journal of Computational Geometry, 2014, 5(1): 104-125. DOI: 10.20382/jocg.v5i1a6

## B. TSP 渐近下界与启发式（问题3 清除巡游）

[4] BEARDWOOD J, HALTON J H, HAMMERSLEY J M. The shortest path through many points[J].
Mathematical Proceedings of the Cambridge Philosophical Society, 1959, 55(4): 299-327.
DOI: 10.1017/S0305004100034095 · https://doi.org/10.1017/S0305004100034095

[5] CROES G A. A method for solving traveling-salesman problems[J].
Operations Research, 1958, 6(6): 791-812. DOI: 10.1287/opre.6.6.791

[6] LIN S, KERNIGHAN B W. An effective heuristic algorithm for the traveling-salesman problem[J].
Operations Research, 1973, 21(2): 498-516. DOI: 10.1287/opre.21.2.498

[7] ROSENKRANTZ D J, STEARNS R E, LEWIS P M. An analysis of several heuristics for the traveling
salesman problem[J]. SIAM Journal on Computing, 1977, 6(3): 563-581. DOI: 10.1137/0206041

## C. 集合覆盖（检测点布设的贪心算法）

[8] CHVÁTAL V. A greedy heuristic for the set-covering problem[J].
Mathematics of Operations Research, 1979, 4(3): 233-235. DOI: 10.1287/moor.4.3.233

[9] FEIGE U. A threshold of ln n for approximating set cover[J].
Journal of the ACM, 1998, 45(4): 634-652. DOI: 10.1145/285055.285059

## D. 纯方位定位与最优传感器几何（问题2、问题4）

[10] DOĞANÇAY K, HMAM H. Optimal angular sensor separation for AOA localization[J].
Signal Processing, 2008, 88(5): 1248-1260. DOI: 10.1016/j.sigpro.2007.11.013

[11] BISHOP A N, FIDAN B, ANDERSON B D O, et al. Optimality analysis of sensor-target localization
geometries[J]. Automatica, 2010, 46(3): 479-492. DOI: 10.1016/j.automatica.2009.12.003

[12] MARTÍNEZ S, BULLO F. Optimal sensor placement and motion coordination for target tracking[J].
Automatica, 2006, 42(4): 661-668. DOI: 10.1016/j.automatica.2005.12.018

[13] NARDONE S C, LINDGREN A G, GONG K F. Fundamental properties and performance of conventional
bearings-only target motion analysis[J]. IEEE Transactions on Automatic Control, 1984, 29(9): 775-787.
DOI: 10.1109/TAC.1984.1103664

[14] MORENO-SALINAS D, PASCOAL A M, ARANDA J. Sensor networks for optimal target localization with
bearings-only measurements in constrained three-dimensional scenarios[J]. Sensors, 2013, 13(8):
10386-10417. DOI: 10.3390/s130810386

## E. 集员估计（±1° 固定偏差的建模依据）

[15] SCHWEPPE F C. Recursive state estimation: Unknown but bounded errors and system inputs[J].
IEEE Transactions on Automatic Control, 1968, 13(1): 22-28. DOI: 10.1109/TAC.1968.1098790

## F. 计算几何工具（问题1）

[16] TOUSSAINT G T. Solving geometric problems with the rotating calipers[C]//
Proceedings of IEEE MELECON '83. Athens, 1983.
（OpenAlex: https://openalex.org/W179323378；无 DOI）

[17] WELZL E. Smallest enclosing disks (balls and ellipsoids)[M]//
New Results and New Trends in Computer Science, LNCS 555. Berlin: Springer, 1991: 359-370.
DOI: 10.1007/BFb0038202

## 检索记录（可复现）

所有条目通过开放学术 API 检索并与 DOI 注册信息交叉核对：

- OpenAlex 检索式示例：
  `https://api.openalex.org/works?filter=title.search:<题名关键词>`
- Crossref 检索式示例：
  `https://api.crossref.org/works?query.bibliographic=<题名关键词>&rows=3`
