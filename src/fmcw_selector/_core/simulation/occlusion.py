# simulation/occlusion.py
"""几何遮挡与车身占位: 判断"本车 -> 目标"的视线上有没有被另一辆车的车身挡住。

只做**二维刚体遮挡** (车身当成 长 x 宽 的矩形), 不做射线追踪、不算衍射、不算多径 ——
这是 P2 明确划的下界: 两档敏感性对比 (完全无遮挡 vs 这一档硬遮挡) 就足以回答"遮挡会不会
改变结论", 更精细的电磁建模对当前问题不增加可信度, 只增加不可核查性。

判据: 目标 q 被挡住 <=> 线段 (本车, q) 穿过第三辆车 k 的矩形车身。
早先版本用"车身圆盘, 半径 = 车长/2 = 2.25 m", 那是把轿车当成 4.5 m 宽的圆盘: 侧偏 2 m 的
车就能挡住 60 m 外的目标, 遮挡面积高估 2.5 倍以上 —— 换成矩形后横向只按 1.8 m 算。

已知偏差 (诚实记录, 别当成物理):
  * 二值判定: 被挡 = 完全看不见, 不做部分遮挡/绕射衰减。
  * 线段终点落在目标自己的车身里, 所以目标自身必须排除在遮挡物之外 (按 vid 排除)。
  * 本车车身不挡自己的视线 (天线装在车顶前部, 真实几何更复杂, 这里不再加自由度)。
  * 全场车辆统一按轿车矩形 (4.5 x 1.8 m), 行人/摩托车弱目标也不例外 (简化, 不区分类别)。
"""

from __future__ import annotations

import numpy as np


def segment_hits_box(p, q, c, head: float, half_l: float, half_w: float) -> bool:
    """线段 p->q 是否穿过中心 c、航向 head、尺寸 (2half_l x 2half_w) 的矩形。

    把两端转进矩形坐标系后对两块"板"(slab) 做 Liang-Barsky 裁剪: 参数区间交集非空即穿过。
    """
    h = np.array([np.cos(head), np.sin(head)])
    v = np.array([-h[1], h[0]])
    R = np.stack([h, v])                                  # 世界坐标 -> 矩形坐标
    a0 = R @ (np.asarray(p, dtype=float) - np.asarray(c, dtype=float))
    a1 = R @ (np.asarray(q, dtype=float) - np.asarray(c, dtype=float))
    d = a1 - a0
    t0, t1 = 0.0, 1.0
    for i, half in enumerate((half_l, half_w)):
        if abs(float(d[i])) < 1e-12:
            if not (-half <= float(a0[i]) <= half):
                return False                              # 整条线在这块板外且与它平行
            continue
        ta, tb = (-half - a0[i]) / d[i], (half - a0[i]) / d[i]
        t0 = max(t0, min(ta, tb))
        t1 = min(t1, max(ta, tb))
        if t0 > t1:
            return False
    return True


def bodies_overlap(xy_a, head_a, xy_b, head_b, length: float, width: float) -> bool:
    """两辆车 (length x width 的矩形) 是否互相穿过 —— 2D 分离轴 (SAT) 判据。

    "中心距 < 车长"这一朴素判据是**错的**: 相邻车道中心只隔 3.5 m, 两辆车并排正常行驶时
    中心距 3.5 m < 4.5 m 会被全部记成穿模 (实测 highway_overtake 因此报了 281 个假事件,
    而真正的路口穿模只有 117~149 个)。宽度取 1.8 m (轿车标准值, 工程选择)。
    """
    a = np.asarray(xy_a, dtype=float)
    b = np.asarray(xy_b, dtype=float)
    d = b - a
    ha = np.array([np.cos(head_a), np.sin(head_a)])
    hb = np.array([np.cos(head_b), np.sin(head_b)])
    na = np.array([-ha[1], ha[0]])
    nb = np.array([-hb[1], hb[0]])
    for axis in (ha, na, hb, nb):
        ext_a = 0.5 * length * abs(float(ha @ axis)) + 0.5 * width * abs(float(na @ axis))
        ext_b = 0.5 * length * abs(float(hb @ axis)) + 0.5 * width * abs(float(nb @ axis))
        if abs(float(d @ axis)) > ext_a + ext_b:
            return False                                  # 找到分离轴: 两个矩形不相交
    return True


def overlap_pairs(xys, heads, length: float, width: float) -> list:
    """全场车体重叠的车对下标列表 (i < j)。"""
    out = []
    for i in range(len(xys)):
        for j in range(i + 1, len(xys)):
            if bodies_overlap(xys[i], heads[i], xys[j], heads[j], length, width):
                out.append((i, j))
    return out


def los_blockage(ego_xy, targets, occluders, length: float, width: float) -> np.ndarray:
    """返回长度 len(targets) 的 bool 数组: True = 该目标的视线被别人的车身矩形挡住。

    Args:
        ego_xy: (2,) 本车位置 (m)
        targets: list[VehicleState] 待判定的目标 (含干扰车)
        occluders: list[VehicleState] 可充当遮挡物的车辆 (通常是场上除本车外的全部车)
        length, width: 车身矩形的长宽 (m)
    """
    out = np.zeros(len(targets), dtype=bool)
    if not occluders or width <= 0.0:
        return out
    p = np.asarray(ego_xy, dtype=float)
    for i, st in enumerate(targets):
        q = np.asarray(st.pos, dtype=float)
        if float(np.linalg.norm(q - p)) <= 1e-6:
            continue                                      # 与本车重合: 场景坏了, 不算被挡
        for k in occluders:
            if k.vid == st.vid:
                continue                                  # 目标不能挡住自己
            if segment_hits_box(p, q, k.pos, k.heading, length / 2.0, width / 2.0):
                out[i] = True
                break
    return out


def occlusion_report(ego_xy, states, length: float, width: float) -> dict:
    """一次快照的遮挡汇总 (供审计打印): 谁挡住了谁、挡住了几成。"""
    blockers = [st for st in states if st.rcs > 0.0 or st.radar is not None]
    blocked = los_blockage(ego_xy, blockers, [st for st in states if not st.is_ego],
                           length, width)
    n_tgt = len(blockers)
    return {'n_targets': n_tgt, 'n_blocked': int(blocked.sum()),
            'blocked_frac': float(blocked.mean()) if n_tgt else 0.0,
            'n_blocked_interferers': int(sum(blockers[i].radar is not None
                                             for i in range(n_tgt) if blocked[i])),
            'blocked_ids': [int(blockers[i].vid) for i in range(n_tgt) if blocked[i]]}
