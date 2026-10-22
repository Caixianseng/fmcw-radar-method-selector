# simulation/traffic.py
"""交通层: 闭合环线 + IDM 跟驰 + "密度曲线 -> 实际车辆数"的构造性控制。

为什么是环: 会话要在几十秒尺度上看到"稀疏 -> 拥堵 -> 消散", 而开放路段的通行时间
(路段长 / 车速) 跟会话时长同量级甚至更长 —— 车还没开出去, 密度曲线就已经翻篇了。
闭合环 + 按密度直接增删车辆 (见 target_count / _maintain_population), 密度旋钮才
真的对应到路上的车数, 审计也才验得了。

代价是"车流演化"里有多少是控制器造的必须交代清楚 —— 所以有三种车流来源 (population_mode,
`ScenarioConfig` 里有完整说明): 'density' 环上凭空增删 (台账逐项记名)、'fixed' 全程固定
车数的受控基准、'flow' 开放路段只由出入口改变车数。验收在
`experiments/stage2_traffic_verification.py` (逐时刻守恒、车体不重叠、实测密度而不是旋钮)。

时间尺度: 本层以 dt_traffic (出厂场景 5 ms) 积分; 接收信号层在 chirp 内用
"固定交通网格状态 + 截停一致的匀加速外推" 求任意时刻的车辆位姿 (75 us 内位移 < 2 mm; 保留一阶项是为了
让 chirp 间与帧间的相对运动正确累积)。

干扰雷达身份在车辆进入场景时一次分配 (见 assign_radar), 之后只按自己的时钟走。

━━ 刻意简化的部分 (2026-10-02 决定, 附实测后果) ━━
1) **全场统一按轿车车身** (4.5 x 1.8 m): 不做车辆类别/尺寸建模, 弱目标 (行人/摩托车量级)
   也按轿车占位、按轿车影子遮挡视线。
2) **没有汇入机动**: 'density' 的补车与 'flow' 的入口到达都是**控制器动作**, 不是"从匝道
   并进来的一辆车" —— 新车速度按当前期望速度给、位置按最大空隙的中点放, 不检查它是否落在
   前后车的 IDM 需求间距里。实测后果 (40 s, seed=1, 逐车步记录原始 IDM 加速度, 可由
   `debugging/_probe_batch_b2.py` 复现, 日志 `artifacts/probe_batch_b_before.log`):
     * corridor_in_out: 28 次入口到达里 18 次连新车自身的 s0+v·T 都不满足; 最差 t=28.4 s
       以 21.9 m/s 插进 3.9 m/s 前车后方 5.3 m 净距 -> 单步 a = −1138.74 m/s² (物理不存在)。
     * 环线补车同理: near_identical_slope 12/12、weak_target_strong_interferer 28/28 辆
       间距不足; 进场同车道 0.5 s 内最低 −9.56 / −9.09 m/s², 而正常跟驰只有 −2.86 / −3.15。
   ⇒ 所以 stage2 的"急刹比例"列 (hard_brake_frac) 数的主要是**这些进场事件**而不是驾驶行为,
     引用它之前必须对照同一时刻的台账增删 (self.ledger)。
   ⇒ 为什么仍不改: 加上"按前车速度汇入 + 两侧需求间距校验"会重画车流本身 (拥堵平台可能补
     不进车、密度旋钮失效), 而本节要的是**干扰形态的演化**, 不是交通工程真实性。要改就得
     连 §2 与主表一起重跑, 由人来决定而不是顺手改。

━━ 交叉口让行为什么没做 (实测过的两条路都不可用) ━━
P2 的要求是"路口要么建模让行, 要么在主结果里排除", 这里选**排除**, 但把试过的两次连同
数字留下, 免得后人重新踩:
  1) **接受间隙比较** (谁先到冲突区谁走, 比较 [t_in, t_in+2r/v] 时间窗): 让行决策一旦
     触发就让本车朝停止线刹停, 而"我什么时候到"是按**当前速度**算的 —— 车一减速, 预计
     到达时刻就更晚, 于是永远"有人比我先到", 主路上的受害者从 t≈2 s 起停在 s=33 m 不动
     直到会话结束 (位移 40 m / 356 m), 且与密度曲线完全无关。
  2) **时间窗预约** (先到先锁、预约不撤销、主路可抢支路未进箱的预约): 消除了上面的饿死,
     但低优先级车流的"不堵箱"与"已进箱不许急刹"两条相互牵制, 结果是既没治好穿模
     (重叠样本 74~137, 最小间距 0.1 m), 又把受害者位移压到 40~108 m。
     顶牛时两辆车互相当成对方的虚拟前车, 出现过 a=-60.8 m/s^2 的死锁对。
根因是这个交通层只有**纵向**跟驰 (IDM) 且车道是折线弧长上的点, 没有横向让路/换道自由度;
真实的交叉口冲突要靠转向路径与车道级让行规则才能同时满足"不穿模"和"通行正常"。所以出厂
场景里 intersection_turn 的两条交叉道路只用作**波束扫过路口车辆**的几何, 交叉处的车辆
重叠 (穿模) 由 experiments/stage2_traffic_verification.py 数出来并如实报告。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fmcw_selector._core.configs.traffic import ScenarioConfig
from fmcw_selector._core.simulation.radar_clock import ChirpTimeline


@dataclass
class Lane:
    """一条车道: 中心线折点 (按行车顺序) + 限速 + 闭合/开放标记。"""
    index: int
    points: np.ndarray          # (K, 2) 道路中心线折点 (m)
    offset: float               # 横向偏移 (m), 右正 (沿折点方向的右手侧)
    speed_limit: float          # m/s
    closed: bool = True

    def __post_init__(self):
        seg = np.diff(self.points, axis=0)
        norm = np.linalg.norm(seg, axis=1)
        left = np.stack([-seg[:, 1], seg[:, 0]], axis=1) / np.maximum(norm[:, None], 1e-9)
        left = np.concatenate([left, left[-1:]])          # 折点数比段数多 1, 末段沿用
        self.center = self.points + self.offset * left
        self.cum = np.r_[0.0, np.cumsum(norm)]
        self.length = float(self.cum[-1])
        self.tangent_vec = np.concatenate([seg / np.maximum(norm[:, None], 1e-9),
                                           [seg[-1] / max(norm[-1], 1e-9)]], axis=0)

    def _seg_index(self, s: float) -> int:
        return int(np.clip(np.searchsorted(self.cum, s) - 1, 0, self.center.shape[0] - 2))

    def wrap(self, s: float) -> float:
        """闭合车道把弧长折回 [0, length); 开放车道原样返回 (越界由 flow 负责离场)。

        闭合环是"会话可以比路段通行时间长"的前提: 受害者不会开几十秒就掉出世界。但代价
        是环上不能凭空加车 —— 要真实涨落就得用开放路段 + 入口到达 (population_mode='flow')。
        """
        if not self.closed or self.length <= 0:
            return float(s)
        return float(s % self.length)

    def s_at(self, xy) -> float:
        """给定平面点, 返回它在车道上的**投影**弧长 (逐段投影取最近点)。

        不能取最近折点: 开放路段的一条直段可以有两百米长, 折点投影会把冲突点甩到
        几十米外, 于是"这条车道经不经过该冲突点"判错、让行规则整条失效 (实测过)。
        """
        p = np.asarray(xy, dtype=float)
        a, b = self.center[:-1], self.center[1:]
        d = b - a
        l2 = np.maximum((d * d).sum(1), 1e-12)
        t = np.clip(((p - a) * d).sum(1) / l2, 0.0, 1.0)
        dist = np.linalg.norm(a + t[:, None] * d - p, axis=1)
        i = int(np.argmin(dist))
        seg_len = float(np.sqrt(l2[i]))
        return float(self.cum[i] + t[i] * seg_len)

    def distance_to(self, xy) -> float:
        """平面点到车道中心线的最近距离 (m) —— 判"这个点落在不在本车道上"。"""
        p = np.asarray(xy, dtype=float)
        a, b = self.center[:-1], self.center[1:]
        d = b - a
        l2 = np.maximum((d * d).sum(1), 1e-12)
        t = np.clip(((p - a) * d).sum(1) / l2, 0.0, 1.0)
        return float(np.linalg.norm(a + t[:, None] * d - p, axis=1).min())

    def position(self, s: float) -> np.ndarray:
        s = self.wrap(s)
        i = self._seg_index(s)
        l = self.cum[i + 1] - self.cum[i]
        r = 0.0 if l <= 1e-9 else (s - self.cum[i]) / l
        return (1.0 - r) * self.center[i] + r * self.center[i + 1]

    def positions(self, s_arr) -> np.ndarray:
        """批量版 position: 让行/避撞每步要算全场位置, 逐个调 position() 会慢二十倍。"""
        s = np.asarray(s_arr, dtype=float)
        s = s % self.length if self.closed else s
        i = np.clip(np.searchsorted(self.cum, s) - 1, 0, self.center.shape[0] - 2)
        l = self.cum[i + 1] - self.cum[i]
        r = np.where(l <= 1e-9, 0.0, (s - self.cum[i]) / np.maximum(l, 1e-9))
        return self.center[i] + r[:, None] * (self.center[i + 1] - self.center[i])

    def tangents(self, s_arr) -> np.ndarray:
        """批量版 tangent。"""
        s = np.asarray(s_arr, dtype=float)
        s = s % self.length if self.closed else s
        i = np.clip(np.searchsorted(self.cum, s) - 1, 0, self.center.shape[0] - 2)
        return self.tangent_vec[i]

    def tangent(self, s: float) -> np.ndarray:
        """行车方向单位向量 (弧长按环折叠)。"""
        i = self._seg_index(self.wrap(s))
        return self.tangent_vec[i]


@dataclass
class Vehicle:
    vid: int
    lane: int
    s: float = 0.0              # 沿车道的弧长 (m)
    v: float = 0.0              # 沿车道的速度 (m/s), 与 direction 同号
    a: float = 0.0
    length: float = 4.5
    rcs: float = 12.0
    is_ego: bool = False
    pinned: bool = False          # 场景点名要存在的车 (关键干扰车/弱目标): 不得被密度调控删掉
    radar: ChirpTimeline | None = None
    specular_phase: float = 0.0


@dataclass
class VehicleState:
    """某一时刻的车辆位姿快照 (供接收机与评价使用)。"""
    vid: int
    lane: int
    pos: np.ndarray             # (2,) m
    vel: np.ndarray             # (2,) m/s
    heading: float              # rad, 车头朝向
    rcs: float
    is_ego: bool
    radar: ChirpTimeline | None = None
    specular_phase: float = 0.0     # 镜面反射/直通路径的固定随机相位 (rad)


def lerp_profile(profile, t: float) -> float:
    """分段线性插值 (profile 为 ((时刻, 值), ...) 且时刻递增)。"""
    pts = np.asarray(profile, dtype=float)
    if t <= pts[0, 0]:
        return float(pts[0, 1])
    if t >= pts[-1, 0]:
        return float(pts[-1, 1])
    return float(np.interp(t, pts[:, 0], pts[:, 1]))


class TrafficFlow:
    """多车道交通仿真器。

    用法:
        flow = TrafficFlow(scen, victim_timeline, rng)
        flow.advance_to(t)                 # 内部按 dt_traffic 积分到位
        ego, others = flow.snapshot(t)     # t 时刻的位姿 (一阶外推)
    """

    def __init__(self, scen: ScenarioConfig, victim: ChirpTimeline, rng: np.random.Generator,
                 property_rng: np.random.Generator | None = None):
        self.scen = scen
        self.victim = victim
        self.rng = rng
        self.property_rng = property_rng if property_rng is not None else rng
        self.t = 0.0
        self._d_t = None            # density() 的 (时刻, 值) 缓存
        self._d_v = 0.0
        self.vehicles: list[Vehicle] = []
        # 数量守恒台账: (t, 车道, vid, 'in'/'out', 原因)。闭合环没有真实出入口, 密度调控
        # 的每一次凭空增删都必须留痕, 否则"车流演化"里有多少是控制器造的就查不出来
        # (P2: 观测区内车辆消失会直接改变干扰源集合, 这是仿真可信度问题不是细节)。
        self.ledger: list[tuple] = []
        # 急刹统计: |a| 超过 b_comfort × hard_brake_factor 的 (车, 步) 计数。计的是**原始
        # IDM 输出** (钳位前的请求值): 四个出厂场景 x 40 s 内没有一步触发 v<0 截断, 所以
        # "原始 a"与"实际 dv/dt"两个口径的计数完全相同 (1348/191/1488/375, 逐场景相等,
        # 见上面的探针日志), 换成 dv/dt 不改变任何数字。真正制造超限值的是凭空进场:
        # 三个环线场景"正常跟驰"档一条都没有, 全部急刹都落在补车同车道 0.5 s 内。
        self.n_hard_brake = 0
        self.n_vehicle_steps = 0
        self._next_vid = 1000
        self.ego = None
        self._build_static()
        if not np.isfinite(scen.dt_traffic) or scen.dt_traffic <= 0:
            raise ValueError("Positive finite traffic grid step required")
        self._refresh_accelerations()

    def ledger_summary(self, t_max: float | None = None) -> dict:
        """数量守恒台账汇总; t_max 给定时只统计到那一刻。

        residual 必须是 0: 场上车辆数 − 台账净值。非 0 就说明有代码路径**没记台账**
        就增删了车辆 (那才是"P2 说的那种凭空消失")。
        """
        rows = [r for r in self.ledger if t_max is None or r[0] <= t_max + 1e-12]
        n_in = sum(1 for r in rows if r[3] == 'in')
        n_out = len(rows) - n_in
        radar_in = sum(1 for r in rows if r[3] == 'in' and r[2] in self._radar_vids)
        reasons: dict[str, int] = {}
        for r in rows:
            reasons[f'{r[3]}:{r[4]}'] = reasons.get(f'{r[3]}:{r[4]}', 0) + 1
        return {'n_in': n_in, 'n_out': n_out, 'n_radar_spawned': radar_in,
                'reasons': reasons, 'net': n_in - n_out,
                'residual': len(self.vehicles) - (n_in - n_out)}

    def flow_summary(self, lane_idx: int, t: float) -> dict:
        """一条车道的实测交通量: 车数 / 密度(veh/km) / 平均速度 / 通过断面的流量。"""
        lane = self.lanes[lane_idx]
        grp = [v for v in self.vehicles if v.lane == lane_idx]
        n = len(grp)
        km = lane.length / 1000.0 if lane.closed else lane.length / 1000.0
        v_mean = float(np.mean([g.v for g in grp])) if grp else float('nan')
        return {'n': n, 'density_veh_per_km': (n / km) if km > 0 else float('nan'),
                'v_mean_mps': v_mean,
                'flow_veh_per_h': (n / km) * v_mean * 3.6 if grp and np.isfinite(v_mean)
                else float('nan')}

    def _was_radar(self, vid: int) -> bool:
        for v in self.vehicles:
            if v.vid == vid:
                return v.radar is not None
        return False

    # ------------------------------------------------ 初始化
    def _build_static(self):
        s = self.scen
        self.lanes = [Lane(i, np.asarray(l.waypoints, dtype=float), l.lateral_offset_m,
                           l.speed_limit_mps, l.closed)
                      for i, l in enumerate(s.lanes)]
        if s.initial_counts is not None:
            if len(s.initial_counts) != len(self.lanes) or any(
                    not isinstance(v, (int, np.integer)) or v < 0 for v in s.initial_counts):
                raise ValueError("initial_counts needs one nonnegative integer per lane")
        if s.arrival_lanes is not None:
            if len(set(s.arrival_lanes)) != len(s.arrival_lanes) or any(
                    not isinstance(v, (int, np.integer)) or not 0 <= v < len(self.lanes)
                    for v in s.arrival_lanes):
                raise ValueError("arrival_lanes must contain distinct valid lane indices")
        self._radar_vids: set[int] = set()
        # 受害者车辆: 放在 ego 车道上, 初值即设定速度; 跟驰用同一个 IDM (见 step)
        self.ego = Vehicle(vid=1, lane=s.ego_lane, s=s.ego_start_s, v=s.ego_speed_mps,
                           is_ego=True, rcs=0.0, specular_phase=0.0)
        self.vehicles.append(self.ego)
        self.ledger.append((0.0, s.ego_lane, self.ego.vid, 'in', '受害者'))
        # 冲突区: 只用于**度量** (哪里会发生穿模) 与铺点卫生 (不许凭空把车摆在箱子里)。
        # 让行/冲突管理当前**不启用** —— 试过两轮 (接受间隙比较 / 时间窗预约), 都不满足
        # 验收, 详见 simulation/traffic 模块末尾的说明与 docs/结果与实测.md §6.8。
        self._conflicts = [self._conflict_record(c[:2], c[2]) for c in s.conflicts]
        self._inflow_next = {}                  # 开放车道: 下一次入口到达的时刻 (s)
        self.inflow_stats = {'entered': {}, 'blocked': {}}   # 需求到达里进得来/进不来的计数
        for lane_idx, lane in enumerate(self.lanes):
            if not lane.closed and (s.arrival_lanes is None or lane_idx in s.arrival_lanes):
                self._inflow_next[lane_idx] = self.headway(0.0) * self.rng.uniform(0.5, 1.0)
                self.inflow_stats['entered'][lane_idx] = 0
                self.inflow_stats['blocked'][lane_idx] = 0
        for lane_idx, s_m, v_mps, rcs in s.weak_targets:
            # 弱目标 (行人/摩托车量级): 固定低 RCS, 不分配雷达, 钉住不被密度调控删除
            self._new_vehicle(lane_idx, s_m, v_mps, rcs=rcs, radar=False, pinned=True,
                              reason='场景点名的弱目标')
        # 场景点名的车必须**先于**随机铺点放好: 铺点的空隙判据 (min_gap) 才看得见它们,
        # 否则一辆随机车可以正好落在点名干扰车身上 (实测 stage2 抓到 1 例同车道重叠)。
        for lane_idx, s0, v0, rcs, with_radar in s.extra_vehicles:
            self._new_vehicle(lane_idx, s0, v0, rcs=rcs, radar=with_radar, pinned=True,
                              reason='场景点名的干扰车')
        self._seed_traffic()

    def _conflict_record(self, xy, radius: float) -> dict:
        """把冲突区投到每条经过它的车道上: {车道: 弧长} (只用于度量与铺点卫生)。

        参与方判据 = 中心线到区心的最近距离 <= 半径。区心与半径由 scenarios._conflict_zones
        从折点几何算出 (一条穿越道路算一个区, 不是一条车道一个区)。
        """
        per_lane = {}
        for lane in self.lanes:
            if lane.distance_to(xy) <= radius:
                per_lane[lane.index] = lane.s_at(xy)
        return {'pos': (float(xy[0]), float(xy[1])), 'radius_m': float(radius),
                'lane_s': per_lane}

    def _new_vehicle(self, lane_idx: int, s_pos: float, v: float, rcs: float | None = None,
                     radar: bool | None = None, pinned: bool = False,
                     reason: str = '初始铺点') -> Vehicle:
        """按配置分布抽一辆车的 RCS / 相位, 并按需分配雷达身份。

        radar=None 按渗透率概率决定; True/False 用于可控场景里必须出现(或必须不出现)
        的干扰车 / 弱目标。reason 进数量守恒台账 (见 self.ledger)。
        """
        vc = self.scen.vehicle_cfg
        veh = Vehicle(vid=self._new_id(), lane=lane_idx, s=s_pos, v=v,
                      length=vc.length_m,
                      rcs=float(self.property_rng.lognormal(np.log(vc.rcs_mean_m2), vc.rcs_sigma_log))
                      if rcs is None else float(rcs),
                      is_ego=False, pinned=pinned,
                      specular_phase=float(self.property_rng.uniform(0.0, 2 * np.pi)))
        want = (self.property_rng.random() < vc.has_radar_prob) if radar is None else bool(radar)
        if self.scen.interference_enabled and want:
            self.assign_radar(veh, self.t)
            self._radar_vids.add(veh.vid)
        self.vehicles.append(veh)
        self.ledger.append((round(self.t, 6), lane_idx, veh.vid, 'in', reason))
        return veh

    def _seed_traffic(self):
        """t=0 就把每条车道铺到该时刻应有的车辆数。

        铺点用与运行时**同一个** _insertion_point (反复二分当前最大空隙 + 避开受害者
        前方禁区): 结果近似均匀带抖动, 不会出现"开场就比 IDM 稳态更挤"的初始条件,
        也不需要"从路段起点慢慢开进来"的生成过程 —— 那正是上一版密度曲线看不到效果的
        原因 (通行时间 >> 会话时长, 密度降了车辆数反而涨)。

        'flow' 模式的铺点数按 Little 定律给 (N = 段内平均逗留时间 / 入口时距),
        所以初始条件与后面的到达率是同一套参数, 不会出现"开场密度与需求脱钩"。
        这个初始铺点本身进台账 (原因='初始铺点'), 因此 N(t)−N(0) 仍然只由出入口解释。
        """
        for lane_idx, lane in enumerate(self.lanes):
            initial_n = (self.target_count(lane_idx, 0.0) if self.scen.initial_counts is None
                         else self.scen.initial_counts[lane_idx])
            for _ in range(initial_n):
                mine = sorted((v for v in self.vehicles if v.lane == lane_idx), key=lambda x: x.s)
                place = self._insertion_point(mine, lane, lane_idx == self.scen.ego_lane,
                                              jitter=0.12)
                if place is None:
                    break
                self._new_vehicle(lane_idx, place,
                                  self.desired_speed(lane, 0.0) * self.rng.uniform(0.85, 1.05))

    def _new_id(self) -> int:
        self._next_vid += 1
        return self._next_vid

    def assign_radar(self, veh: Vehicle, t_now: float):
        """为干扰车一次分配稳定的雷达身份 (此后不再重抽)。

        抽样顺序是刻意的: 先扫频时长 T (斜率多样性主要由带宽提供, T 默认锁死), 再定
        重复周期 P —— 下界取 max(P_v(1-j), T + min_reset) 就是"这个 chirp 装得下复位"
        这一物理要求本身, 所以**按构造**不会生成发不出来的身份 (旧实现各抽各的, 实测
        10000 台里 4974 台 T+复位 > P)。装不下的 T 记一次拒绝, 最后把拒绝原因分布报出来,
        免得"配置看起来合理、其实一半发不出来"这种死法悄悄发生。
        duration_choices_s 非空时按给定概率抽离散 T; repeat_idle_s 非 None 时
        使用 P=T+idle, 不再受原周期带限制。身份仍然只在入场时分配一次。
        """
        r = self.scen.radar_cfg
        if not np.isfinite(self.victim.slope) or self.victim.slope == 0:
            raise ValueError('victim slope must be finite and nonzero')
        gap = r.min_relative_slope_gap
        if not np.isfinite(gap) or gap <= 0:
            raise ValueError('min_relative_slope_gap must be finite and positive')
        if not np.isfinite(r.min_reset_s) or r.min_reset_s <= 0:
            raise ValueError('min_reset_s must be finite and positive')
        durations = np.asarray(r.duration_choices_s, dtype=float)
        duration_probs = np.asarray(r.duration_choice_probs, dtype=float)
        if durations.ndim != 1 or duration_probs.ndim != 1:
            raise ValueError('duration choices and probabilities must be one-dimensional')
        if durations.size:
            if not np.all(np.isfinite(durations)) or np.any(durations <= 0):
                raise ValueError('duration_choices_s must contain finite positive durations')
            if (duration_probs.size != durations.size
                    or not np.all(np.isfinite(duration_probs))
                    or np.any(duration_probs < 0)
                    or not np.isclose(duration_probs.sum(), 1.0, rtol=0.0, atol=1e-12)):
                raise ValueError('duration_choice_probs must match duration_choices_s, '
                                 'be finite and non-negative, and sum to 1')
            # Normalize only roundoff within the accepted tolerance for NumPy's sampler.
            duration_probs = duration_probs / duration_probs.sum()
        elif duration_probs.size:
            raise ValueError('duration_choice_probs requires nonempty duration_choices_s')
        bounds_to_check = (r.bandwidth_range, r.tx_power_range_dbm)
        if not durations.size:
            bounds_to_check += (r.duration_range,)
        for bounds in bounds_to_check:
            if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or not 0 < bounds[0] <= bounds[1]:
                raise ValueError('duration/bandwidth/power ranges must be '
                                 'finite, positive and ordered')
        off = r.start_offset_range      # 时偏允许 0 起 (0 = 与受害者完全同相开始)
        if (len(off) != 2 or not np.all(np.isfinite(off))
                or not 0 <= off[0] <= off[1]):
            raise ValueError('start_offset_range must be finite, non-negative and ordered')
        if r.repeat_idle_s is not None:
            if (not np.isfinite(r.repeat_idle_s)
                    or r.repeat_idle_s < r.min_reset_s):
                raise ValueError('repeat_idle_s must be finite and at least min_reset_s')
            p_lo = p_hi = None
        elif len(r.period_range_s) == 2:
            if (not np.all(np.isfinite(r.period_range_s))
                    or not 0 < r.period_range_s[0] <= r.period_range_s[1]):
                raise ValueError('period_range_s must be finite, positive and ordered')
            p_lo, p_hi = float(r.period_range_s[0]), float(r.period_range_s[1])
        else:
            j = float(r.chirp_period_jitter)
            if not np.isfinite(j) or not 0.0 <= j < 1.0:
                raise ValueError('chirp_period_jitter must be in [0, 1)')
            p_lo, p_hi = self.victim.chirp_period * (1 - j), self.victim.chirp_period * (1 + j)
        reject = {'周期装不下扫频+复位': 0, '相对斜率差为 0': 0}
        for _ in range(512):
            duration = (float(self.property_rng.choice(durations, p=duration_probs))
                        if durations.size else float(self.property_rng.uniform(*r.duration_range)))
            if r.repeat_idle_s is not None:
                period = duration + float(r.repeat_idle_s)
            else:
                lo = max(p_lo, duration + r.min_reset_s)
                if lo > p_hi:                          # 这个 T 无论怎么抽 P 都发不出来
                    reject['周期装不下扫频+复位'] += 1
                    continue
                period = float(self.property_rng.uniform(lo, p_hi)) if lo < p_hi else p_hi
            sign = 1.0 if self.property_rng.random() < r.slope_sign_probs[0] else -1.0
            bandwidth = float(self.property_rng.uniform(*r.bandwidth_range))
            # 条件采样真实 B/T；周期失配不能代替扫频斜率失配。身份只在入场分配一次。
            if abs(sign * bandwidth / duration - self.victim.slope) / abs(self.victim.slope) < gap:
                reject['相对斜率差为 0'] += 1
                continue
            break
        else:
            why = ', '.join(f'{k} {v} 次' for k, v in reject.items())
            period_desc = (f'周期 = 扫频 + {r.repeat_idle_s * 1e6:.1f} us 空闲'
                           if r.repeat_idle_s is not None
                           else f'周期带 [{p_lo * 1e6:.1f}, {p_hi * 1e6:.1f}] us')
            duration_desc = (f'离散扫频 {list(durations * 1e6)} us' if durations.size
                             else f'扫频带 [{r.duration_range[0] * 1e6:.1f}, '
                                  f'{r.duration_range[1] * 1e6:.1f}] us')
            raise ValueError(
                'Cannot sample a compliant interferer identity within configured ranges '
                f'(512 次全部被拒: {why}; {period_desc}, {duration_desc}, '
                f'复位下限 {r.min_reset_s * 1e6:.1f} us) —— 放宽哪一条要显式决定, 不要偷偷绕过')
        veh.radar = ChirpTimeline(
            radar_id=veh.vid, vehicle_id=veh.vid,
            t_start=self.victim.t_start + self.property_rng.uniform(*r.start_offset_range),
            chirp_period=period,
            sweep_duration=duration,
            bandwidth=bandwidth,
            slope_sign=sign,
            drift_ppm=float(self.property_rng.uniform(-r.clock_drift_ppm, r.clock_drift_ppm)),
            tx_power_w=float(10.0 ** (self.property_rng.uniform(*r.tx_power_range_dbm) / 10.0) / 1000.0),
            fc=self.victim.fc, frame_chirps=r.frame_chirps, frame_period=r.frame_period,
        )

    # ------------------------------------------------ 步进
    def density(self, t: float) -> float:
        """相对密度 0~1 (0 稀疏, 1 拥堵)。同一时刻只做一次插值。

        缓存不是微优化: _idm 与 target_count/desired_speed 每步每车都要读它, 不缓存就是
        每步几千次 np.interp (实测把 40 s 会话的交通积分从 4 s 拖到 11 s)。
        """
        if t != self._d_t:
            self._d_t, self._d_v = t, float(np.clip(lerp_profile(self.scen.density_profile, t),
                                                   0.0, 1.0))
        return self._d_v

    def desired_speed(self, lane: Lane, t: float) -> float:
        d = self.density(t)
        return lane.speed_limit * (1.0 - (1.0 - self.scen.congestion_speed_factor) * d)

    def _idm(self, veh: Vehicle, lead, v_des: float | None = None) -> float:
        """Intelligent Driver Model 纵向加速度 (m/s^2)。

        a = a_max·[1-(v/v0)^δ] - a_max·(s*/gap)²,  s* = s0 + max(0, vT + vΔv/(2√(a·b)))
        出处 Treiber et al. 2000 (cond-mat/0002177)。v_des=None 时按车道限速与密度给
        期望速度; 受害者传自己的设定值, 不被拥堵因子二次压低。
        T 与 target_count 用的是同一个 idm['T'] —— 密度调控铺出来的车数才不会比
        跟驰模型要求的更挤 (两边不一致就会在环上留下永久压缩队列)。
        """
        p = self.scen.idm
        v = max(veh.v, 1e-6)
        v_des = max(self.desired_speed(self.lanes[veh.lane], self.t), 1e-6) \
            if v_des is None else max(v_des, 1e-6)
        a_free = p['a_max'] * (1.0 - (v / v_des) ** p['delta'])
        if lead is None:
            return a_free
        leader, gap = lead
        dv = v - leader.v
        s_star = p['s0'] + max(0.0, v * p['T'] + v * dv /
                               (2.0 * np.sqrt(p['a_max'] * p['b_comfort'])))
        return a_free - p['a_max'] * (s_star / gap) ** 2

    def _leaders(self) -> dict:
        """每辆车的前车与**净间距** (闭合环上每辆车都有前车; 开放路段最后一辆没有)。

        返回 {vid: (leader, gap)}; gap 已扣掉本车车长, 下限 0.5 m 防除零。
        开放车道不能绕回: 让"最后一个"看见"第一个"会让车流自己凭空刹住。
        """
        by_lane: dict[int, list[Vehicle]] = {}
        for veh in self.vehicles:
            by_lane.setdefault(veh.lane, []).append(veh)
        out = {}
        for lane_idx, group in by_lane.items():
            if len(group) < 2:
                continue
            L = self.lanes[lane_idx].length
            closed = self.lanes[lane_idx].closed
            group.sort(key=lambda x: x.s)
            for i, veh in enumerate(group):
                if closed:
                    nxt = group[(i + 1) % len(group)]
                    if nxt is veh:
                        continue
                    gap = (nxt.s - veh.s) % L if L > 0 else nxt.s - veh.s
                else:
                    if i + 1 >= len(group):
                        continue              # 开路最下游的车: 自由流
                    nxt, gap = group[i + 1], group[i + 1].s - veh.s
                out[veh.vid] = (nxt, max(gap - veh.length, 0.5))
        return out

    def headway(self, t: float) -> float:
        """'flow' 模式的入口到达时距 h(d) (s): d=0 取 max_headway_s (疏), d=1 取 min_headway_s (堵)。"""
        c = self.scen
        return float(c.max_headway_s + (c.min_headway_s - c.max_headway_s) * self.density(t))

    def target_count(self, lane_idx: int, t: float) -> int:
        """该车道此刻"应有几辆车" —— 三种车流来源 (population_mode) 给三种算法。

        'fixed'   : 直接返回 fixed_counts, 全程不变 —— 受控基准, 用来回答"没有任何
                    凭空增删时车流自己会怎么样"。
        'flow'    : 按 **Little 定律** N = q·T_res = (1/h)·(L/v) 换算: 入口时距 h(d) 决定
                    到达率, 期望速度决定逗留时间。它只用于 t=0 的初始铺点 (以及初始铺点
                    与后面到达率的一致性), 运行中的车数变化一律由真实出入口驱动。
        'density' : n = floor(环长 / (s0 + 车长 + v0(d)·T))。
                    ⚠ 这**不是** IDM 在 v0(d) 下的平衡间距: 平衡条件
                    (s0 + v·T)/s_净 = [1 − (v/v0)³]^(1/δ) 在 v→v0 时右边趋于 0, 也就是
                    "以 v0 跑"的平衡间距是无穷大, 环上根本放不出那种车数。所以这里走的是
                    另一种口径 —— **给定 N 反解平衡速度** v_eq < v0(d):
                        s0 + v_eq·T = s_净 · [1 − (v_eq/v0)³]^(1/δ)
                    "密度旋钮决定车数", v_eq 只是这个车数对应的自由流速度, 它是被检查的
                    预测量而不是被满足的约束 (stage2 把两者并排列出: highway_overtake 三档
                    实测中位 23.2 / 8.4 / 16.4 m/s, 对应 v_eq 22.1 / 8.39 / 20.6 m/s)。
                    这样密度直接决定路上的车数, 而不是只改一个没人执行的生成率参数 ——
                    上一版的开放路段 + 生成器就是这个问题 (密度降了车数反增)。

        'density' 里两处都是刻意选择, 少一处就会把车流逼成压缩态:
          * 用 **floor** 而不是 round: 多塞一辆就等于要求比 T 更小的时距, IDM 会立刻
            刹车去消除它, 环上传不出去就变成永久拥堵。
          * T 只此一个来源: 早版把"稀疏车头时距 4 s"这种*到达间隔*当成 IDM 的期望时距,
            26 m/s 下要求 91 m 间距, 结果疏段实测中位间距 100 m、堵段直接锁死 (中位 0 m/s)。
        """
        c = self.scen
        p = c.idm
        lane = self.lanes[lane_idx]
        if c.population_mode == 'fixed':
            return int(c.fixed_counts[lane_idx]) if lane_idx < len(c.fixed_counts) else 0
        v = max(self.desired_speed(lane, t), 1.0)
        if c.population_mode == 'flow':
            return int(lane.length // max(v * self.headway(t), 1e-6))
        spacing = p['s0'] + c.vehicle_cfg.length_m + v * p['T']
        return int(lane.length // max(spacing, p['s0'] + c.vehicle_cfg.length_m))

    def _maintain_population(self):
        """'density' 模式的目标车数跟踪: 加在空隙处, 删在最挤处。

        只有这一种模式会在**场内**凭空增删车辆 —— 'fixed' 全程零增删, 'flow' 的增删只
        发生在段首/段尾。台账 (self.ledger) 把每一条都记名, 所以"车流演化里有多少是控制
        器造的"是一个可查的数而不是印象 (实测 40 s 会话: 68 进 64 出, 其中 36 台带雷达)。
        """
        if self.scen.population_mode != 'density':
            return
        for lane_idx, lane in enumerate(self.lanes):
            if not lane.closed:
                continue              # 开放车道的车数由出入口管, 控制器不插手
            mine = sorted((v for v in self.vehicles if v.lane == lane_idx), key=lambda x: x.s)
            want = self.target_count(lane_idx, self.t) + (1 if lane_idx == self.scen.ego_lane else 0)
            while len(mine) < want:
                place = self._insertion_point(mine, lane, lane_idx == self.scen.ego_lane)
                if place is None:
                    break                 # 只剩下受害者前面的禁区, 宁可不补车
                v = self._new_vehicle(lane_idx, place,
                                      self.desired_speed(lane, self.t) * self.rng.uniform(0.85, 1.05),
                                      reason='密度上升')
                mine.append(v)
                mine.sort(key=lambda x: x.s)
            while len(mine) > want:
                drop = self._crowded_vehicle(mine, lane.length)
                if drop is None:
                    break
                self.vehicles.remove(drop)
                mine.remove(drop)
                self.ledger.append((round(self.t, 6), lane_idx, drop.vid, 'out', '密度下降'))

    def _insertion_point(self, sorted_veh, lane, avoid_ego: bool, jitter: float = 0.0):
        """补车的位置: 最大的那段空隙的中点 (环上按弧长绕回), 可加 ±jitter·间隙 的抖动。

        avoid_ego 时丢掉落在"受害者前方 IDM 稳态间距"里的候选点 —— 凭空出现在本车
        刹车距离里的车不是交通, 是仪器扰动: 它会让受害者一步踩出远超 a_max 的减速度,
        直接污染多普勒与几何变化 (tests/test_scenario_fidelity.py 钉住这点)。
        """
        n = len(sorted_veh)
        if n == 0:
            return float(self.rng.uniform(0.0, max(lane.length, 1.0)))
        L = lane.length
        p = self.scen.idm
        safe = p['s0'] + self.scen.vehicle_cfg.length_m + max(self.ego.v, 1.0) * p['T']
        # 空隙至少要放得下一辆车 + s0 的静止余量, 否则"补车"本身就是穿模: 补进 6 m 的
        # 空隙等于把新车摆在两辆车之间各 3 m 处 (实测 stage2 抓到 1 例同车道重叠)。
        # 宁可不补 (实测车数低于目标, 由 stage2 的"实测/目标"列暴露) 也不凭空造重叠。
        min_gap = self.scen.vehicle_cfg.length_m + p['s0']
        best, best_gap = None, -1.0
        for i, veh in enumerate(sorted_veh):
            gap = L if n == 1 else (sorted_veh[(i + 1) % n].s - veh.s) % L
            if gap < min_gap:
                continue
            cand = (veh.s + gap * (0.5 + self.rng.uniform(-jitter, jitter))) % L
            if avoid_ego and (cand - self.ego.s) % L < safe:
                continue                  # 候选点落在本车前方 safe 米内
            if self._in_conflict_box(lane.index, cand):
                continue                  # 不许凭空把车造在交叉口箱体里 (几何上就是穿模)
            if gap > best_gap:
                best, best_gap = cand, gap
        return None if best is None else float(best)

    def _in_conflict_box(self, lane_idx: int, s_pos: float) -> bool:
        """弧长 s_pos 落在不在本车道的某个冲突区 (含区前 s0+车长的停车余量) 里。

        必须用**带符号的最短弧长差**: 用 (s - s_cp) % L 的话, "在区心前 0.1 m"会算成
        "在区心后一整圈", 于是铺点时正好压在区心上的车一律漏判 —— 实测 intersection_turn
        一开场就有两对车在箱子里互相穿过, 之后整个会话都被这对"穿模"污染。
        """
        lane = self.lanes[lane_idx]
        m = self.scen.idm['s0'] + self.scen.vehicle_cfg.length_m
        for c in self._conflicts:
            s_cp = c['lane_s'].get(lane_idx)
            if s_cp is None:
                continue
            d = (s_pos - s_cp) % lane.length
            d = d - lane.length if d > lane.length / 2.0 else d      # 带符号最短差
            if abs(d) <= c['radius_m'] + m:
                return True
        return False

    def _crowded_vehicle(self, sorted_veh, length):
        """消散时该删谁: **前间距最小**的那辆可删车 (车队里最挤的一处)。

        删在最大空隙处等于把车从已经不堵的路段抽走, 拥堵团原封不动地留在环上 ——
        环线上没有上下游可以排水, 这种"压缩团"是亚稳态, 实测会一直堵到会话结束
        (14 辆车的平均间隙有 35 m, 中位间隙却只有 7 m)。删在最挤的地方才真的把
        队列松开。受害者与场景点名车辆 (pinned) 不可删。
        """
        cand = [(((sorted_veh[(i + 1) % len(sorted_veh)].s - v.s) % length), i)
                for i, v in enumerate(sorted_veh) if not (v.is_ego or v.pinned)]
        return sorted_veh[min(cand)[1]] if cand else None

    # ------------------------------------------------ 开放路段的真实出入口
    def _entrance_free(self, lane_idx: int, lane: Lane) -> bool:
        """段首是否空得下一辆新车 (否则本次到达记为"受阻", 不硬塞出重叠车)。"""
        p, L = self.scen.idm, self.scen.vehicle_cfg.length_m
        need = p['s0'] + L + 1.0
        s_min = min((v.s for v in self.vehicles if v.lane == lane_idx), default=np.inf)
        return bool(s_min > need)

    def _open_lane_exits(self, lane_idx: int, lane: Lane):
        for veh in [v for v in self.vehicles
                    if v.lane == lane_idx and not v.is_ego and v.s >= lane.length]:
            self.vehicles.remove(veh)
            self.ledger.append((round(self.t, 6), lane_idx, veh.vid, 'out', '驶出下游端'))
        if lane_idx == self.scen.ego_lane and self.ego.s >= lane.length:
            # 受害者开出观测段 = 会话后半程没有雷达了, 而且它是"凭空消失"的那种。
            # 路段长度必须按 ego_start_s + v·时长 反推 (见 scenarios.corridor_in_out), 越界
            # 就是场景配错了, 直接报错比悄悄把受害者留在段外诚实。
            raise ValueError(
                f'受害者已开出观测段 (t={self.t:.2f} s, s={self.ego.s:.1f} m > 段长 '
                f'{lane.length:.1f} m): 开放路段至少要长 '
                f'{self.scen.ego_start_s + self.scen.duration_s * self.ego.v:.0f} m')

    def _open_lane_arrivals(self, lane_idx: int, lane: Lane):
        """按 h(d) 的时刻表在段首放车; 时刻表**照走**即使进不来 (受阻只计数, 不排队)。

        这样"实测通过量 ≤ 需求"成为拥堵的真实后果, 而不是被生成率参数保证的结果。
        到达间隔乘 U(0.7,1.3) 的抖动: 定长时距等于挂钟车流, 也不符合真实到达的随机性
        (分布本身是工程选择, 均值得到 h(d) —— 没有论文支持具体形状, 别把它当模型)。
        """
        while self.t + 1e-12 >= self._inflow_next[lane_idx]:
            if self._entrance_free(lane_idx, lane):
                self._new_vehicle(lane_idx, 0.0,
                                  self.desired_speed(lane, self.t) * self.rng.uniform(0.9, 1.0),
                                  reason='入口到达')
                self.inflow_stats['entered'][lane_idx] += 1
            else:
                self.inflow_stats['blocked'][lane_idx] += 1
            self._inflow_next[lane_idx] += self.headway(self.t) * self.rng.uniform(0.7, 1.3)
            if self._inflow_next[lane_idx] < self.t - 30 * self.scen.dt_traffic:
                self._inflow_next[lane_idx] = self.t      # 段首长期占死: 不补放历史到达

    def _refresh_accelerations(self):
        leaders = self._leaders()
        for veh in self.vehicles:
            veh.a = self._idm(veh, leaders.get(veh.vid),
                             v_des=self.scen.ego_speed_mps) if veh.is_ego else self._idm(veh, leaders.get(veh.vid))

    @staticmethod
    def _motion(v, acceleration, dt):
        # Constant acceleration between driver updates; never reverse after braking.
        moving = min(dt, -v / acceleration) if acceleration < 0 and v + acceleration*dt < 0 else dt
        return v*moving + .5*acceleration*moving*moving, max(0., v + acceleration*dt)

    def step(self, dt: float):
        if not np.isfinite(dt) or dt <= 0:raise ValueError('Positive finite traffic step required')
        self._refresh_accelerations()
        hard = self.scen.idm['b_comfort'] * self.scen.hard_brake_factor
        self.n_hard_brake += sum(1 for veh in self.vehicles if veh.a < -hard)
        self.n_vehicle_steps += len(self.vehicles)
        for veh in self.vehicles:
            distance, veh.v = self._motion(veh.v, veh.a, dt)
            veh.s = self.lanes[veh.lane].wrap(veh.s + distance)
        self.t += dt
        if self.scen.population_mode == 'flow':
            for lane_idx, lane in enumerate(self.lanes):
                if not lane.closed:
                    self._open_lane_exits(lane_idx, lane)
                    if lane_idx in self._inflow_next:
                        self._open_lane_arrivals(lane_idx, lane)
        self._maintain_population()
        self._refresh_accelerations()

    def advance_to(self, t: float):
        """Commit ONLY complete fixed-grid steps; queries never change driver cadence.

        self.t is the last committed traffic tick, <= requested t. snapshot projects
        between ticks without creating arrivals or drawing random numbers.
        """
        if not np.isfinite(t) or t < self.t - 1e-12:
            raise ValueError('Traffic query must be finite and not precede committed state')
        grid = self.scen.dt_traffic
        target = int(np.floor((t + 1e-12) / grid))
        while self.t < target*grid - 1e-12:
            next_tick = (int(np.floor((self.t+1e-12)/grid)) + 1)*grid
            self.step(next_tick-self.t)
            self.t = next_tick

    def snapshot(self, t: float) -> tuple[VehicleState, list[VehicleState]]:
        self.advance_to(t)
        dt = max(0., t - self.t)
        def state(veh):
            lane = self.lanes[veh.lane]
            distance, velocity = self._motion(veh.v, veh.a, dt)
            position = veh.s + distance
            tang = lane.tangent(position)
            return VehicleState(vid=veh.vid, lane=veh.lane,
                                pos=lane.position(position), vel=tang*velocity,
                                heading=float(np.arctan2(tang[1], tang[0])),
                                rcs=veh.rcs, is_ego=veh.is_ego, radar=veh.radar,
                                specular_phase=veh.specular_phase)
        return state(self.ego), [state(v) for v in self.vehicles if not v.is_ego]

    def radar_count(self) -> int:
        return sum(1 for v in self.vehicles if v.radar is not None and not v.is_ego)
