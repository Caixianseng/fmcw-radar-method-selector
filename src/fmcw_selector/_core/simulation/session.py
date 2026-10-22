# simulation/session.py
"""会话驱动: 连续交通场景 -> 逐 chirp 的观测帧流。

一个会话按时间顺序分成若干"处理窗口" (run)。窗口之间交通照常演化但不出帧,
这样既能覆盖"稀疏 -> 拥堵 -> 变疏"的全过程 (出厂会话 40 s = 40 万 chirp), 又不用
把整段都送进算法链。划分训练/标定时必须按 run (整段场景片段) 分, 不能把相邻 chirp
随机拆开 (技术方案第六节的实验要求); 窗口位置由 phase_windows() 按密度折点给。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fmcw_selector._core.common.types import ChirpFrame
from fmcw_selector._core.configs import SystemConfig
from fmcw_selector._core.configs.traffic import ScenarioConfig
from fmcw_selector._core.simulation.radar_clock import ChirpTimeline
from fmcw_selector._core.simulation.receiver import VictimReceiver
from fmcw_selector._core.simulation.traffic import TrafficFlow


@dataclass
class Run:
    """一个连续处理窗口。"""
    name: str
    t_start: float          # 起始时刻 (s)
    n_chirps: int           # 窗口内 chirp 数

    def end_at(self, timeline: ChirpTimeline) -> float:
        """Next acquisition start on the actual clock; includes inter-frame idle."""
        first = timeline.first_index_at_or_after(self.t_start)
        return float(timeline.chirp_start(first + self.n_chirps))


class TrafficSession:
    """受害者视角的连续会话。

    Args:
        scen: 场景配置
        cfg: 系统配置 (雷达/算法)
        seed: 随机种子 (车流、RCS、相位、噪声、干扰雷达参数)
    """

    def __init__(self, scen: ScenarioConfig, cfg: SystemConfig | None = None,
                 seed: int = 0, receiver_factory=VictimReceiver):
        self.scen = scen
        self.cfg = cfg or SystemConfig()
        traffic_seed, property_seed, noise_seed = np.random.SeedSequence(seed).spawn(3)
        self.rng = np.random.default_rng(traffic_seed)
        self.property_rng = np.random.default_rng(property_seed)
        self.noise_rng = np.random.default_rng(noise_seed)
        self.random_stream_version = 'traffic_properties_noise_independent_v1' 
        rc = self.cfg.radar
        self.victim = ChirpTimeline(radar_id=0, vehicle_id=1, t_start=0.0,
                                    chirp_period=rc.chirp_period,
                                    sweep_duration=rc.chirp_duration,
                                    bandwidth=rc.bandwidth, slope_sign=1.0,
                                    drift_ppm=0.0, tx_power_w=rc.tx_power_w, fc=rc.fc,
                                    frame_chirps=rc.frame_chirps,frame_period=rc.frame_period)
        self.flow = TrafficFlow(scen, self.victim, self.rng, self.property_rng)
        # 遮挡是**两档敏感性对比**的第二档: 默认关 (None = 完全无遮挡), 打开时车身被当成
        # 长 x 宽的矩形, 挡住的目标不注入回波/干扰 (见 simulation/occlusion.py)
        self.receiver = receiver_factory(self.cfg, self.victim, self.noise_rng,
                                       occluder_size=((scen.vehicle_cfg.length_m,
                                                       scen.vehicle_cfg.occluder_width_m)
                                                      if scen.occlusion_enabled else None))
        self.t_cursor = 0.0

    # ------------------------------------------------------------ 出帧
    def run(self, run: Run):
        """推进交通到窗口起点, 然后逐 chirp 产出帧 (生成器)。

        多次调用必须时间单调 (共享同一个 TrafficFlow 状态), 否则报错而不是悄悄重来。
        """
        if not np.isfinite(run.t_start) or run.n_chirps < 1 or int(run.n_chirps) != run.n_chirps:
            raise ValueError('Finite run start and positive integer chirp count required')
        if run.t_start < self.t_cursor - 1e-12:
            raise ValueError(f'run {run.name} 起点 {run.t_start} 早于当前游标 {self.t_cursor}; '
                             f'会话状态是连续的, 不能回退时间')
        m0 = self.victim.first_index_at_or_after(run.t_start)
        self.t_cursor = float(self.victim.chirp_start(m0))
        self.flow.advance_to(self.t_cursor)
        for k in range(run.n_chirps):
            m = m0 + k
            t0 = float(self.victim.chirp_start(m))
            ego, others = self.flow.snapshot(t0)
            yield self.receiver.synthesize(t0, ego, others, m)
        self.t_cursor = float(self.victim.chirp_start(m0+run.n_chirps))

    def run_end(self, run: Run) -> float:
        return run.end_at(self.victim)

    def chirp_start(self, m: int) -> float:
        return self.receiver.chirp_start(m)

    # ------------------------------------------------------------ 统计
    def describe_run(self, run: Run) -> dict:
        """一个窗口的场景构成 (离线诊断用, 不喂给算法)。"""
        self.flow.advance_to(run.t_start)
        ego, others = self.flow.snapshot(run.t_start)
        radars = [v for v in others if v.radar is not None]
        return {'run': run.name, 't_start': run.t_start, 'n_vehicles': len(others),
                'n_radar_cars': len(radars),
                'density': self.flow.density(run.t_start)}


def phase_windows(scen: ScenarioConfig, n_chirps: int, chirp_period: float,
                  guard_s: float = 0.5, take: int = 0, block: int = 1,
                  frame_chirps: int = 0, frame_period: float | None = None) -> list[Run]:
    """在密度折线的每一段中点各开一个处理窗口, 返回按时间递增的 Run 列表。

    为什么按密度段而不是按会话等分: 会话 40 s 里"疏 -> 密 -> 堵 -> 散"是折点定义的,
    等分会把窗口骑在折点上, 于是同一个窗口里前半稀疏后半拥堵 —— 想按交通状态分组
    评价 (拥堵段长缺口多) 就分不干净。窗口必须整段落在同一条折线内 (两端各让 guard_s),
    放不下就跳过这一段, 不硬塞。

    block: 把窗口 chirp 数**向下取整到它的整数倍**。距离-多普勒评价以帧 (block 个 chirp)
    为单位, 不整除就会在窗口边界留下半帧; 而跨窗口拼帧在时间上不连续 (中间的交通被跳过了),
    那种"帧"既不是真实积累也不该用来算 Pd。
    take>0 时再从这些窗口里均匀抽 take 个 (跑不动全量时用, 保证仍覆盖首尾两端)。
    """
    block = max(int(block), 1)
    n_chirps = int(n_chirps) // block * block
    if n_chirps < block:
        return []
    pts = np.asarray(scen.density_profile, dtype=float)
    timeline = (ChirpTimeline(0, 0, 0., chirp_period, chirp_period, 1.,
                             frame_chirps=frame_chirps, frame_period=frame_period)
                if frame_period is not None else None)
    span = (int(np.ceil(n_chirps/frame_chirps))*frame_period if timeline else n_chirps*chirp_period)
    runs = []
    for (t0, _), (t1, _) in zip(pts[:-1], pts[1:]):
        mid = 0.5 * (t0 + t1)
        start = mid - span/2.
        if timeline is not None:
            first = timeline.first_index_at_or_after(start)
            # Begin on a real TX frame, never collect a partial frame at a window edge.
            first = int(np.ceil(first/frame_chirps))*frame_chirps
            start = float(timeline.chirp_start(first))
            end = float(timeline.chirp_start(first+n_chirps))
        else:
            end = start+span
        if start > t0 + guard_s and end < min(t1, scen.duration_s) - guard_s:
            runs.append(Run(f'w{len(runs)}_d{np.interp(mid, pts[:, 0], pts[:, 1]):.2f}'
                            f'@{mid:.0f}s', start, n_chirps))
    if 0 < take < len(runs):
        idx = sorted(set(np.linspace(0, len(runs) - 1, take).round().astype(int)))
        runs = [runs[i] for i in idx]
    return runs
