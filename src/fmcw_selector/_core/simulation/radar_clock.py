# simulation/radar_clock.py
"""雷达时序层: 每部雷达自己的 chirp 时钟 (独立、一次性分配、之后按时钟运行)。

受害者与每台干扰雷达都用同一个 ChirpTimeline 描述:
    第 m 个 chirp 的起始时刻  t_m = t_start + m * Tc * (1 + eps)
    Tc  = chirp 重复周期, eps = 归一化时钟漂移 (ppm/1e6)

相位约定 (解析延拓): 扫频相位 φ(t) = 2π [ f0*u + 0.5*K*u^2 ], u = t - t_m。
在空闲段 (u > T) 上该式继续外推, 只在 `transmitting` 为真的时段才把信号注入接收机。
本模型采用 ramp-local 相位基准, 不声称发射相位跨复位连续; 发射通断由时偏/周期差决定 —— 相邻 chirp 的
重叠情况因此会自然变化, 而不是被人为平滑。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C = 299792458.0


@dataclass
class ChirpTimeline:
    """一部雷达的发射时钟。

    字段:
      radar_id / vehicle_id : 稳定的雷达身份 (进场景时分配, 整个会话不变)
      t_start               : 第一个 chirp 起始时刻 (s, 会话绝对时间)
      chirp_period          : Tc (s); 采样时长 sweep_duration (s) 后的部分是空闲
      bandwidth             : 扫频带宽 (Hz), 符号由 slope_sign 决定
      slope_sign            : +1 上扫频, -1 下扫频
      drift_ppm             : 时钟漂移 (ppm)
      tx_power_w            : 发射功率 (W)
      fc                    : 中心频率 (Hz), 各雷达公称相同
    """
    radar_id: int
    vehicle_id: int
    t_start: float
    chirp_period: float
    sweep_duration: float
    bandwidth: float
    slope_sign: float = 1.0
    drift_ppm: float = 0.0
    tx_power_w: float = 0.01
    fc: float = 77e9
    frame_chirps: int = 0
    frame_period: float | None = None   # None: legacy uninterrupted chirp train

    def __post_init__(self):
        if not np.isfinite(self.period) or self.period <= 0:
            raise ValueError('Positive finite radar period required')
        if self.frame_period is not None:
            if self.frame_chirps < 1 or int(self.frame_chirps) != self.frame_chirps:
                raise ValueError('Positive integer chirps per frame required')
            if not np.isfinite(self.frame_period) or self.frame_period < self.frame_chirps*self.chirp_period:
                raise ValueError('Frame must contain all chirp slots and their reset intervals')

    @property
    def frame_repeat(self):
        return None if self.frame_period is None else self.frame_period*(1+self.drift_ppm*1e-6)

    @staticmethod
    def _tick(q):
        rounded = np.rint(q)
        return np.floor(np.where(abs(q-rounded)<1e-10, rounded, q))

    def chirp_start(self, index):
        index = np.asarray(index)
        if self.frame_period is None:
            return self.t_start + index*self.period
        return (self.t_start + np.floor_divide(index,self.frame_chirps)*self.frame_repeat
                + np.mod(index,self.frame_chirps)*self.period)

    def first_index_at_or_after(self, t):
        if t <= self.t_start:
            return 0
        index = int(self.chirp_index(t))
        return index if self.chirp_start(index) >= t-1e-12 else index+1

    @property
    def period(self) -> float:
        return self.chirp_period * (1.0 + self.drift_ppm * 1e-6)

    @property
    def slope(self) -> float:
        """调频斜率 K (Hz/s), 含扫频方向符号。"""
        return self.slope_sign * self.bandwidth / self.sweep_duration

    @property
    def f0(self) -> float:
        """起始频率 (Hz): 上扫频从 fc - B/2 起, 下扫频从 fc + B/2 起。"""
        return self.fc - self.slope_sign * self.bandwidth / 2.0

    def chirp_index(self, t):
        relative = np.asarray(t,dtype=float)-self.t_start
        if self.frame_period is None:return self._tick(relative/self.period)
        frame = self._tick(relative/self.frame_repeat)
        slot = np.minimum(self._tick((relative-frame*self.frame_repeat)/self.period), self.frame_chirps-1)
        return frame*self.frame_chirps + slot

    def local_time(self, t):
        """u = t - t_m: 相对本 chirp 起点的时间 (s), 空闲段按外推继续增长。"""
        t = np.asarray(t, dtype=float)
        return t - self.chirp_start(self.chirp_index(t))

    def transmitting(self, t) -> np.ndarray:
        """该时刻是否真的在发射 (bool 数组)。"""
        u = self.local_time(t)
        return (np.asarray(t) >= self.t_start) & (u >= -1e-12) & (u < self.sweep_duration)

    def tx_phase(self, t) -> np.ndarray:
        """发射相位 (rad), 外推到空闲段。"""
        u = self.local_time(t)
        return 2.0 * np.pi * (self.f0 * u + 0.5 * self.slope * u ** 2)

    def instantaneous_freq(self, t) -> np.ndarray:
        """标称瞬时发射频率 (Hz)。"""
        return self.f0 + self.slope * self.local_time(t)
