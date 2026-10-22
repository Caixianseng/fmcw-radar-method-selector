# simulation/propagation.py
"""几何、天线朝向与链路预算 (接收信号层的物理输入)。

所有量都用功率 (W) 算, 再取平方根得到复基带采样幅度:
    目标回波  功率 ∝ R^-4  ->  幅度 ∝ R^-2
    直接干扰  功率 ∝ R^-2  ->  幅度 ∝ R^-1
旧实现直接把 R^-4 / R^-2 当作幅度用, 会把远近强弱关系压平一个平方, 这里改正。
"""

from __future__ import annotations

import numpy as np

from fmcw_selector._core.configs.radar import RadarConfig
from fmcw_selector._core.simulation.traffic import VehicleState


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    """二维叉积的 z 分量 (numpy 的 np.cross 对 2 维向量已弃用)。"""
    return float(a[0] * b[1] - a[1] * b[0])


def aspect(state_from: VehicleState, state_to: VehicleState):
    """从 state_from (观察者) 看 state_to 的几何量。

    Returns:
        range_m      瞬时距离 (m)
        closing_mps  接近速度 (m/s), 为正表示距离在缩小
        ang_obs      目标相对观察者艏向的偏角 (rad) —— 观察者接收天线朝向
        ang_tgt      观察者相对目标艏向的偏角 (rad) —— 目标发射天线朝向
    """
    d = state_to.pos - state_from.pos
    r = float(np.linalg.norm(d))
    if r < 1e-6:
        r = 1e-6
    u = d / r
    dv = state_to.vel - state_from.vel
    closing = -float(np.dot(dv, u))
    h_obs = np.array([np.cos(state_from.heading), np.sin(state_from.heading)])
    h_tgt = np.array([np.cos(state_to.heading), np.sin(state_to.heading)])
    ang_obs = float(np.arctan2(_cross2(h_obs, u), np.dot(h_obs, u)))
    ang_tgt = float(np.arctan2(_cross2(h_tgt, -u), np.dot(h_tgt, -u)))
    return r, closing, ang_obs, ang_tgt


class LinkBudget:
    """把 (距离, 相对速度, 天线朝向) 变成复基带采样幅度 (单位: sqrt(W))。"""

    def __init__(self, radar: RadarConfig):
        self.rc = radar

    def target_amplitude(self, range_m: float, rcs_m2: float, ang_obs: float) -> float:
        """单站目标回波的采样幅度 = sqrt(接收功率)。

        发射与接收都用观察者自己的方向图 (单站双程), 因此是 g(ang_obs)^2。
        目标自身朝向对散射强度的影响已折进 RCS 的量级散布里, 不再单独建模。
        """
        g = float(self.rc.antenna_gain(ang_obs))
        p = self.rc.echo_power_w(range_m, rcs_m2, g, g)
        return float(np.sqrt(p))

    def interference_amplitude(self, range_m: float, tx_power_w: float,
                               ang_obs: float, ang_tgt: float) -> float:
        """直接干扰的采样幅度 = sqrt(接收功率), 功率 ∝ R^-2。

        g(ang_obs): 受害者接收天线朝向干扰机的增益;
        g(ang_tgt): 干扰机发射天线朝向受害者的增益 (对向车因此吃到主瓣)。
        """
        p = self.rc.interference_power_w(range_m, float(self.rc.antenna_gain(ang_tgt)),
                                         float(self.rc.antenna_gain(ang_obs)), tx_power_w)
        return float(np.sqrt(p))
