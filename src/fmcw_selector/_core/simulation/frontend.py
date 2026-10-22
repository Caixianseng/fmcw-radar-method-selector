# simulation/frontend.py
"""显式接收前端状态模型 (M1-A): 把 idle/LO/ADC/DFE 的硬件行为声明写成可对拍的代码。

背景 (docs/整个项目优化方案_2026-10-04.md §四.1)：
`simulation/receiver.py` 原来把 clean/interf/noise **每 chirp 各自**做一次零状态 SOS 低通，
即"数字/模拟前端在每个 chirp 开始时都处于零状态"。M0-C 已经量出：在本体制
(fs=50 MHz、通带 20 MHz、3 阶椭圆、空闲 25 µs) 下，1% 容差的稳定时间只有 17 点 = 340 ns，
所以"每 chirp 复位"在**数字级**是合理近似。它没有覆盖的是:

  * 持续供电的模拟前端在 idle 期间还在演化；
  * LO 在 idle 期间是否继续工作 (决定 idle 里有没有干扰差拍)；
  * ADC 在 idle 期间是否采样 (决定 DFE 能不能看到 idle 输入)；
  * AGC 的时间常数、限幅和可读取增益 (原来是每 chirp 瞬时归一化, 没有状态)。

本模块把这些逐一变成**声明**，并提供两种命名预设：

  * ``per_chirp_reset``：当前生产行为。模拟前端按无记忆处理 (``receiver._anti_alias``
    只给幅频), DFE 每 chirp 零状态启动, AGC 每 chirp RMS 归一化。逐位复现旧输出。
  * ``continuous``：连续模拟前端 + DFE 不复位。idle 期间滤波器状态**不跳过**：用声明的
    idle 输入 (zeros/noise/interferer) 继续递推；chirp 之间状态通过 ``:class:`FrontendChain`
    传递。

严格说清楚本模型**没有**做到的事：

  1. 只有一级有状态 SOS 滤波器。它同时代表模拟抗混叠和数字抽取链 (两者在本体制里
     同截止/同阻带)，所以 ``analog_continuous=True`` 与 ``dfe_reset_per_chirp=False``
     必须一起取 —— 不允许"模拟连续但 DFE 复位"这种本模型无法表达的组合。
  2. 没有独立参考接收机或实测数据，所有结论只能叫**解析一致性**，不能叫实测验证。
  3. AGC 的新状态模型作用在 ADC 之后的归一化尺度上 (align with ``ChirpFrame.agc_scale``),
     PGA/削顶顺序本轮不动。

对拍约定 (验收, 方案 §四.1)：把 [chirp0, idle0, chirp1, idle1, ...] 拼成一条流，
`scipy.signal.sosfilt` 一次通过的结果与逐块带状态递推必须逐位一致。这是状态记账的
自洽性检查，不是物理验证；物理泛化要等 SUMO/FCD 真实轨迹 (M1-B)。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import sosfilt

#: 前端模式命名。加新模式必须同时给它写 idle/LO/ADC/DFE 四项声明，不许只改一个布尔。
FRONTEND_MODES = ('per_chirp_reset', 'continuous')

#: idle 期间驱动状态滤波器的输入声明。'zeros' = 门控/保持零输入;
#: 'noise' = 只有热噪声; 'interferer' = 完整 IF (含仍在发射的干扰差拍)。
IDLE_INPUTS = ('zeros', 'noise', 'interferer')

#: AGC 模式。'per_chirp_rms' 是旧口径 (每次调用瞬时归一化, 无状态);
#: 'tracking_db' 是一阶跟踪环 (有时间常数/限幅/可读取增益)。
AGC_MODES = ('per_chirp_rms', 'tracking_db')

#: 零输入演化提前停止的判据: 状态范数低于该绝对阈值就认为已经衰减到零。
#: 阈值取机器 eps 量级的平方附近, 保证不会把还有实际影响的振铃当成零。
_STATE_ZERO_TOL = 1e-24


@dataclass(frozen=True)
class FrontendSpec:
    """一条完整的前端硬件行为声明。

    字段分组:
      * 连续性: ``analog_continuous`` / ``dfe_reset_per_chirp``
      * idle 输入: ``lo_running_during_idle`` / ``adc_gated_during_idle`` / ``idle_input``
      * AGC: ``agc_mode`` / ``agc_tau_s`` / 增益限幅
    """

    mode: str = 'per_chirp_reset'
    analog_continuous: bool = False
    lo_running_during_idle: bool = True
    adc_gated_during_idle: bool = True
    dfe_reset_per_chirp: bool = True
    idle_input: str = 'zeros'
    agc_mode: str = 'per_chirp_rms'
    agc_tau_s: float = 1e-3
    agc_gain_min_db: float = -30.0
    agc_gain_max_db: float = 30.0

    def __post_init__(self):
        if self.mode not in FRONTEND_MODES:
            raise ValueError(f'未知前端模式 {self.mode!r}, 只允许 {FRONTEND_MODES}')
        if self.idle_input not in IDLE_INPUTS:
            raise ValueError(f'未知 idle 输入 {self.idle_input!r}, 只允许 {IDLE_INPUTS}')
        if self.agc_mode not in AGC_MODES:
            raise ValueError(f'未知 AGC 模式 {self.agc_mode!r}, 只允许 {AGC_MODES}')
        if not np.isfinite(self.agc_tau_s) or self.agc_tau_s <= 0:
            raise ValueError('AGC 时间常数必须是有限正数')
        if not self.agc_gain_min_db < self.agc_gain_max_db:
            raise ValueError('AGC 限幅必须满足 min < max')
        if self.dfe_reset_per_chirp and self.analog_continuous:
            raise ValueError(
                '本模型只有一级有状态滤波器: analog_continuous=True 与 '
                'dfe_reset_per_chirp=True 无法同时表达。要么两者复位 (per_chirp_reset), '
                '要么两者连续 (continuous)。')
        if self.adc_gated_during_idle and self.idle_input != 'zeros':
            raise ValueError(
                f'ADC 门控 (adc_gated_during_idle=True) 时 idle 输入只能是 zeros, '
                f'不能是 {self.idle_input!r}; 要让 idle 噪声/干扰推进 DFE, '
                f'请显式声明 adc_gated_during_idle=False。')

    @property
    def is_legacy(self) -> bool:
        return (self.mode == 'per_chirp_reset' and self.agc_mode == 'per_chirp_rms')

    @property
    def idle_advances_state(self) -> bool:
        """idle 是否推进滤波器状态 (复位预设为 False)。"""
        return not self.dfe_reset_per_chirp

    def describe(self) -> dict:
        """给报告用的一行声明摘要。"""
        return {
            'mode': self.mode,
            'analog_continuous': self.analog_continuous,
            'lo_running_during_idle': self.lo_running_during_idle,
            'adc_gated_during_idle': self.adc_gated_during_idle,
            'dfe_reset_per_chirp': self.dfe_reset_per_chirp,
            'idle_input': self.idle_input,
            'idle_advances_state': self.idle_advances_state,
            'agc_mode': self.agc_mode,
            'agc_tau_s': self.agc_tau_s,
            'agc_gain_limits_db': [self.agc_gain_min_db, self.agc_gain_max_db],
        }


#: 两个命名预设。自定义组合用 ``dataclasses.replace`` 从这两个之一派生，
#: 因为 ``FrontendSpec`` 是 frozen 的。
FRONTEND_PRESETS = {
    'per_chirp_reset': FrontendSpec(),
    'continuous': FrontendSpec(
        mode='continuous', analog_continuous=True, dfe_reset_per_chirp=False,
        adc_gated_during_idle=True, idle_input='zeros'),
}


def spec_from_radar(rc) -> FrontendSpec:
    """从公开的 ``RadarConfig`` 字段构造前端声明。

    非法组合在 ``FrontendSpec.__post_init__`` 里当场拒绝, 不静默降级。
    """
    mode = getattr(rc, 'frontend_mode', 'per_chirp_reset')
    idle_input = getattr(rc, 'frontend_idle_input', 'zeros')
    agc_mode = getattr(rc, 'frontend_agc_mode', 'per_chirp_rms')
    tau = float(getattr(rc, 'frontend_agc_tau_s', 1e-3))
    limits = tuple(getattr(rc, 'frontend_agc_gain_limits_db', (-30.0, 30.0)))
    common = dict(agc_mode=agc_mode, agc_tau_s=tau,
                  agc_gain_min_db=float(limits[0]), agc_gain_max_db=float(limits[1]))
    if mode == 'per_chirp_reset':
        if idle_input != 'zeros':
            raise ValueError(
                f'per_chirp_reset 预设下 idle_input={idle_input!r} 没有意义: '
                f'复位策略不推进状态。要研究 idle 输入请用 continuous 预设。')
        return FrontendSpec(mode=mode, **common)
    if mode == 'continuous':
        return FrontendSpec(mode=mode, analog_continuous=True, dfe_reset_per_chirp=False,
                            adc_gated_during_idle=(idle_input == 'zeros'),
                            idle_input=idle_input, **common)
    raise ValueError(f'未知 frontend_mode {mode!r}, 只允许 {FRONTEND_MODES}')


class FrontendChain:
    """把声明变成状态的执行器: 逐 chirp 处理 + idle 演化, 状态跨调用保持。

    用法 (对拍/审计):
        chain = FrontendChain(sos, spec)
        y0 = chain.process_chirp(x0)
        chain.advance_idle(n_idle)          # zeros 声明; noise/interferer 必须传 raw
        y1 = chain.process_chirp(x1)

    状态记账不变量:
      * ``process_chirp`` 不自动复位; ``dfe_reset_per_chirp`` 时它只把本次输入前的
        状态清零 (等价于 ``sosfilt`` 的零状态启动), 处理完仍保存末状态 (对复位预设无影响)。
      * 波形切换 (chirp 长度改变) 不会被自动吸收: 未 ``reset()`` 就换长度当场报错。
        真实雷达换波形时滤波器设计也变了, 旧状态对新滤波器没有意义。
      * ``advance_idle`` 在复位预设下是 no-op (声明: idle 不需要演化); 在连续预设下
        必须给出 idle 输入 (raw) 或使用 zeros 声明 —— 绝不静默跳过。
    """

    def __init__(self, sos, spec: FrontendSpec):
        self.sos = np.asarray(sos, dtype=float)
        self.spec = spec
        self.n_sections = int(self.sos.shape[0])
        self.reset()

    # ------------------------------------------------------------ 状态
    def reset(self):
        self.zi = np.zeros((self.n_sections, 2), dtype=complex)
        self.last_block_len: int | None = None
        self.n_chirps = 0
        self.n_idle_samples = 0
        self.zero_idle_shortcut = False

    @property
    def state_norm(self) -> float:
        return float(np.linalg.norm(self.zi))

    def state_dict(self) -> dict:
        return {'state_norm': self.state_norm, 'n_chirps': self.n_chirps,
                'n_idle_samples': self.n_idle_samples,
                'zero_idle_shortcut': self.zero_idle_shortcut,
                'spec': self.spec.describe()}

    # ------------------------------------------------------------ 处理
    def process_chirp(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=complex)
        if self.spec.dfe_reset_per_chirp:
            self.zi[...] = 0
        elif self.last_block_len is not None and x.size != self.last_block_len:
            raise ValueError(
                f'波形切换: 上一块 {self.last_block_len} 点, 本块 {x.size} 点。'
                f'连续预设不会自动吸收滤波器设计变化; 请先 reset() 并显式记录这次切换。')
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        self.last_block_len = int(x.size)
        self.n_chirps += 1
        return y

    def advance_idle(self, n_samples: int, raw: np.ndarray | None = None) -> None:
        """idle 期间的状态演化。

        Args:
            n_samples: idle 采样点数 (>=0)。零输入演化按实际空闲长度推进。
            raw: idle 的原始输入。 ``idle_input='zeros'`` 时省略 (内部用零输入演化);
                 'noise'/'interferer' 必须由调用方给出 (噪声/干扰流在调用方的随机源里,
                 chain 不自己造随机数, 否则对拍两边会拿到不同实现)。
        """
        n = int(n_samples)
        if n < 0:
            raise ValueError('idle 采样点数不能为负')
        if self.spec.dfe_reset_per_chirp or n == 0:
            return
        if raw is None:
            if self.spec.idle_input != 'zeros':
                raise ValueError(
                    f'idle_input={self.spec.idle_input!r} 需要调用方提供 raw 采样; '
                    f'chain 不生成随机数。')
            self._evolve_zero_state(n)
        else:
            self.process_idle(n, raw)
            return
        self.n_idle_samples += n

    def process_idle(self, n_samples: int, raw: np.ndarray | None = None) -> np.ndarray:
        """推进 idle 段并返回该段输出 (对拍/证据用; 长 idle 的日常路径用 advance_idle)。

        与 ``advance_idle`` 的差别只有返回值: 这里会把 idle 段的滤波输出物化,
        因此不做零输入提前截断 —— 调用方对内存负责。
        """
        n = int(n_samples)
        if n < 0:
            raise ValueError('idle 采样点数不能为负')
        if self.spec.dfe_reset_per_chirp or n == 0:
            return np.zeros(n, dtype=complex)
        if raw is None:
            if self.spec.idle_input != 'zeros':
                raise ValueError(
                    f'idle_input={self.spec.idle_input!r} 需要调用方提供 raw 采样; '
                    f'chain 不生成随机数。')
            y, self.zi = sosfilt(self.sos, np.zeros(n, dtype=complex), zi=self.zi)
        else:
            if self.spec.idle_input == 'zeros':
                raise ValueError(
                    "声明是 idle_input='zeros' (ADC 门控/零输入), 却传入了 raw 采样; "
                    "要么改用 noise/interferer 声明, 要么不要给 raw。")
            raw = np.asarray(raw, dtype=complex)
            if raw.size != n:
                raise ValueError(f'idle raw 长度 {raw.size} != 声明的 {n}')
            y, self.zi = sosfilt(self.sos, raw, zi=self.zi)
        self.n_idle_samples += n
        return y

    def _evolve_zero_state(self, n: int, chunk: int = 256,
                           max_chunks: int = 8192) -> None:
        """零输入演化: zi[k+1] = A zi[k]。

        用 256 点分块递推, 状态范数降到 ``_STATE_ZERO_TOL`` 以下就提前停 (默认体制
        几十点内就到); 超过 ``max_chunks`` (2M 点) 还没衰减说明这个滤波器有极慢极点,
        这时宁可报错也不静默跳过 —— 方案 §四.1 明确禁止跳过 idle 状态演化。
        """
        z = np.zeros(chunk, dtype=complex)
        for _ in range(max_chunks):
            take = min(chunk, n)
            if take == 0:
                return
            _, self.zi = sosfilt(self.sos, z[:take], zi=self.zi)
            n -= take
            if self.state_norm < _STATE_ZERO_TOL:
                self.zero_idle_shortcut = n > 0
                return
        raise RuntimeError(
            f'零输入演化 {max_chunks} 块后状态范数仍有 {self.state_norm:.3e}; '
            f'这个滤波器不适合用"分块递推到衰减"的截断, 请改用整段演化。')


class TrackingAGC:
    """一阶 dB 域跟踪归一化器 (M1-A 的新 AGC 状态模型)。

    与旧口径的差别只有一个: 旧口径 ``scale = 1/rms`` 是**瞬时**的, 没有状态;
    这里把 scale 当作 dB 域一阶环节, 增益有上限/下限, 并可通过 ``scale_db`` 读出。

    顺序声明: 本类作用在 ADC 之后的数字归一化尺度 (``ChirpFrame.agc_scale`` 口径);
    PGA/削顶在 receiver 里保持原顺序。第一次调用使用 ``initial_scale_db`` (默认 0 dB,
    即不归一), 之后按 ``w = 1 - exp(-dt/tau)`` 朝目标推进。误差必须包含**当前增益**,
    否则环会一路顶到限幅 (实测踩过):

        out_rms  = measured_rms * scale            # measured_rms 是加增益前的
        err_db   = -20 log10(out_rms / target_rms)
        scale_db = clip(scale_db + w * err_db, [min, max])
    """

    def __init__(self, spec: FrontendSpec, initial_scale_db: float = 0.0,
                 target_rms: float = 1.0):
        if spec.agc_mode != 'tracking_db':
            raise ValueError(f'TrackingAGC 只服务 tracking_db 声明, 收到 {spec.agc_mode!r}')
        if not np.isfinite(initial_scale_db) or target_rms <= 0:
            raise ValueError('初始增益与目标 RMS 必须是有限值 / 正数')
        self.spec = spec
        self.target_rms = float(target_rms)
        self.scale_db = float(np.clip(initial_scale_db,
                                      spec.agc_gain_min_db, spec.agc_gain_max_db))

    def reset(self, initial_scale_db: float = 0.0):
        self.scale_db = float(np.clip(initial_scale_db,
                                      self.spec.agc_gain_min_db, self.spec.agc_gain_max_db))

    @property
    def scale(self) -> float:
        return float(10.0 ** (self.scale_db / 20.0))

    def update(self, measured_rms: float, dt_s: float) -> None:
        """用本块测得的 RMS 更新下一次的增益 (因果: 本块用的是更新前的增益)。"""
        if not np.isfinite(dt_s) or dt_s < 0:
            raise ValueError('AGC 更新间隔必须是有限非负数')
        if not np.isfinite(measured_rms) or measured_rms <= 0:
            return
        out_rms = measured_rms * self.scale
        err_db = 20.0 * np.log10(max(self.target_rms, 1e-300) / max(out_rms, 1e-300))
        w = 1.0 - np.exp(-dt_s / self.spec.agc_tau_s)
        self.scale_db = float(np.clip(self.scale_db + w * err_db,
                                      self.spec.agc_gain_min_db,
                                      self.spec.agc_gain_max_db))

    def state_dict(self) -> dict:
        return {'agc_mode': 'tracking_db', 'scale_db': self.scale_db,
                'agc_tau_s': self.spec.agc_tau_s,
                'agc_gain_limits_db': [self.spec.agc_gain_min_db,
                                       self.spec.agc_gain_max_db]}


# ---------------------------------------------------------------- 整段参考
def whole_segment_reference(blocks, sos, reset_each: bool = False):
    """整段流式参考: 把 blocks 拼起来一次 ``sosfilt`` (或逐块零状态) 后切回。

    与 ``FrontendChain`` 的逐块带状态递推应当逐位一致 (M0-C 已经验证过这个性质)。
    """
    sos = np.asarray(sos, dtype=float)
    parts = [np.asarray(b, dtype=complex) for b in blocks]
    if reset_each:
        return [sosfilt(sos, p) for p in parts]
    y = sosfilt(sos, np.concatenate(parts))
    out, i = [], 0
    for p in parts:
        out.append(y[i:i + p.size])
        i += p.size
    return out


def spec_variant(base: FrontendSpec, **kwargs) -> FrontendSpec:
    """``dataclasses.replace`` 的别名, 让自定义组合在报告里也走同一个校验。"""
    return replace(base, **kwargs)
