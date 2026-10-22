# configs/radar.py
"""受害者雷达与接收机的公开配置。

这里每一项都是公开配置量：调频参数、链路预算假设、量程设定。
默认值尚未对应一台经过标定的实机，不能把量程配置当成探测能力证明。
定位/重构算法只允许读这一层（以及 algorithms/budget），不允许读场景真值。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C = 299792458.0          # 光速 (m/s)
K_B = 1.380649e-23       # 玻尔兹曼常数 (J/K)


def db(x):
    return 10.0 ** (x / 10.0)


def db2lin(x):
    return 10.0 ** (x / 10.0)


@dataclass
class RadarConfig:
    # ---------- 调频体制 (对应旧项目 config.py 的论文 Table 1) ----------
    fc: float = 77e9                 # 载频 (Hz)
    bandwidth: float = 300e6         # 扫频带宽 (Hz)
    chirp_duration: float = 75e-6    # 单个 chirp 的采样时长 (s)
    chirp_period: float = 100e-6     # chirp 重复周期 = 采样 75us + 空闲 25us (s)
    sample_rate: float = 50e6        # ADC 采样率 (Hz)
    frame_chirps: int = 64           # 一个雷达帧内的 chirp 数 (慢时间/多普勒维)

    frame_period: float | None = None  # explicit frame repeats; None preserves legacy continuous train

    # ---------- 接收机前端 ----------
    lpf_cutoff: float = 20e6         # 模拟/数字低通通带截止 (Hz)
    lpf_stop_edge: float = 23.5e6    # 阻带起始频率 (Hz), 决定过渡带陡峭度
    lpf_pass_ripple_db: float = 0.5
    lpf_stop_atten_db: float = 40.0
    # 每 chirp 单独零状态启动的数字滤波器要多久才进入稳态。判据: 阶跃响应相对稳态值的
    # 偏差首次降到本容差以内并保持在以内 (由 simulation/receiver.py 现场从同一份 SOS 算,
    # 不手填点数)。这一段被单独标成"启动无效", 不混进污染标签 —— 它是前端状态问题,
    # 换个前端配置就会变, 当成"干扰"或当成"干净"都是错的 (M0-C)。
    frontend_settle_rel_tol: float = 0.01
    # ---- 接收前端连续状态声明 (M1-A, simulation/frontend.py) ----
    # 默认 'per_chirp_reset' = 已发表数字的旧口径: 模拟前端按无记忆处理, DFE 每 chirp
    # 零状态启动, AGC 每 chirp RMS 归一化。'continuous' = 连续模拟前端 + DFE 不复位,
    # idle 期间用 idle_input 声明继续推进滤波器状态 (不跳过)。两种预设的硬件含义
    # (idle/LO/ADC/DFE) 由 simulation/frontend.py 的 FrontendSpec 写清楚。
    frontend_mode: str = 'per_chirp_reset'          # 'per_chirp_reset' | 'continuous'
    frontend_idle_input: str = 'zeros'              # 'zeros' | 'noise' | 'interferer'
    # AGC: 'per_chirp_rms' 是旧口径 (无状态瞬时归一化); 'tracking_db' 是一阶跟踪环,
    # 时间常数/限幅/可读取增益都公开。默认保持旧口径, 新模型只在显式声明时启用。
    frontend_agc_mode: str = 'per_chirp_rms'        # 'per_chirp_rms' | 'tracking_db'
    frontend_agc_tau_s: float = 1e-3
    frontend_agc_gain_limits_db: tuple = (-30.0, 30.0)
    noise_figure_db: float = 10.0    # 等效噪声生成用；与额外损耗的参考面须对照实机核实
    #: 声明式噪声缩放 (dB, 叠加在 NF 之上)。默认 0 = 原配置不变。
    #: 低噪声/高信噪比场景由配置显式声明 (例如 P0 低噪声工况), 不是理想化的物理 NF。
    noise_scale_db: float = 0.0
    #: 噪声模式 (2026-10-11 v3):
    #:   'scaled'   = 声明式缩放: kTB*NF*fs*10^(scale/10) (默认, 原行为不动);
    #:   'physical' = 接收机参数生成热噪声 (仅 NF): kTB*NF*fs, 忽略 noise_scale_db;
    #:   'off'      = 不注入随机噪声 (chirp 内与空闲期); 分量门限与干扰标签仍用
    #:                固定参考功率 noise_power_ref_w, 目标/干扰集合不变。
    noise_mode: str = 'scaled'
    temperature_k: float = 290.0
    mixer_loss_db: float = 6.0       # 独立信号/干扰功率损耗；不再次衰减已生成的等效噪声
    adc_full_scale_v: float = 3.0e-4   # PGA 输出满量程 (sqrt(W) 口径); 近距强干扰削顶用
    agc_lock_power_w: float = 1e-9   # AGC 参考电平: 帧功率归一化到这个量

    # ---------- 发射与天线 (链路预算用) ----------
    tx_power_w: float = 0.01585      # 12 dBm
    # 车载 77 GHz 前向雷达的折中方向图: 15 dBi 主瓣 + 30 度波束 (中程 ARS/MRR 之间)。
    # 8 度窄波束会让邻车道/路口车辆全部落到旁瓣, 回波被可忽略分量门限滤掉;
    # 60 度宽波束配 13 dBi 又把干噪比压得过低, 复现不出"干扰强于目标几十 dB"的实测现象。
    antenna_peak_gain: float = 31.62  # 15 dBi 单向主瓣增益 (线性)
    beamwidth_rad: float = np.deg2rad(30.0)
    sidelobe_db: float = -15.0       # 波束外触底电平 (相对主瓣 dB)

    # ---------- 公开量程设定 ----------
    max_range_m: float = 300.0       # 应用距离范围；实际可检测性由完整帧Pd/虚警验证
    max_speed_mps: float = 60.0      # 应用标称最大相对速度
    # 默认 1000 m 为用户指定的仿真场景距离上限，不代表接收灵敏度。
    # None = 不设干扰距离上限；正有限值支持明确声明的其他实验范围。
    interference_max_range_m: float | None = 1000.0
    # 可忽略分量门限 (相对单采样点噪声功率, dB)。两者不同是因为"能不能被看见"的增益不同:
    # 目标要经过距离 FFT (~33 dB) + 帧内多 chirp 相参积累 (~18 dB) 才判决, 所以比噪声低
    # 40 多 dB 的回波仍是可检测目标, 必须保留; 干扰是带内连续波, 低 25 dB 已不影响任何指标。
    target_floor_db: float = -45.0
    interferer_floor_db: float = -25.0
    min_range_m: float = 5.0         # 路径损耗的距离下限 (近场/车长极限, 防 R^-4 发散)

    # 每个 chirp 的采样点数与时间轴 (属性, 不写死)
    @property
    def slope(self) -> float:
        """调频斜率 K = B/T (Hz/s)。"""
        return self.bandwidth / self.chirp_duration

    @property
    def n_samples(self) -> int:
        return int(round(self.sample_rate * self.chirp_duration))

    @property
    def wavelength(self) -> float:
        return C / self.fc

    @property
    def idle_duration(self) -> float:
        return self.chirp_period - self.chirp_duration

    @property
    def range_resolution(self) -> float:
        return C / (2.0 * self.bandwidth)

    @property
    def max_unambiguous_range(self) -> float:
        """受 LPF 通带限制的单 chirp 最大不模糊距离 (m)。"""
        return self.lpf_cutoff * C / (2.0 * self.slope)

    @property
    def operating_max_beat(self) -> float:
        """算法允许假设的工作频带上限 (Hz)。

        只由公开配置推出: 标称量程的拍频 + 标称速度的多普勒偏移 + 10% 余量。
        重构器用它限制频率搜索范围, 不读取当前场景真值。
        """
        f_range = 2.0 * self.slope * self.max_range_m / C
        f_doppler = 2.0 * self.max_speed_mps * self.fc / C
        return 1.1 * (f_range + f_doppler)

    @property
    def noise_power_ref_w(self) -> float:
        """固定的分量筛选参考功率 (W) = k * T_sys * fs * 10^(scale/10)。

        用于分量可忽略门限 (目标/干扰) 与干扰标签的比较基准, **与噪声模式无关**;
        off 模式下目标/干扰集合因此保持不变。标签版本固定为 'noise-ref-scaled-v1'。
        """
        t_sys = self.temperature_k * db2lin(self.noise_figure_db)
        return K_B * t_sys * self.sample_rate * db2lin(self.noise_scale_db)

    @property
    def noise_power_w(self) -> float:
        """实际注入噪声功率 (W): off=0, physical=NF 档, scaled=参考值。"""
        if self.noise_mode == 'off':
            return 0.0
        if self.noise_mode == 'physical':
            t_sys = self.temperature_k * db2lin(self.noise_figure_db)
            return K_B * t_sys * self.sample_rate
        if self.noise_mode == 'scaled':
            return self.noise_power_ref_w
        raise ValueError(f"未知 noise_mode {self.noise_mode!r}; 可选 off/scaled/physical")

    def antenna_gain(self, theta_rad: float) -> float:
        """单向天线功率增益 (线性)。

        主瓣用归一化 sinc^2 近似, 系数 0.8864 使 theta = beamwidth/2 处正好 -3 dB;
        触底到旁瓣电平 (默认 sidelobe_db=-15 配 15 dBi 主瓣 => 0 dBi 旁瓣底)。
        对 8° 波束, 侧向 30° 的目标/干扰机只剩旁瓣, 这是近邻车道与后方车辆
        不构成强干扰的主要来源。
        """
        u = np.asarray(theta_rad, dtype=float) / self.beamwidth_rad * 0.8864
        main = self.antenna_peak_gain * np.sinc(u) ** 2
        floor = self.antenna_peak_gain * db2lin(self.sidelobe_db)
        return np.maximum(main, floor)

    def echo_power_w(self, range_m, target_rcs_m2, gain_tx, gain_rx):
        """单站目标回波接收功率 (W): 雷达方程, 功率 ∝ R^-4。

        返回的是功率; 复基带采样幅度取 sqrt(该功率)。
        """
        lam = self.wavelength
        num = self.tx_power_w * gain_tx * gain_rx * lam ** 2 * target_rcs_m2
        r = np.maximum(np.asarray(range_m, float), self.min_range_m)
        den = (4.0 * np.pi) ** 3 * r ** 4
        return num / den * db(-self.mixer_loss_db)

    def interference_power_w(self, range_m, gain_tx, gain_rx, tx_power_w=None):
        """直接干扰接收功率 (W): 单向自由空间, 功率 ∝ R^-2。"""
        lam = self.wavelength
        pt = self.tx_power_w if tx_power_w is None else tx_power_w
        num = pt * gain_tx * gain_rx * lam ** 2
        r = np.maximum(np.asarray(range_m, float), self.min_range_m)
        den = (4.0 * np.pi) ** 2 * r ** 2
        return num / den * db(-self.mixer_loss_db)

    def interference_range_allows(self, range_m: float) -> bool:
        """干扰场景的可选距离门限；None 不按距离排除干扰源。"""
        limit = self.interference_max_range_m
        if limit is None:
            return True
        if isinstance(limit, (bool, np.bool_)) or not np.isfinite(limit) or limit <= 0:
            raise ValueError('interference_max_range_m 必须为 None 或正有限距离 (m)')
        return bool(range_m <= limit)

    def beat_frequency(self, range_m, radial_speed=0.0):
        """给定距离与径向速度的标称拍频 (Hz): f_b = K*tau - f_d。"""
        return self.slope * 2.0 * range_m / C - 2.0 * radial_speed * self.fc / C
