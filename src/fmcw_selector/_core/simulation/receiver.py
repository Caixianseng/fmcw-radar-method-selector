# simulation/receiver.py
"""接收信号层: 把"某时刻的车辆位姿 + 各雷达时钟"变成一个 chirp 的复基带差拍数据。

每个 chirp 的生成过程:
    1. 受害者发射相位 φ_v(t) 由其自身时钟给出 (t = 绝对接收时刻)
    2. 散射中心 q:  ψ_q(t) = φ_v(t) - φ_v(t - 2R_q(t)/c) + 镜面相位
       干扰雷达 j:  ψ_j(t) = φ_v(t) - φ_j(t - R_j(t)/c) + 镜面相位,
                    且只在"发射时刻 t-R/c 上它真的在扫"时才注入 (异步 + 空闲段)
    3. 幅度 = sqrt(接收功率): 目标功率 ∝ R^-4, 直接干扰功率 ∝ R^-2
    4. 叠加热噪声 -> SOS 低通 (chirp 内递推, 不是整帧循环 FFT) -> 满量程削顶 -> AGC

真值 (TruthRecord) 与观测量在同一 AGC 尺度上, 但只有 evaluation/ 和离线标定允许读它。

M1-A (2026-10-04): 第 4 步的滤波走法由 `RadarConfig.frontend_mode` 声明, 两种预设:
  * `per_chirp_reset` (默认, 旧口径): 每 chirp 零状态启动, AGC 每 chirp RMS 归一化。
  * `continuous`: 连续前端 (idle 期间状态不跳过, 由 `frontend_idle_input` 声明输入),
    DFE 不复位, 可配 tracking AGC。此时 `TruthRecord.frontend_residual/clip_residual`
    把"状态响应"和"削顶残差"单列, 保证公共尺度上恒等式
    observed = clean + interf + noise + frontend_residual + clip_residual 成立,
    不把前端瞬态或削顶误差平摊进线性的 clean/interf/noise 里。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from fmcw_selector._core.common.dsp import (apply_lpf, design_lpf, filter_settle_samples,
                        sliding_mean_power)
from fmcw_selector._core.common.types import ChirpFrame, InterfererTruth, TargetTruth, TruthRecord
from fmcw_selector._core.configs import SystemConfig
from fmcw_selector._core.configs.radar import C, RadarConfig
from fmcw_selector._core.simulation.frontend import FrontendChain, TrackingAGC, spec_from_radar
from fmcw_selector._core.simulation.occlusion import los_blockage
from fmcw_selector._core.simulation.propagation import LinkBudget, aspect
from fmcw_selector._core.simulation.radar_clock import ChirpTimeline
from fmcw_selector._core.simulation.traffic import VehicleState


def _ratio_db(num: float, den: float, floor: float = 1e-30) -> float:
    """Power ratio with explicit zero-component semantics (no fictitious floor power)."""
    if num < 0 or den < 0:
        raise ValueError('negative component power')
    if den == 0:
        return np.inf if num > 0 else np.nan
    if num == 0:
        return -np.inf
    return float(10.0 * np.log10(num / den))


class VictimReceiver:
    """受害者雷达接收机 (逐 chirp 合成, 会话内保持同一条噪声随机流)。

    Args:
        cfg: 系统配置 (radar / detection)
        victim: 受害者自己的 chirp 时钟
        rng: 会话级随机源, 噪声逐 chirp 连续取数
        occluder_size: 二元组 (长, 宽) m, >0 时启用二维几何遮挡 (车身矩形); None = 完全
            无遮挡。P2 要求的是**两档敏感性对比**, 所以默认关掉, 由调用方显式打开。
    """

    def __init__(self, cfg: SystemConfig, victim: ChirpTimeline, rng: np.random.Generator,
                 occluder_size=None):
        self.cfg = cfg
        self.rc: RadarConfig = cfg.radar
        self.victim = victim
        self.rng = rng
        self.occluder_size = tuple(occluder_size) if occluder_size else None
        self.sos = design_lpf(self.rc.sample_rate, self.rc.lpf_cutoff, self.rc.lpf_stop_edge,
                              self.rc.lpf_pass_ripple_db, self.rc.lpf_stop_atten_db)
        # 每 chirp 零状态启动 (apply_lpf 不跨 chirp 保留状态), 所以头若干个采样点不是稳态
        # 观测。这个长度从**同一份 SOS** 现场算出来并单独标成标签, 不手填、也不靠
        # "25 us 空闲远大于群时延"这句话默认它为 0 (那句话对不对由 dsp 里的判据说)。
        self.settle_samples = filter_settle_samples(
            self.sos, self.rc.frontend_settle_rel_tol)
        self.links = LinkBudget(self.rc)
        self.dt = np.arange(self.rc.n_samples, dtype=float) / self.rc.sample_rate
        self.full_scale = self.rc.adc_full_scale_v
        # M1-A 前端声明: 默认 `per_chirp_reset` 时两条都为空, 走旧路径, 输出逐位不变。
        self.frontend = spec_from_radar(self.rc)
        self._chain = (None if self.frontend.is_legacy
                       else FrontendChain(self.sos, self.frontend))
        self._agc = (TrackingAGC(self.frontend)
                     if self.frontend.agc_mode == 'tracking_db' else None)
        self._gap_prev = float(victim.period)   # 前一 chirp 起点间隔 (synthesize 内按帧结构更新)
        self._frontend_end = None      # 上一 chirp 采样窗的结束时刻 (连续状态的时间锚点)
        self._agc_last_t = None
        # 长 idle 的噪声推进不需要从头积: 滤波器状态的相关时间由稳定时间界定,
        # 先烧入这么多点后状态已与"从会话开始连续推进"同分布。只影响 noise 声明,
        # 且烧入长度与用掉的长度都记账 (方案 §四.1 "经验证的状态传播方法")。
        self._idle_burnin_samples = max(64 * self.settle_samples, 1)
        self._idle_burnin_used = 0

    # ------------------------------------------------------------ 入口
    def chirp_start(self, m: int) -> float:
        """第 m 个 chirp 的起始时刻 (s), 完全由受害者时钟推出。"""
        return float(self.victim.chirp_start(m))

    def reset_frontend(self):
        """丢弃前端连续状态与 AGC 历史; 下一帧从零状态/初始增益开始。

        真实换波形/换会话时必须显式调用: 旧状态对新滤波器没有意义, 不许自动吸收。
        """
        if self._chain is not None:
            self._chain.reset()
        if self._agc is not None:
            self._agc.reset()
        self._frontend_end = None
        self._agc_last_t = None
        self._idle_burnin_used = 0

    def frontend_state(self) -> dict:
        """只读的状态/声明摘要 (报告用), 不推进任何状态。"""
        out = dict(self.frontend.describe(), is_legacy=self.frontend.is_legacy)
        out['idle_burnin_samples'] = self._idle_burnin_samples
        out['idle_burnin_samples_used'] = self._idle_burnin_used
        if self._chain is not None:
            out['chain'] = self._chain.state_dict()
        if self._agc is not None:
            out['agc'] = self._agc.state_dict()
        return out

    def _advance_frontend_idle(self, t0: float):
        """把连续前端从上一 chirp 末尾推进到本 chirp 起点 (M1-A)。

        只支持 'zeros'/'noise' 两种 idle 输入; 'interferer' 需要 idle 期间的交通几何,
        参考链在 debugging/frontend_validation.py, receiver 不假装自己有那份输入。
        """
        if self._frontend_end is None:
            return
        gap_s = t0 - self._frontend_end
        if gap_s < -1e-9:
            raise ValueError(
                f'时间回退: 本 chirp 起点 {t0:.9g} 早于上一 chirp 末尾 '
                f'{self._frontend_end:.9g}; 连续前端状态不能倒放, 请显式 reset_frontend()。')
        n_idle = int(round(max(gap_s, 0.0) * self.rc.sample_rate))
        if n_idle <= 0:
            return
        if self.frontend.idle_input == 'zeros':
            self._chain.advance_idle(n_idle)
        elif self.frontend.idle_input == 'noise':
            # 会话窗口之间可能跳过秒级时间; 噪声驱动状态在烧入后已是平稳分布,
            # 用最后 burn-in 段推进即可 (记账见 _idle_burnin_used)。
            # 2026-10-11 v3: off 模式空闲期也不注入随机噪声, 按零输入推进。
            if self.rc.noise_power_w <= 0.0:
                self._chain.advance_idle(n_idle)
            else:
                n_used = min(n_idle, self._idle_burnin_samples)
                half = np.sqrt(self.rc.noise_power_w / 2.0)
                raw = (self.rng.normal(0.0, half, n_used)
                       + 1j * self.rng.normal(0.0, half, n_used))
                self._chain.advance_idle(n_used, raw)
                self._idle_burnin_used += n_idle - n_used
        else:
            raise NotImplementedError(
                f"receiver 集成只支持 idle_input in ('zeros','noise'), 收到 "
                f"{self.frontend.idle_input!r}; 参考链入口见 "
                f"debugging/frontend_validation.py。")

    def synthesize(self, t0: float, ego: VehicleState, states: list[VehicleState],
                   session_chirp: int) -> ChirpFrame:
        rc, n = self.rc, self.rc.n_samples
        ts = t0 + self.dt                        # 绝对接收时刻 (s)
        # 2026-10-10 修复: 受害者自身相位改用本地时间 (u = 采样在本 chirp 内的偏移, 小量,
        # 无大数相减), 避免绝对时间 ts (~秒级) 的 float64 舍入被 2π·fc 放大成 ~1e-4 rad。
        self._gap_prev = self._previous_chirp_gap(session_chirp)
        phi_v = self._victim_phase(self.dt)
        # 2026-10-11 v3 解耦: 分量门限使用**固定参考功率** (与噪声模式无关),
        # 保证 off 模式下目标/干扰集合与 scaled 完全一致; 实际注入功率单独取用。
        t_floor_w = rc.noise_power_ref_w * 10.0 ** (rc.target_floor_db / 10.0)
        i_floor_w = rc.noise_power_ref_w * 10.0 ** (rc.interferer_floor_db / 10.0)

        clean = np.zeros(n, dtype=complex)
        interf = np.zeros(n, dtype=complex)
        mask_inband = np.zeros(n, dtype=bool)
        targets: list[TargetTruth] = []
        interferers: list[InterfererTruth] = []
        # 二维遮挡判定每个 chirp 只做一次 (车辆位姿在 chirp 内不动): 被挡住的车**不注入**
        # 任何分量, 但仍登记真值 (occluded=True) —— 否则评价会把"几何上看不见"当成
        # "算法没查到", 而这两件事的处置方式完全不同。
        blocked = (los_blockage(ego.pos, states, states, *self.occluder_size)
                   if self.occluder_size else np.zeros(len(states), dtype=bool))

        for idx, st in enumerate(states):
            r0, closing, ang_obs, ang_tgt = aspect(ego, st)
            r_t = r0 - closing * self.dt                    # 接收时刻的瞬时距离 (m)
            if blocked[idx]:
                if st.rcs > 0.0 and r0 < rc.max_range_m * 1.2:
                    targets.append(TargetTruth(vehicle_id=st.vid, range_m=r0,
                                               radial_speed_mps=closing, rcs_m2=st.rcs,
                                               beat_hz=0.0, power_w=0.0, in_band=False,
                                               occluded=True))
                if st.radar is not None and rc.interference_range_allows(r0):
                    interferers.append(InterfererTruth(
                        radar_id=st.radar.radar_id, vehicle_id=st.vid, range_m=r0,
                        power_w=0.0, beat_slope_hz_s=0.0, beat_hz_center=0.0, duty=0.0,
                        inband_mask=np.zeros(n, dtype=bool), occluded=True))
                continue
            if st.rcs > 0.0 and r0 < rc.max_range_m * 1.2:
                comp = self._target_component(st, ts, r_t, phi_v, r0, closing, ang_obs,
                                              targets, t_floor_w)
                if comp is not None:
                    clean += comp
            if st.radar is not None and rc.interference_range_allows(r0):
                comp, mask = self._interferer_component(st, ts, r_t, phi_v, r0, closing,
                                                        ang_obs, ang_tgt, interferers,
                                                        i_floor_w)
                if comp is not None:
                    interf += comp
                    mask_inband |= mask

        # 2026-10-11 v3: 实际注入功率由 noise_mode 决定 (off=0, physical=NF 档, scaled=参考)。
        # off 模式不消耗噪声随机流, 也不产生任何随机样本。
        p_gen = rc.noise_power_w
        if p_gen > 0.0:
            half = np.sqrt(p_gen / 2.0)
            noise = self.rng.normal(0.0, half, n) + 1j * self.rng.normal(0.0, half, n)
        else:
            noise = np.zeros(n, dtype=complex)

        # ---- M1-A: 前端处理分两条声明清楚的路径 -------------------------------
        # 无论哪条路径, `clean_f/interf_f/noise_f` 都是**零状态**分量响应, 即离线标签
        # 一直以来的口径; 连续路径多出来的状态响应与削顶残差单列, 不平摊进分量。
        if self.frontend.is_legacy:
            clean_f = apply_lpf(clean, self.sos)
            interf_f = apply_lpf(interf, self.sos)
            noise_f = apply_lpf(noise, self.sos)
            proc = clean_f + interf_f + noise_f
            obs_phys, clipped_frac = self._clip(proc)
            frontend_residual = None
            clip_residual = None
        else:
            clean_f = apply_lpf(clean, self.sos)
            interf_f = apply_lpf(interf, self.sos)
            noise_f = apply_lpf(noise, self.sos)
            if self._chain is not None:
                self._advance_frontend_idle(t0)
                proc = self._chain.process_chirp(clean + interf + noise)
                # 线性分解 T(x,s) = T(x,0) + ZIR(s): 状态响应单独成一路。
                frontend_residual = proc - (clean_f + interf_f + noise_f)
            else:
                proc = clean_f + interf_f + noise_f
                frontend_residual = None
            obs_phys, clipped_frac = self._clip(proc)
            clip_residual = obs_phys - proc
            self._frontend_end = t0 + rc.chirp_duration

        p_clean = float(np.mean(np.abs(clean_f) ** 2))
        p_interf = float(np.mean(np.abs(interf_f) ** 2))
        p_noise = float(np.mean(np.abs(noise_f) ** 2))
        rms = float(np.sqrt(np.mean(np.abs(obs_phys) ** 2)))
        if self._agc is not None:
            # 跟踪 AGC: 本 chirp 用更新前的增益 (因果), 再用本 chirp 实测 RMS 更新下一次。
            scale = self._agc.scale
            dt_s = (t0 - self._agc_last_t) if self._agc_last_t is not None else rc.chirp_period
            self._agc.update(rms, max(float(dt_s), 0.0))
            self._agc_last_t = t0
        else:
            scale = 1.0 / rms if rms > 0.0 else 1.0         # AGC: 观测归一化到单位 RMS

        margin_db = self.cfg.detection.label_margin_db
        local = sliding_mean_power(interf_f, self.cfg.detection.window_len)
        # 同一个 AGC 口径、同一个滑窗判据, 但**不**与 RF 通带窗求交: 抗混叠过渡带
        # (lpf_cutoff~lpf_stop_edge) 与数字滤波器展宽都会在 mask_inband 之外留下高于
        # 底噪的 IF 干扰, 求交会把它们整段抹掉。两条分开存, 差异由
        # debugging/label_support_audit.py 量出来 (M0-C)。
        # 2026-10-11 v3 标签解耦: 有噪声时判据与历史完全一致 (实测噪声 × 10^margin);
        # off 模式 (实际注入=0) 改用**精确的"窗口内存在非零能量"支撑判据**:
        # 布尔前缀和判定, 避免功率前缀和差分在小功率下的相消误差 (sliding_mean_power)。
        if p_noise > 0.0:
            if_support = local >= p_noise * 10.0 ** (margin_db / 10.0)
        else:
            nz = (np.abs(interf_f) > 0.0).astype(np.int64)
            cc = np.concatenate(([0], np.cumsum(nz)))
            ii = np.arange(nz.size)
            lo_i = np.maximum(0, ii - self.cfg.detection.window_len + 1)
            if_support = (cc[ii + 1] - cc[lo_i]) > 0
        # 下面这一行的定义与 labels-v1 完全一致, **故意不动**: 现有一切 IoU / 误删率
        # 报告仍然可比。要不要把拖尾算进训练标签是 M2 的决定, 而且必须同时用同一批数组
        # 重算新旧算法, 不能只替新算法改分母。
        mask_sig = if_support & mask_inband
        startup = np.zeros(n, dtype=bool)
        if self.settle_samples:
            startup[:min(self.settle_samples, n)] = True

        has_noise = p_noise > 0.0
        truth = TruthRecord(
            clean=clean_f * scale, interf=interf_f * scale, noise=noise_f * scale,
            mask_inband=mask_inband, mask_significant=mask_sig,
            noise_power_w=p_noise * scale ** 2,   # 与 clean/interf/noise 同一 AGC 口径
            # 2026-10-11 v3: 噪声模式与参考功率随 chirp 落盘; off 模式下 SNR/INR 无定义,
            # 显式标 None (主要评价改用重建误差/检出/损伤/虚警)。
            noise_mode=rc.noise_mode,
            noise_power_ref_w=rc.noise_power_ref_w * scale ** 2,
            snr_db=_ratio_db(p_clean, p_noise) if has_noise else None,
            inr_db=_ratio_db(p_interf, p_noise) if has_noise else None,
            sinr_db_in=_ratio_db(p_clean, p_interf + p_noise) if has_noise else None,
                # 2026-10-09 交叉审查修复: 目标功率存**同一 AGC 口径** (与 clean/noise/
                # noise_power_w 一致)。旧实现存前 AGC 物理 W, 与后 AGC 的 noise_power_w
                # 差 scale^2 (~122 dB), 导致所有 t.power_w/noise_power_w 的目标 SNR 诊断错误。
                targets=[replace(t, power_w=t.power_w * scale ** 2, power_physical_w=t.power_w, power_agc_w=t.power_w * scale ** 2) for t in targets],
            interferers=[replace(i, power_physical_w=i.power_w, power_agc_w=i.power_w * scale ** 2) for i in interferers], n_in_scene=len(states),
            unrecoverable=bool(mask_sig.mean() > 0.97 or clipped_frac > 0.02),
            mask_if_support=if_support, mask_startup_invalid=startup,
            frontend_residual=(None if frontend_residual is None
                               else frontend_residual * scale),
            clip_residual=(None if clip_residual is None
                           else clip_residual * scale))
        return ChirpFrame(session_chirp=session_chirp, t0=t0, observed=obs_phys * scale,
                          agc_scale=scale, truth=truth, adc_clipped_frac=clipped_frac,
                          frame_index=session_chirp // rc.frame_chirps,
                          chirp_in_frame=session_chirp % rc.frame_chirps)

    def _victim_phase(self, u: np.ndarray) -> np.ndarray:
        """受害者扫频相位 2π(f0·u + K·u²/2), u 为相对本 chirp 起点的本地时间 (小量)。"""
        v = self.victim
        return 2.0 * np.pi * (v.f0 * u + 0.5 * v.slope * u ** 2)

    def _previous_chirp_gap(self, session_chirp: int) -> float:
        """第 session_chirp 个 chirp 起点与前一 chirp 起点的间隔 (由时钟参数解析给出)。"""
        v = self.victim
        if v.frame_period is None:
            return float(v.period)
        slot = int(session_chirp) % int(v.frame_chirps)
        if slot > 0:
            return float(v.period)
        return float(v.frame_repeat - (v.frame_chirps - 1) * v.period)

    def _anti_alias(self, f_beat: np.ndarray) -> np.ndarray:
        """模拟抗混叠前端的线性过渡带增益 (0~1)。

        必须先带限再采样: 差频 |f| > fs/2 = 25 MHz 的分量会被 50 MHz 采样折叠进通带,
        事后任何数字滤波器都分不开。真实接收机里这一步是混频后的模拟 LPF, 通带边缘
        lpf_cutoff、阻带起始 lpf_stop_edge; 这里用两者之间的线性过渡近似其幅频,
        之后再由 SOS 数字滤波器完成通带整形 (不再承担抗混叠职责)。
        """
        rc = self.rc
        return np.clip((rc.lpf_stop_edge - np.abs(f_beat)) /
                       (rc.lpf_stop_edge - rc.lpf_cutoff), 0.0, 1.0)

    # ------------------------------------------------------------ 分量合成
    def _target_component(self, st: VehicleState, ts, r_t, phi_v, r0, closing,
                          ang_obs, out: list, floor_w: float):
        """一个散射中心的差拍回波 (复数), 同时登记真值; 低于门限返回 None。"""
        rc = self.rc
        a = self.links.target_amplitude(r0, st.rcs, ang_obs)
        if a ** 2 < floor_w:
            return None
        tau = 2.0 * r_t / C                                  # 双程时延 (s)
        active = self.victim.transmitting(ts - tau)
        # 受害者发射相位差 φ_v(u) - φ_v(u - τ), u 为本 chirp 内偏移; u-τ<0 时取前一 chirp 的
        # 本地时间 (+ 前一起点间隔)。全部为小量运算, 与绝对时间无关。
        # 整 chirp 合成时 ts 即 t0+dt, 直接用精确的采样偏移; 其它调用方 (单测) 退回时钟函数。
        u_v = self.dt if ts.size == self.dt.size else self.victim.local_time(ts)
        u_e = u_v - tau
        u_e = np.where(u_e < 0.0, u_e + self._gap_prev, u_e)
        psi = phi_v - self._victim_phase(u_e)
        # d phi(t - tau(t))/dt = 2pi f(t-tau) * (1 - tau_dot).
        # Analytic frequency avoids differencing across phase resets or silent slots.
        f_beat = (self.victim.instantaneous_freq(ts)
                  - self.victim.instantaneous_freq(ts-tau)*(1.+2.*closing/C))
        comp = a * active * self._anti_alias(f_beat) * np.exp(1j * (psi + st.specular_phase))
        in_band = active & (np.abs(f_beat) <= rc.lpf_cutoff)
        sel = f_beat[in_band] if in_band.any() else f_beat
        out.append(TargetTruth(vehicle_id=st.vid, range_m=r0, radial_speed_mps=closing,
                               rcs_m2=st.rcs, beat_hz=float(np.median(sel)),
                               power_w=a ** 2, in_band=bool(np.any(in_band))))
        return comp

    def _interferer_component(self, st: VehicleState, ts, r_t, phi_v, r0, closing,
                              ang_obs, ang_tgt, out: list, floor_w: float):
        """一台干扰雷达的差拍分量 + 其几何/通带有效窗 (真值掩码)。"""
        rc = self.rc
        tl = st.radar
        a = self.links.interference_amplitude(r0, tl.tx_power_w, ang_obs, ang_tgt)
        if a ** 2 < floor_w:
            return None, np.zeros(r_t.size, dtype=bool)
        tau = r_t / C                                        # 单程时延 (s)
        t_emit = ts - tau                                    # 对应的发射时刻 (绝对)
        active = tl.transmitting(t_emit)
        psi = phi_v - tl.tx_phase(t_emit)
        f_beat = (self.victim.instantaneous_freq(ts)
                  - tl.instantaneous_freq(t_emit)*(1.+closing/C))
        aa = self._anti_alias(f_beat)
        mask = active & (np.abs(f_beat) <= rc.lpf_cutoff)
        comp = a * active * aa * np.exp(1j * (psi + st.specular_phase))
        idx = np.flatnonzero(mask)
        slope = 0.0
        if idx.size > 2:
            slope = float((f_beat[idx[-1]] - f_beat[idx[0]]) /
                          ((idx[-1] - idx[0]) / rc.sample_rate))
        out.append(InterfererTruth(radar_id=tl.radar_id, vehicle_id=st.vid, range_m=r0,
                                   power_w=a ** 2, beat_slope_hz_s=slope,
                                   beat_hz_center=float(np.median(f_beat[mask])) if idx.size
                                   else 0.0, duty=float(mask.mean()), inband_mask=mask))
        return comp, mask

    def _clip(self, x: np.ndarray):
        """ADC/PGA 满量程: 按模硬削顶, 保留相位。返回 (削顶后信号, 削顶样本占比)。"""
        mag = np.abs(x)
        over = mag > self.full_scale
        if not over.any():
            return x, 0.0
        out = x.copy()
        out[over] *= self.full_scale / mag[over]
        return out, float(over.mean())
