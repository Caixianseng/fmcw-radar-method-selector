"""高采样率复包络→Blackman→传播/RCS→叠加→IQ 混频→FIR→ADC。"""
import numpy as np
from scipy.signal import fftconvolve, firwin, kaiserord
from fmcw_selector._core.common.types import ChirpFrame, TargetTruth, InterfererTruth, TruthRecord
from fmcw_selector._core.configs.radar import C, K_B
from fmcw_selector._core.simulation.propagation import LinkBudget, aspect

PIPELINE_VERSION = 'complex_envelope_phase_clock_fir_adc_v4_range_policy'
NOISE_VERSION = 'high_rate_complex_thermal_per_chirp_seed_v1'


def baseband_chirp(time, bandwidth, duration, sample_rate, sign=1.):
    """以载频为中心的复包络；相位传播用复指数，对称 Blackman 加窗。"""
    time = np.asarray(time)
    last = (int(round(duration * sample_rate)) - 1) / sample_rate
    active = (time >= 0.) & (time <= last)
    w = np.zeros_like(time, dtype=complex)
    u = time[active]
    taper = .42 - .5*np.cos(2*np.pi*u/last) + .08*np.cos(4*np.pi*u/last)
    slope = sign * bandwidth / duration
    start = -sign * bandwidth/2
    w[active] = taper * np.exp(2j*np.pi*(start*u + .5*slope*u*u))
    return w


def ratio_db(num, den):
    if den == 0 or num == 0:
        return None
    return float(10*np.log10(num/den))


class InspectReceiver:
    def __init__(self, cfg, victim, rng, occluder_size=None, generation=None):
        if occluder_size:
            raise ValueError('当前生成链使用无遮挡四车道场景')
        self.cfg, self.rc, self.victim, self.rng = cfg, cfg.radar, victim, rng
        options = generation or {}
        self.noise_seed = options.get('noise_seed')
        self.sim_fs = float(options.get('simulation_sample_rate_hz', 2e9))
        # 链路功率、方向图、噪声系数均沿用 RadarConfig，不另加增益或指定 SNR。
        self.rx_gain = 1.
        q = self.sim_fs / self.rc.sample_rate
        if not np.isfinite(q) or q < 1 or abs(q-round(q)) > 1e-9:
            raise ValueError('仿真采样率必须是 ADC 采样率的整数倍')
        self.decimation = int(round(q))
        self.u = np.arange(self.rc.n_samples*self.decimation)/self.sim_fs
        self.lo = baseband_chirp(self.u, self.rc.bandwidth, self.rc.chirp_duration, self.sim_fs)
        width = (self.rc.lpf_stop_edge-self.rc.lpf_cutoff)/(self.sim_fs/2)
        taps, beta = kaiserord(self.rc.lpf_stop_atten_db, width)
        taps = max(3, int(taps) | 1)
        self.fir = firwin(taps, (self.rc.lpf_cutoff+self.rc.lpf_stop_edge)/2,
                          window=('kaiser', beta), fs=self.sim_fs)
        self.links = LinkBudget(self.rc)
        self.components, self.stages = {}, {}

    def chirp_start(self, m):
        return float(self.victim.chirp_start(m))

    def filter_adc(self, mixed):
        """离线 FIR 补偿群延迟后降采样；保留 [0,T) 的 ADC 采样点。"""
        mixed = np.asarray(mixed)
        if mixed.ndim == 1:
            return fftconvolve(mixed, self.fir, mode='same')[::self.decimation]
        if mixed.shape[0] == 0:
            return np.empty((0, self.rc.n_samples), dtype=complex)
        return fftconvolve(mixed, self.fir[None, :], mode='same', axes=-1)[:, ::self.decimation]

    def receiver_noise_rf(self, session_chirp):
        """Independent receiver noise, before mixing/filtering; no target-based scaling.

        An explicit session seed makes a chirp reproducible regardless of which
        acquisition windows are requested. Traffic/property clocks use other streams.
        E|n|^2 = k*T*fs_high*F*10^(scale/10), with half the variance per I/Q.
        """
        rc = self.rc
        if rc.noise_mode == 'off':
            return np.zeros(self.u.size, complex)
        if rc.noise_mode not in ('scaled', 'physical'):
            raise ValueError('noise_mode must be off, scaled or physical')
        power = K_B*rc.temperature_k*self.sim_fs*10**(rc.noise_figure_db/10)
        if rc.noise_mode == 'scaled':
            power *= 10**(rc.noise_scale_db/10)
        rng = self.rng if self.noise_seed is None else np.random.default_rng(
            np.random.SeedSequence([int(self.noise_seed), int(session_chirp), 0x4e4f4953]))
        half = np.sqrt(power/2)
        return rng.normal(0., half, self.u.size)+1j*rng.normal(0., half, self.u.size)

    def synthesize(self, t0, ego, states, session_chirp):
        rc = self.rc
        ts = t0 + self.u
        target_rf, interference_rf, targets, interferers = {}, {}, [], []
        high_inband = np.zeros(self.u.size, dtype=bool)
        beat_traces, active_traces, clock_traces = {}, {}, {}
        # 同参考代码的停跳几何：每条 chirp 固定位姿，单/双程多普勒另行调制。
        for st in states:
            r, closing, ang_obs, ang_tgt = aspect(ego, st)
            if st.rcs > 0 and r < rc.max_range_m*1.2:
                a = self.links.target_amplitude(r, st.rcs, ang_obs)
                tau = 2*r/C
                fd = 2*closing/rc.wavelength
                delayed = baseband_chirp(self.u-tau, rc.bandwidth, rc.chirp_duration, self.sim_fs)
                carrier = np.exp(1j*(np.remainder(-2*np.pi*rc.fc*tau+st.specular_phase, 2*np.pi) + 2*np.pi*fd*self.u))
                target_rf[st.vid] = a*delayed*carrier
                beat = rc.slope*tau-fd
                targets.append(TargetTruth(st.vid, r, closing, st.rcs, beat, a*a,
                                           abs(beat) <= rc.lpf_cutoff))
            if st.radar is not None and rc.interference_range_allows(r):
                tl = st.radar
                a = self.links.interference_amplitude(r, tl.tx_power_w, ang_obs, ang_tgt)
                tau = r/C
                fd = closing/rc.wavelength
                emit_time = ts-tau
                emit_u = tl.local_time(emit_time)
                active = tl.transmitting(emit_time)
                delayed = baseband_chirp(emit_u, tl.bandwidth, tl.sweep_duration,
                                     self.sim_fs, tl.slope_sign)
                delta_fc = tl.fc-rc.fc
                phase0 = np.remainder(-2*np.pi*tl.fc*tau+st.specular_phase+2*np.pi*delta_fc*t0, 2*np.pi)
                carrier = np.exp(1j*(phase0 + 2*np.pi*(fd+delta_fc)*self.u))
                interference_rf[st.vid] = a*delayed*carrier*active
                f_v = -rc.bandwidth/2+rc.slope*self.u
                f_j = -tl.slope_sign*tl.bandwidth/2+tl.slope*emit_u
                beat = f_v-f_j-delta_fc-fd
                beat_traces[st.vid], active_traces[st.vid] = beat, active
                first_index = int(tl.chirp_index(t0-tau))
                clock_traces[st.vid] = (float(tl.chirp_start(first_index)), float(tl.period), float(tau))
                mask = (active & (np.abs(beat) <= rc.lpf_cutoff))[::self.decimation]
                high_inband |= active & (np.abs(beat) <= rc.lpf_cutoff)
                interferers.append(InterfererTruth(tl.radar_id, st.vid, r, a*a,
                    rc.slope-tl.slope, float(np.median(beat[::self.decimation][mask])) if mask.any() else 0.,
                    float(mask.mean()), mask))
        target_sum = sum(target_rf.values(), np.zeros(self.u.size, dtype=complex))
        interf_sum = sum(interference_rf.values(), np.zeros(self.u.size, dtype=complex))
        # 沿用原有热噪声参数，在高采样率下产生，再由 LPF/ADC 限制带宽。
        awgn, thermal = np.zeros(self.u.size, dtype=complex), np.zeros(self.u.size, dtype=complex)
        thermal = self.receiver_noise_rf(session_chirp)
        self.components, source_mixed = {}, {}
        for kind, rf in [('target', target_rf), ('interferer', interference_rf)]:
            ids = sorted(rf)
            # 与 MATLAB dechirp 一致：xref * conj(received)，传播相位不转成幅度。
            mixed = np.array([self.lo*np.conj(rf[vid]*self.rx_gain) for vid in ids], dtype=complex).reshape(-1, self.u.size)
            source_mixed[kind] = mixed
            self.components[kind] = dict(ids=ids, filtered=self.filter_adc(mixed))
        clean = self.components['target']['filtered'].sum(axis=0)
        interf = self.components['interferer']['filtered'].sum(axis=0)
        noise_mixed = self.lo*np.conj((awgn+thermal)*self.rx_gain)
        noise = self.filter_adc(noise_mixed)
        # 噪声只生成一次，各分量共享这份噪声实现。
        observed = clean+interf+noise
        self.residual = np.zeros_like(observed)
        self.stages = dict(time_s=self.u, sample_rate_hz=self.sim_fs,
            tx_reference_complex=self.lo, target_received=target_sum, interferer_received=interf_sum,
            awgn_received=awgn, receiver_thermal_noise=thermal,
            receiver_input=target_sum+interf_sum+awgn+thermal,
            receiver_output=(target_sum+interf_sum+awgn+thermal)*self.rx_gain,
            mixed_before_lpf=self.lo*np.conj((target_sum+interf_sum+awgn+thermal)*self.rx_gain),
            target_vehicle_ids=np.array(self.components['target']['ids'], dtype=int),
            interferer_vehicle_ids=np.array(self.components['interferer']['ids'], dtype=int),
            target_mixed=source_mixed['target'], interferer_mixed=source_mixed['interferer'],
            interferer_beat_hz=np.array([beat_traces[v] for v in self.components['interferer']['ids']]).reshape(-1, self.u.size),
            interferer_rx_active=np.array([active_traces[v] for v in self.components['interferer']['ids']], dtype=bool).reshape(-1, self.u.size),
            interferer_clock_s=np.array([clock_traces[v] for v in self.components['interferer']['ids']]).reshape(-1, 3))
        mask_inband = high_inband[::self.decimation]
        p_clean, p_interf, p_noise = [float(np.mean(np.abs(x)**2)) for x in (clean, interf, noise)]
        significant = (np.abs(interf)**2 >= p_noise*10**(self.cfg.detection.label_margin_db/10)) & (interf != 0)
        for truths, kind in [(targets, 'target'), (interferers, 'interferer')]:
            lookup = {vid: j for j, vid in enumerate(self.components[kind]['ids'])}
            for truth in truths:
                truth.power_physical_w = truth.power_w
                truth.power_w = float(np.mean(np.abs(self.components[kind]['filtered'][lookup[truth.vehicle_id]])**2))
                truth.power_agc_w = None
        tr = TruthRecord(clean, interf, noise, mask_inband, significant & mask_inband,
            p_noise, ratio_db(p_clean, p_noise), ratio_db(p_interf, p_noise),
            ratio_db(p_clean, p_interf+p_noise), targets=targets, interferers=interferers,
            n_in_scene=len(states), noise_mode=rc.noise_mode, noise_power_ref_w=p_noise,
            mask_if_support=significant, mask_startup_invalid=np.zeros(rc.n_samples, dtype=bool),
            frontend_residual=np.zeros_like(observed), clip_residual=np.zeros_like(observed),
            label_schema='waveform-generation-v3')
        return ChirpFrame(session_chirp, session_chirp//rc.frame_chirps,
                          session_chirp%rc.frame_chirps, t0, observed, agc_scale=1., truth=tr)
