# configs/algorithms.py
"""感知器、定位器、重构器与调度器的算法参数。

约定:
  - 掩码一律为 True = 判定污染、需要重写; 所有重构器只允许写掩码内样本。
  - 采样率 fs=50 MHz 时 1 点 = 20 ns, 100 点 = 2 us。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SensingConfig:
    # 常驻轻量级感知 (L1)
    ewma_alpha: float = 0.99         # |x|^2 的 EWMA, 时间常数 ~100 点 = 2 us
    base_pct: float = 20.0           # 背景取帧内低分位 (%)
    base_ema: float = 0.05           # 背景跨 chirp 更新的步长 (仅可信样本)
    freeze_frac: float = 0.25        # 污染占比超过该值则冻结背景更新
    anomaly_gate_db: float = 3.0     # 能量异常判定的最低裕量
    block: int = 64                  # 流式处理块长 (点)
    reset_per_chirp: bool = True     # 每 chirp 重置 EWMA (背景 EMA 不重置)
    # 目标可见度 (距离 FFT 口径)
    vis_guard_bins: int = 6
    vis_hist_chirps: int = 8         # 参与轨迹支持的上一帧 chirp 数
    # 按需精细感知 (L2) —— 窗长/膨胀在 configs/algorithms.py 的 DetectionConfig
    # (stft_nperseg/stft_noverlap/stft_dilate), 这里只留 d2 真正读的两项。
    # 曾经有 fine_nperseg/fine_noverlap/fine_cfar_train/fine_cfar_guard/fine_dilate
    # 五个同名字段: 全仓无人读取, 调它们等于没调 (2026-10-02 审计删除)。
    fine_cfar_os: float = 0.625      # 按CUT可用参考单元数计算的OS分位
    fine_margin_db: float = 10.0     # 2026-10-06 D2 可信参考标定: 6->10 (三场景一致: 过度排除 -35~47%, 可信跨度 +10~43%, 虚警/漏检不变或更好); # 单频格判决门限 (dB, 相对 CFAR 底)
    # 背景估计口径对比: 'quantile' | 'os' | 'mode'
    background_method: str = 'quantile'


@dataclass
class DetectionConfig:
    # 2026-10-06 晋升: 时频域特征 (stft_log_v1: 加逐切片峰均比 + 脊线斜率一致性) 重训,
    # 训练池再平衡 + SimpleSignalSpec/常频音爆发同族样本。弱干扰召回 0.63→**0.872**
    # (边界收紧前 0.901, 旧模型 0.91), 帧级虚警 5/300→0/300, 道路召回 0.79→0.99。
    # 注意: 晋升的是 D1 **检测模型**; 默认动作库仍以 D2 为主 (D1 未进默认调度)。
    # 旧模型 detector_d1_calibrated.pkl / detector_d1_strength_v1.pkl 保留作对照。
    d1_model_file: str = 'detector_d1_tf_v5d.pkl'  # 独立版本；旧权重保留
    window_len: int = 100            # 滑窗特征窗长 (点), = 2 us
    train_step: int = 1
    test_step: int = 15
    label_margin_db: float = 0.0     # 窗级"显著污染"标签的能量裕量
    hidden: tuple = (32, 16)
    max_iter: int = 200
    decision_offset: float = 0.5     # MLP 概率判决偏移 (标定流上选出)
    # 2026-10-06 D1 弱干扰边界收紧 (仅 CFAR 支撑细化失败的回退路径): 短窗包络边界
    # 在当前掩码内再裁 (门限相对低尾分位)。道路弱工况实测: 误删 0.0657→0.0468
    # (逐帧口径 12.1%→约 9%), IoU 0.726→0.757, 召回 0.917→0.886; 强干扰路径不变。
    d1_boundary_trim: bool = True
    d1_trim_high_db: float = 4.0
    d1_trim_low_db: float = 2.0
    d1_trim_merge: int = 16
    d1_trim_env_window: int = 16
    # D0 时域 CFAR
    cfar_train: int = 16
    cfar_guard: int = 4
    cfar_k_db: float = 8.0
    cfar_os: float = 0.75
    cfar_dilate: int = 2
    # D0 受污染背景 (P1, 方案 §4.1): 低尾分位只作初值, 再剔除候选后重估; 参考不足
    # 返回 uncertain 而不是满置信度空掩码。门限/分位是工程参数, 待独立负对照标定。
    d0_background_quantile: float = 0.2
    d0_background_iterations: int = 3
    d0_min_reference_fraction: float = 0.10
    # 2026-10-05 D0 虚警修复: 写掩码只取能量分支, 门限独立于 cfar_k_db (默认 10 dB);
    # 时域 CFAR 分支在所有实测场景 (受控 15/0/-20 dB + 道路 t=6/12) 召回 0%, 却贡献
    # 干净帧 98.4% 的 chirp 级虚警 (面积 0.5%), 因此降级为候选证据 (candidate_mask,
    # 不授权重写)。独立对照: 能量门限 10 dB 时干净 chirp 虚警 0/192, 受控召回 84%
    # (8 dB 时虚警 52%、召回 96.6%); 标定/验证种子见 docs/D0_虚警修复_2026-10-05.md。
    d0_energy_gate_db: float = 10.0
    # 2026-10-05 D1 审计修复 (docs/D1_D3_R6_AC审计_2026-10-05.md): 干净帧 91.3% 触发是
    # **多重检验** —— 每帧 ~245 个窗, 干净窗概率的 0.1% 尾在帧级必现; MLP 本身可分性
    # 良好 (干净窗 p99=0.277, 干扰窗 p50=1.0)。修复 = 提高判决偏移 (干净+噪声对照标定)
    # + 要求连续 ≥N 个窗同时超门限。标定点: offset 0.99 / N=2 → 干净 chirp 0/1280、
    # 受控召回 1.000、IoU 0.69。偏移随模型 pkl 保存; 这里是缺省值。
    d1_min_consecutive_windows: int = 2
    # D2 STFT-CFAR
    stft_nperseg: int = 128
    stft_noverlap: int = 96
    stft_dilate: int = 2
    stft_peak_ratio: float = 16.0   # 谱峰需超过窗内谱中位数，抑制逐格最大值的噪声误报
    stft_broad_fraction: float = 0.20  # 宽带异常不要求孤立高峰，避免干扰抬高谱中位数自屏蔽
    stft_energy_ratio: float = 1.5    # 窗总能量须高于时间背景，抑制噪声多重检验
    stft_sweep_fallback: bool = True  # Conditional phase-verified D4 handoff; timed as part of detection.
    stft_energy_sigma: float = 10.0   # Robust reference dispersion guard, empirical calibration.
    stft_background_quantile: float = 0.20  # Initial only; exclude candidates and refit.
    stft_background_iterations: int = 3
    stft_min_reference_windows: int = 4
    stft_min_reference_fraction: float = 0.10
    # 轨迹确认分支 (阶段 A): 结构证据 + 净频移 + 连续窗数 + 单调方向, **不**依赖整体
    # 能量门控。相邻窗允许的最大频率跳变 (格); 连续窗数与净频移复用
    # sweep_min_windows/sweep_min_bins。单条扫频在一个窗内只占窄带, 结构格数上限
    # 按窗长比例给 (有色噪声的宽频带会凑出假轨迹, 实测误确认 75 窗/掩码 46%)。
    stft_traj_max_jump: int = 4
    stft_traj_max_cells_fraction: float = 0.05
    # 轨迹最短窗数: 4 窗时有色噪声的随机游走仍能凑出单调路径 (实测 4 窗误确认,
    # 5 窗起为 0); 真实扫频在 33% 污染下跨 ~10 窗, 不受影响。
    stft_traj_min_windows: int = 5
    # 边界保护 (P1, 方案 §4.1/4.2): 短窗复功率包络的双门限 + 短间断合并, 取代按瞬时
    # 幅度逐点切割。门限与合并长度是工程参数, 需要在独立负对照上标定; 这里的默认值
    # 只保证"振荡低谷不被切成干净点", 不宣称虚警率。
    boundary_env_window: int = 16     # 短窗长度 (点), 复功率包络
    # 确认核心门限取 9 dB: 包络是 16 点平滑功率, 方差低于瞬时功率, 用旧点门限的
    # 8 dB 会放宽虚警 (实测 middle 短脉冲组 0.0203 > 0.02)。9 dB 为**待独立标定**的
    # 临时工程值, 不是普适参数 (方案 §4.1)。
    boundary_high_db: float = 9.0
    boundary_low_db: float = 3.5      # empirical edge support; merge short valleys
    boundary_merge_gap: int = 8       # 可合并的最长低谷 (点)
    # 包络地板分位: 密集污染时 q20 也落在污染里, 会把真实边缘裁掉 (实测强污染
    # 边缘召回 0.974 < 0.99)。低尾 q05 尽量落进残余噪声; 全污染时本就没有干净参考,
    # 此时边界精修不可信, 只能报未知/不可恢复。
    boundary_floor_quantile: float = 0.05
    # 保护区长度 (点); 0 = 由公开 LPF 稳定时间现场推出 (不手填)。保护区只禁止进入
    # 拟合 (fit_exclude), **不**授权重写正式输出 (write_mask 不变)。
    boundary_guard_samples: int = 0
    startup_guard_samples: int = 0    # 启动无效区; 0 = 同上
    # D5 实验性观测残差定位；未加入默认动作库/闭环，稀疏污染与稳定拍频假设。
    residual_max_tones: int = 6
    residual_max_nfev: int = 30
    residual_scan_alpha: float = 0.01  # iid已知噪声Gamma/Bonferroni指导值，实际虚警需独立标定
    residual_point_ratio: float = 4.0
    residual_dilate: int = 1
    residual_min_core_points: int = 3  # 窗内至少3个高残差证据，抑制噪声孤岛切碎AR训练段
    residual_holdout_ratio: float = 2.5
    residual_max_fraction: float = 0.15
    residual_max_correlation: float = 0.20  # 与已声明噪声相关形状的偏差；仍拒绝目标失配
    residual_noise_profile: str = 'iid'  # iid | receiver_lpf；仅实际走同一LPF的输入显式启用
    # D3 多分辨率
    short_noverlap: int = 32
    short_nperseg: int = 64
    long_nperseg: int = 256
    long_noverlap: int = 192
    sweep_min_bins: float = 4.0     # D4/R6最少跨越FFT格数，近恒频不可辨识时拒绝
    sweep_peak_ratio: float = 16.0  # 候选峰相对窗内谱中位功率
    sweep_min_windows: int = 4
    # 距离-多普勒输出级检测 (evaluation/processor.py)。这是**评价口径**, 不是在线动作。
    # 门限在功率图 |·|² 上按 dB 功率比比较。数值来源: stage1 第 7(b) 节在 12 张纯噪声
    # (接收机 noise 分量走同一条 LPF+AGC+RD 链) 图上扫门限, 12 dB -> 9.7 个虚警/图,
    # 14 dB -> 0 个/图, 于是取"满足 0.1 个/图预算的最灵敏门限"14 dB。改它必须重跑标定。
    rd_margin_db: float = 14.0         # 峰值超出邻域中位参考多少 dB 才算检出
    rd_far_target: float = 0.1        # 标定时允许的逐帧虚警数 (纯噪声图上)
    rd_train: int = 8                 # 邻域参考窗半径 (格), CUT 自身包含在内
    rd_guard_r: int = 3
    rd_guard_d: int = 3
    rd_weak_rcs_m2: float = 3.0       # 弱目标口径 (RCS 阈值)
    # ⚠ 下面两对数字是**两件事**, 2026-09-29 评审第 6 条要求拆开:
    #   * 匹配容差 (rng_tol/vel_tol): "这个峰算不算命中那个对象", 要留足估计误差 (~5 bin);
    #   * 可分辨门限 (resolve_bins_*): "两个目标是否共用一个 RD 格", 只该由网格决定 (~2 bin)。
    # 以前两者共用 2.5 m / 1.5 m, 再叠加传递式 union-find, 结果"相邻差 1 m 的十个目标"被并成
    # 一个单元 —— 物理上分得开的目标被记成分不开, Pd 分母与弱目标保持率一起失真。
    rd_rng_tol_m: float = 2.5         # 匹配 (关联) 用距离容差 ≈5 个距离 bin
    rd_vel_tol_mps: float = 1.5       # 匹配用速度容差 ≈5 个多普勒 bin
    rd_resolve_bins_r: float = 2.0    # 可分辨判据: 距离维跨度 ≤2 bin (=1.0 m @0.5 m/bin)
    rd_resolve_bins_d: float = 2.0    # 可分辨判据: 速度维跨度 ≤2 bin (≈0.61 m/s @0.304)


@dataclass
class ReconstructionConfig:
    # R1 插值: 准入门限在 control/actions.py::_short_gap (按拍频周期表达)。
    # 旧字段 interp_max_gap 全仓无读者, 已删 (2026-10-02 审计)。
    # R2 复 Burg AR
    # 2026-10-06 评审: AR 自己的最低干净参考 (Burg 本体 20 点起; 不满足 MP 的 96
    # 不代表不能试短程 AR)。路由/动作准入用这个, 不用 mp_min_clean。
    ar_min_clean: int = 24
    allow_basic_mp: bool = False  # opt-in bounded suppression when outer geometry is unavailable
    basic_mp_max_inner_error: float = 0.25
    # TF-AR: continuous TF runs, independently fitted forward/backward models.
    tfar_nperseg: int = 128
    tfar_max_order: int = 16
    tfar_holdout_frac: float = 0.3     # geometric holdout selector; raw gap span has priority
    tfar_min_fit_frames: int = 6
    tfar_validation_guard_samples: int = 16  # ADC guard beyond held-out block; calibrate for LPF
    tfar_min_paired_improvement: float = 0.0
    ar_max_order: int = 24
    ar_train_len: int = 800          # 每侧最多用的连续干净样本数 (= 16 us)
    ar_default_max_pred: int = 800   # 每侧外推硬上限, 与 train_len 一起决定视界
    ar_max_mse_ratio: float = 0.25   # 创新误差预算 / 训练段功率
    ar_amplitude_limit_factor: float = 10.0
    ar_order_criterion: str = 'aicc'
    # R3 / R4 Matrix Pencil
    mp_max_order: int = 12           # 阶数上限由观测量与预算再收紧, 不是"目标数"
    mp_min_clean: int = 96           # 做频率估计所需的最少连续干净样本
    mp_iter: int = 3                 # 论文式迭代重构轮数
    # mp_rank_eps 已删: MP 的奇异值截断比在 reconstruction/matrix_pencil.py 里
    #   固定为 1e-3, 从没有读过来这个字段 (2026-10-02 审计)。
    mp_holdout_blocks: int = 5       # 留出块数 (连续块, 不逐点抽)
    # R4 受限非线性优化
    mp_opt_max_nfev: int = 35
    mp_opt_freq_slack_bins: float = 2.0
    # R5 低秩 Hankel + 稀疏
    hankel_rank: int = 14            # 阶数**上限**; 实际阶由奇异值谱定 (下一行)
    hankel_rank_eps: float = 5e-3    # 连续可见段SV相对阈值下限，另加中位SV噪声阈值; None = 固定阶
    hankel_lamb: float = 0.05        # 稀疏项判据之一: 残差 > lamb·rms(low)
    #   (另一个判据是 SPARSE_K 倍稳健σ, 见 reconstruction/lowrank_sparse.py)
    hankel_iter: int = 20            # 交替投影轮数 (补全收敛很慢, 见 lowrank_sparse 的实测)
    # **连续块留出**自检 (P3): 把一段干净连续样本对拟合完全藏起来再让模型填, 误差超预算
    # 就拒绝填充 (退路=挖掉)。留出块数直接乘进 R5 的每 chirp 耗时, 所以取小值。
    hankel_holdout_blocks: int = 2
    hankel_holdout_frac: float = 0.25     # 留出块长 = 最长干净连续段 × 本值
    # 留出段上模型误差必须 <= 本值 × 最好基线误差 (基线 = 填零 / 端点线性内插, 都不用真值)。
    # 取 0.85 = "至少比不动手好 15 %" 才允许往缺口里写; 绝对误差本身不能当判据 (噪声占主导)。
    hankel_holdout_gain_max: float = 0.85
    # 上面那条判据有**内在可通界**: 留出段里信号与噪声功率为 c/n 时, 任何"不预测噪声"的模型
    # 最好也只能做到误差 ≈ 噪声功率 n, 于是 model = n/(c+n) ; 要求 model <= 0.85·1 (填零基线)
    # 就等价于 c/n >= 0.15/0.85 = 0.176, 即 SNR >= -7.54 dB (评审 2026-09-29 第 7 条算的正是
    # 这一条)。低于它, "被拦下"不能当成"模型不可用"的证据 —— 那是题目本身无解。
    # 仅当 noise_w 明确是噪声功率时才算留出段SNR；ResidentSensor的含目标背景功率
    # 标为 signal_background，不作为热噪声。无法可靠估噪时仍执行留出误差门限。
    # 可解释的SNR低于本值时判 not_testable。取 -6 dB 而不是 -7.54 dB: 留一点余量, 免得把边界上的可测样本也扔掉。
    hankel_holdout_min_snr_db: float = -6.0

    # CPI 快时间x慢时间联合恢复 (2026-10-06, 路线三): 历史 chirp 联合拟合 +
    # 当前 chirp 独立成对验证; 低 SNR 弱目标靠跨 chirp 相干增益可估。
    cpi_grid_points: int = 1024       # 拍频粗网格点数 (覆盖 0~LPF 截止)
    cpi_history_chirps: int = 8       # 参与联合拟合的最大历史 chirp 数
    cpi_min_history: int = 3          # 少于此拒绝 (cold_start)
    cpi_min_train: int = 96           # 每 chirp 最少可信样本
    cpi_max_tones: int = 6            # 最多目标音数
    cpi_snr_threshold: float = 4.0    # 拟合半边单音相干能量/噪声下限 (再交给交叉验证筛)
    cpi_val_snr_threshold: float = 10.0  # 验证块上相干能量/噪声的阈值 (控制族错误率)
    cpi_val_block_min: int = 32       # 可信段至少这么长才产生验证块 (后 25%)
    cpi_val_guard: int = 16           # 拟合/验证块之间的隔离样本 (LPF 相关长度)

    # 目标-干扰联合分离 (P1 路线一, sep): 缺口 chirplet 拟合 + 目标重估
    sep_max_chirplets: int = 3        # 每个连续段内最多 chirplet 数
    sep_gap_min: int = 64             # 缺口少于这么多样本不做分离
    sep_gap_min_per_run: int = 48     # 连续段短于此不拟合 (初值/留出需要)
    sep_max_nfev: int = 60            # chirplet 优化迭代上限
    sep_min_paired_improvement: float = 0.02  # 段留出成对相对改进下限 (过滤噪声级分量)
    sep_min_slope_mhz_per_us: float = 0.15    # 轨迹线最小斜率 (低于此视为常频目标, 不建轨迹)
    sep_max_slope_mhz_per_us: float = 12.0    # 轨迹线最大斜率 (物理界: 斜率差 ~2×本车)
    sep_clean_gain_factor: float = 0.25       # 缺口样本加入拟合集的门: 减除后功率 < 0.25×原
    # R6: 掩码内相干chirplet对消；不覆盖传统缺口重建器。
    cancel_min_samples: int = 192
    cancel_holdout_fraction: float = 0.2
    cancel_holdout_ratio: float = 0.35
    cancel_max_nfev: int = 60
    cancel_max_tones: int = 6
    cancel_max_coherence: float = 0.3
    # 2026-10-06 P2 结构化对消 (cancel): 单 chirplet + 目标音模型; 掩码内连续后段
    # 留出成对检验 (复用 cancel_holdout_ratio 作裕量, 待独立标定)。
    # Experimental alternatives, not enabled in the calibrated action library.
    ac_min_reference_windows: int = 4
    ac_dilate_frequency: int = 2  # FFT main-lobe/leakage protection, empirical
    ac_dilate_time: int = 1       # no wrapping across record endpoints
    # Explicit cfar_ac_enhanced hybrid, not paper AC and not a calibrated action.
    ac_noise_window_bins: int = 11
    ac_line_power_ratio: float = 3.0
    ac_min_amplitude_gain: float = 0.5  # never zero a weak line from lack of significance
    ac_min_phase_coherence: float = 0.75
    ac_holdout_ratio: float = 0.50
    # 2026-10-05 AC 可验证化: 相干扩展的**连续参考窗块留出**成对裕量。留出块跨度
    # ≥ 实际缺口, 拟合只用其余参考窗; D/baseline > 本值才允许 verified (否则拒绝)。
    # 待独立负例标定 (与 mp/cross 裕量分开)。
    ac_min_paired_improvement: float = 0.05
    cross_history_chirps: int = 16
    cross_min_history: int = 4
    # 阶段 C: 同一快时间频率允许的慢时间 Doppler 线条数 (同距离不同速度)。
    # 旧模型每个快频率只配一条, 两个同拍频目标会互相污染; 贪心提取, 留出验证决定。
    cross_max_dopplers: int = 2
    cross_max_tones: int = 3
    cross_holdout_ratio: float = 0.98  # 旧固定比值: 保留为诊断/负对照记录, 不再作准入
    # 阶段 C 成对准入: 留出块上 D = 基线误差(不动作) - 候选误差, 要求
    # D > 本值 × 基线误差。0.005 是**临时**裕量 (噪声负例在候选阶段已被拒, 未触到该门),
    # 必须在独立负对照上重新标定后才可收紧/放宽。
    cross_min_paired_improvement: float = 0.005
    # 2026-10-05 R3/R4/R2 的成对留出裕量 (与 cross 的分开标定): 修复视界上限后, 长缺口
    # 可测了, 但 `D>0` 无裕量在 72 个强制掩码负例上放行 11/72 (r3) / 8/72 (r4);
    # 负例 D/baseline 最大 0.017 (r3) / 0.046 (r4), 被准入的正例 ≥0.39 → 0.05 分离。
    # 待独立验证集复核 (72+72) 后收紧/放宽; R2 暂共用, 需单独标定。
    mp_min_paired_improvement: float = 0.05
    cross_min_phase_coherence: float = 0.85
    # 2026-10-06 评审落地: 弱残留污染"干净"样本时, 用 IRLS/Huber 稳健拟合替代
    # 纯 LS + 硬相位门; 相位解释度不足但内点占比高时仍进候选 (最终仍由留出 D 决定)。
    cross_huber_c: float = 1.345      # Huber 阈值 (标准 1.345σ)
    cross_huber_iters: int = 3        # IRLS 迭代数
    cross_min_inlier_fraction: float = 0.8   # 稳健残差判据: 慢时间内点占比下限
    cross_min_outlier_energy: float = 0.5    # 稳健残差判据: 离群样本占残差能量下限
    cross_max_gap_periods: float = 3.0
    # 2026-10-05 跨帧空闲续用 (默认开)。**不放宽**上面的 within-frame 守卫: 只有
    # "长但有限、时间戳严格递增、AGC 变化 ≤2×"的空闲才续用, 且下一条 chirp 必须通过
    # 跨空闲留出 (当前 chirp 的干净样本整条留出、不参与拟合, 用空闲前的历史按绝对时间
    # 预测它) 才允许写入; 全污染则拒绝, 不做无证据外推。
    # 依据: 阶段 C 的"必须先验证绝对时间相位预测与目标生灭, 而不是放宽
    # cross_max_gap_periods"; 帧空闲审计 (3 种子) 写入 32→60 / 11→15 / 2→3,
    # 含拒绝 NMSE 16.7→0.017 等全面下降; 目标消失/时间回退/AGC>2× 反例全部拒绝。
    cross_idle_reuse: bool = True
    cross_idle_max_s: float = 5e-3
    # 2026-10-05 增量跨 chirp (评审实施顺序第 3 项, 默认关): 稳定期复用上一帧已验证的
    # 频率/多普勒线, 只用当前 chirp 的训练观测做幅相 LS 更新 + 留出成对检验; 留出不过、
    # 模型过期、跨空闲、当前无支撑或周期上限到点 -> 回退完整搜索 (事件触发重搜)。
    # 属算法变化: 晋升前须重新验证目标出生/消失、同距不同速与弱目标保持。
    cross_incremental: bool = False
    cross_incremental_max_age: int = 4      # 缓存模型最多复用几个 chirp 周期
    cross_incremental_max_streak: int = 8   # 连续复用上限, 到点强制完整搜索 (查新目标)
    # 2026-10-06 第二步复核 (新目标出现): 增量幅相 LS 后, 训练样本上的**未解释残差**
    # 占比超过本值 -> 拒绝增量、回完整搜索。配对改进的基线太弱, 解释不了"新增线"
    # (实跑: 新目标帧 incremental_reuse=True 照常放行, 缺口漏掉新线); 残差检查补上
    # 这一条。0.15 为保守初值 (稳定期残差≈噪声占比 ~1e-3), 待独立标定。
    cross_incremental_max_residual_fraction: float = 0.15
    # 2026-10-05 P0: 失败搜索成本上限与退避 (评审: 连续无目标线时每条 chirp 付近 1 s
    # 的搜索成本, 必须限最坏; "历史不足"与"反复无目标线"不能同样重试频率)。
    #   * cross_search_max_s: 单次完整搜索的 PC 墙钟上限 (分段检查, 超时拒绝);
    #   * cross_max_rank_trials: 阶数循环上限;
    #   * 持久失败 (无目标线/相位失配/无成对改进) 按 2^streak 退避, 上限 max 个周期;
    #     干净支撑显著变化或周期到点会重新尝试 (新目标仍能及时重新发现)。
    cross_search_max_s: float = 0.5
    cross_max_rank_trials: int = 3
    cross_backoff_base: int = 4
    cross_backoff_max: int = 32
    # 2026-10-06 第二步: 异步完整搜索 (默认关)。需要完整搜索时当前帧走保守退路
    # (复合臂同掩码置零, reason=search_deferred), 搜索任务入队 (队列上限 1, 新任务
    # 替换旧任务); 输出定稿后执行 (executor.run_deferred_searches), 结果只写增量缓存
    # 供后续帧复用。一致性: 配置签名/任务年龄/任务帧之前的付费历史不足 -> 取消。
    # 后台消耗仍计入 t_total (异步不减少算力, 只保证当前输出不依赖搜索)。
    async_search: bool = False
    search_task_max_age_chirps: int = 4
    # 2026-10-05 P2b 二维联合稀疏基线 (实验, 不进默认库): 多 chirp 可信观测 ->
    # (n_fast × D) 系数 FISTA (慢时间基用真实时刻), 外层连续留出成对检验才 verified。
    # λ = js_lambda × max|C=0 梯度| (尺度自适应); 迭代数固定, 成本上界可算。
    js_history_chirps: int = 12
    js_min_history: int = 4
    js_dopplers: int = 33
    js_iters: int = 40
    js_lambda: float = 0.05
    js_max_support: int = 4          # 支撑上限 (选峰/去偏共用)
    js_peak_floor: float = 0.1       # 选峰地板 (相对最强峰; 压 −13 dB 旁瓣)
    js_support_rel: float = 0.05     # (仅网格 FISTA 路径) 支撑相对剪枝
    js_use_fista: bool = False       # 默认 MP 选峰+连续细化; True = 细网格 FISTA 对照
    js_min_paired_improvement: float = 0.05
    # 失败判据 (统一)
    fail_rel_energy_floor: float = 1e-6   # 掩码内输出相对观测量级低于此判为失败


@dataclass
class ActionConfig:
    """动作库候选: (定位层级, 重构层级) 组合。

    名字在 control/actions.py 里展开成实际对象, 这里只管开/关与优先级偏置。
    """
    # 2026-10-05 精简 (评审实施顺序第 6 项): 默认只保留有证据的角色 ——
    #   bypass                   可信干净时直通 (低时域能量不单独放行);
    #   d2_zero                  低成本基线与终端退路 (由 d2 启用自动补);
    #   d2_cross_chirp_zero      D2+跨 chirp+失败置零 (增量/完整搜索为同一臂的计算档)。
    # 被移出默认的臂保留为可选 (显式 enabled 即用), 证据:
    #   d0_r1 pre_eligible 实际不成立; d0/d1 帧级虚警 15.7%/91.3% (docs/帧级虚警独立标定);
    #   d2_r2 与 d2_zero ΔSINR 并列但 p50 6.19 vs 0.1 ms (被支配); d2_r3/d2_r4 verified≠恢复;
    #   d3_r5 p50 2.2 s 仅离线参照。
    # 2026-10-06 P0: 默认在线库**不含置零与复合置零退路** (它们仍注册, 显式启用可做
    # 消融)。P1-P4 完成后并入五路线候选池: 双向AR(r2)/MP精修(r4)/增强AC/结构化对消/
    # 跨 chirp; 规则路由 (routing='rules') 按干扰特点在它们之间选择, 调度器默认仍可
    # 用 (Q 未标定新臂时由规则路由/固定选择承担)。
    enabled: tuple = ('bypass', 'd2_r2', 'd2_r3', 'd2_r4', 'd2_cfar_ac_enhanced',
                      'd2_cancel', 'd2_cross_chirp', 'd2_seg')
    # 'none' preserves historical runs; 'd1' requires a newly calibrated context table.
    prelocator: str = 'none'
    # 2026-10-06 评审: 单方法诊断需要关闭外部回退, 否则失败帧会被回退链顶替,
    # 结果无法归属到实际算法 (消失探针里 cpi 的拒绝被 cross 的 verified 掩盖)。
    retry_fallback: bool = True
    # 2026-10-05 重建路由模式 (实验): 'scheduler' = 现状 (打分+recheck);
    # 'rules' = 检出后按 control/routing 的规则选重建方法 (冷启动单段缺口优先 R3/R4),
    # 换路由仍走预算二次准入, 路由臂失败先由回退链/复合臂兜底, 最后终端置零。
    # 2026-10-06 P0 质量优先模式 (默认关): 选择只看质量, 不看耗时。
    #   * 调度器: 去掉时间惩罚与预算可行性过滤、去掉切换惩罚 (score = Q);
    #   * 管线: 去掉超预算降级、重探/回退的预算门、recheck 的预算降级;
    #   * 跨 chirp: 去掉 cross_search_max_s 搜索截断, 迭代上限放宽到 quality_max_nfev。
    # 耗时照记 (t_total/模型), 只是不参与选择。**不取消任何可靠性检查**:
    # 可信样本、相位一致、独立留出、视界、目标保护、残差防线全部保留 —— 观测不足
    # 被拒的方法, 增加计算也救不回来, 依然拒绝。
    quality_first: bool = False
    quality_max_nfev: int = 100
    # 2026-10-06: 默认改为**规则路由** —— 实时闭环按干扰特点在五路线候选池
    # (AR/MP/AC/对消/跨 chirp) 间选择; 'scheduler' 仍可选 (Q 表需按新臂重标定后才
    # 有竞争力)。规则来自 P1-P4 实测优势边界, 各路线结果仍"验证通过才 verified"。
    routing: str = 'rules'
    routing_detector: str = 'd2'
    switch_penalty: float = 0.05     # mu * 1(a != a_prev), 无量纲质量分
    # 2026-10-05 质量标签的峰损失权重: 标定 Q = ΔSINR − w × 目标峰损失(dB)。
    # **默认 0.0**。试过 w=1: 闭环转向小掩码的 d1_zero (ΔSINR −0.41 dB、虚警 1.0→0.3),
    # 但评审第 6 条明确"目标峰值不能直接作为更好判据" —— 统一验收要求**固定帧级虚警
    # 预算**下的弱目标 Pd/目标漏失, 而不是把峰值损失塞进在线标签。目标保护改由复合
    # 路由 (d2_cross_chirp_zero) 承担, 它在固定条件下的对照里已弱支配置零。w 保留为
    # 研究开关, 开启时必须同时报告虚警与 ΔSINR 的交换。
    quality_label_peak_weight: float = 0.0
    # P3: 已验证质量证据的受限在线项 (独立留出证据 EMA)。它**不写回**真值训练的 Q 表,
    # 只以一个小权重叠加在调度分数上; weight=0 或 feedback=False 即无质量反馈消融。
    # 只在置信度 >= control.quality.QUALITY_MIN_CONFIDENCE 时更新 (方案 §M4.3)。
    # 2026-10-04 P4 有界道路对照: 权重 0.1 的 closed 臂 ΔSINR 3.35 < 无反馈 3.49, 且
    # 超响应截止率 61% vs 18% (1 ms 档) —— 道路会话上无净收益, **默认降为 0.0**
    # (机制保留, 研究时显式开启; 证据 artifacts/stage4_budget_ladder.json)。
    quality_evidence_weight: float = 0.0
    quality_evidence_rate: float = 0.05
    # 2026-10-06 第三步: 质量反馈三态 (语义写死, 不再靠"weight=0 但还在积累"的隐式状态):
    #   'disabled' = 不更新 EMA、不参与排序 (与 feedback=False 等价);
    #   'shadow'   = 记录证据与预测 (EMA 累积 + detail 留痕), **不参与排序** (权重强制 0);
    #   'active'   = 经独立验收后, 按 quality_evidence_weight 有限影响排序。
    # 默认 disabled; 影子 -> 生效必须过"能识别可改善的坏结果 + 开启反馈带来同预算收益"。
    quality_mode: str = 'disabled'


