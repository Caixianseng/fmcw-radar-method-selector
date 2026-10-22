# configs/traffic.py
"""交通层与雷达时序层配置。

三个时间尺度 (技术方案第五节):
  交通层   dt_traffic = 5 ms —— 出厂场景传的是 simulation/scenarios.py 的 DT_TRAFFIC;
            ScenarioConfig 的字段默认值仍是 1 ms, 只有显式不传它的测试在用
  雷达时序层 每辆车的 chirp 周期、帧周期、初始时偏、时钟漂移, 入场时一次分配
  接收信号层 每个受害者 chirp 内按 fs 逐点算传播时延、天线增益、拍频与功率
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InterfererRadarConfig:
    """干扰雷达身份参数的随机范围 (进入场景时一次分配, 之后按时钟运行)。

    斜率差 ΔK = K_v - K_j 才是决定"污染有多长"的量, 不是 chirp 重复周期差:
        差拍瞬时频率 f_b(t) = [f0_v - f0_j - K_v*t_v + K_j*t_j] + ΔK * t
    带内时长 = |f_b| <= lpf_cutoff 与发射重叠窗的交, 上界是 min(重叠时长, 2*lpf_cutoff/|ΔK|);
    但这个上界只有当差拍**中心** f_c = f0_v - f0_j - K_j*Δt 够得着通带时才拿得到 ——
    ΔK 很小而时偏 Δt = 20 us 时 f_c ≈ -80 MHz, 整段都在带外, 带内时长是 0 不是 75 us
    (实测见 experiments/param_factor_sweep.py)。所以 ΔK 必须能独立
    控制, 不能靠 chirp_period_jitter 代偿 —— 那两个是不同参数。
    min_relative_slope_gap 必须为正: 出厂场景一律禁止完全同斜率 (要它做解析对拍时
    直接构造 ChirpTimeline)。

    时序自洽 (P1 加的约束): 扫频时长 T 与重复周期 P 不是独立的 —— 一个 chirp 必须
    装下"扫频 + 复位/空闲"。所以先抽 T, 再取
        P ~ U( max(P_v(1-j), T + min_reset_s),  P_v(1+j) )
    下界就是可行性约束本身, 于是**按构造**不会出现 T + 复位 > P 的身份; 装不下的 T
    直接判为不可达并重抽 (并把拒绝原因计数打印出来)。实测旧实现是各抽各的, 10000 台
    里 4974 台发的出来才怪。
    """

    frame_chirps: int = 0
    frame_period: float | None = None  # configured profile, not a population distribution

    chirp_period_jitter: float = 0.06     # 相对受害者的周期失配 ±6%
    period_range_s: tuple = ()            # 非空 = 直接用绝对周期带 (文献固定参数基准用)
    min_reset_s: float = 5e-6             # 复位+空闲下限: 工程选择, 量级参照 Roudiere 图1
    #   的 4 us 复位 (300 MHz/20 us 扫频, 接收带宽 30~40 MHz); 不是论文给的通用值。
    start_offset_range: tuple = (0.0, 300e-6)   # 初始时偏 (s)
    clock_drift_ppm: float = 50.0         # 时钟漂移 (ppm), 长期改变相对时偏
    bandwidth_range: tuple = (240e6, 360e6)     # 干扰带宽 (Hz)
    duration_range: tuple = (75e-6, 75e-6)     # 干扰 chirp 扫频时长 (s); 默认锁死,
    #   斜率多样性只由 bandwidth_range 提供 —— 这样"带宽+时长都固定"的配置必然配出
    #   完全同斜率, 会被 assign_radar 拒绝 (tests/test_interferer_slopes.py 钉住这条)。
    #   注意它是**容许带**: 配上 min_reset_s 之后真正可达的带上界会被削掉 (见上面)。
    duration_choices_s: tuple = ()       # 非空时用离散扫频时长替代 duration_range
    duration_choice_probs: tuple = ()    # 与 choices 一一对应, 非负且总和为 1
    repeat_idle_s: float | None = None   # 非 None: P = T + idle, 覆盖原周期带抽样
    #   每辆车只在入场时抽取一次 T; idle 必须至少为 min_reset_s。
    slope_sign_probs: tuple = (0.6, 0.4)        # 上扫频/下扫频概率
    min_relative_slope_gap: float = 1e-4  # |K_i-K_v|/|K_v| 下限, 必须为正: 出厂场景
    #   一律不允许完全同斜率 (那是退化且非现实的用例)
    tx_power_range_dbm: tuple = (10.0, 13.0)


def radar_preset(name: str, **kw) -> InterfererRadarConfig:
    """四组波形设定: 两组是我们自己的敏感性分布, 两组是文献实测的固定参数。

    near            ΔK/K ∈ [1e-4, 0.002]: 慢扫, 一个 chirp 内只挪 ±0.1%·B, 近似窄带驻留
                                          —— MP/长缺口这一路的压力用例
    sensitive       带宽 240–360 MHz、扫频 60–100 us、周期 ±6% 独立随机
                                          —— **本项目的自定义敏感性分布**。它不是任何论文
                                             给的统计: 之前把它叫 "diverse" 并在文字里挂到
                                             Roudiere 名下是错的, 已改名。
    roudiere_fixed  Roudiere 2023 (Sensors 23:5765) 图 1 的那台雷达: B=300 MHz, 扫频 20 us,
                    复位 4 us, 接收带宽 30 MHz (可至 40 MHz)。周期 24 us = 扫频+复位, 是
                    **我们的取值** (论文只给了扫频与复位; 它那句 50% 占空比讲的是时频资源
                    划分, 不是 chirp 帧)。K=15e12 Hz/s 是受害者 4e12 的 3.75 倍 -> 差拍
                    穿带极快 (2*f_LPF/|dK| ~ 3.6 us), 污染是"零星短段"。
    prinz_fixed     Prinz 2021 (LNICST 364, 表 1) 的异步干扰设置: 两台都 B=425 MHz,
                    受害者 25.6 us/42 us、干扰机 22.5 us/38.8 us (K=16.60 vs 18.89 e12,
                    相对斜率差 13.8%)。这里把两档扫频时长摊成一个连续带 [22.5, 25.6] us
                    和周期带 [38.8, 42] us —— **表 1 是两组离散值, 不是分布**。

    完全同斜率 (ΔK=0) 不在这里提供: 出厂场景一律要求 gap>0。需要它做解析对拍时,
    直接构造 ChirpTimeline (见 tests/test_two_vehicle_benchmark.py), 不要绕过这个不变量。
    """
    presets = {
        'near': dict(bandwidth_range=(299.4e6, 300.6e6), duration_range=(75e-6, 75e-6),
                     slope_sign_probs=(1.0, 0.0), min_relative_slope_gap=1e-4,
                     chirp_period_jitter=0.002),
        'sensitive': dict(bandwidth_range=(240e6, 360e6), duration_range=(60e-6, 100e-6),
                          slope_sign_probs=(0.6, 0.4), min_relative_slope_gap=1e-4,
                          chirp_period_jitter=0.06),
        'roudiere_fixed': dict(bandwidth_range=(300e6, 300e6), duration_range=(20e-6, 20e-6),
                               period_range_s=(24e-6, 24e-6), min_reset_s=4e-6,
                               slope_sign_probs=(1.0, 0.0), min_relative_slope_gap=1e-4,
                               tx_power_range_dbm=(12.0, 12.0)),
        'prinz_fixed': dict(bandwidth_range=(425e6, 425e6),
                            duration_range=(22.5e-6, 25.6e-6),
                            period_range_s=(38.8e-6, 42.0e-6), min_reset_s=4e-6,
                            slope_sign_probs=(1.0, 0.0), min_relative_slope_gap=1e-4,
                            tx_power_range_dbm=(7.0, 12.0)),
    }
    if name not in presets:
        raise KeyError(f'未知斜率设定 {name}; 可选 {sorted(presets)}')
    return InterfererRadarConfig(**{**presets[name], **kw})


@dataclass
class VehicleConfig:
    length_m: float = 4.5
    rcs_mean_m2: float = 12.0             # 轿车正面 RCS 均值
    rcs_sigma_log: float = 0.45           # RCS 对数正态散布
    has_radar_prob: float = 0.4           # 该车装前向雷达的概率 (车队渗透率)
    # 几何遮挡用的"车身矩形" (P2 的两档敏感性之一): 长 = length_m, 宽取轿车标准值 1.8 m。
    # 工程选择, 没有实测依据; 早先用"半径=车长/2 的圆盘"等效于 4.5 m 宽, 遮挡被高估。
    # 全场统一按轿车档, 行人/摩托车弱目标也不例外 (交通层刻意简化, 不做车辆类别建模)。
    occluder_width_m: float = 1.8



@dataclass
class LaneConfig:
    """一条车道。waypoints 一律按行车顺序给 (起点 -> 终点), 不设"反向"标志,
    对向车流就是把折点反着写 —— 少一个概念, 也避免位置与速度方向不一致。

    closed=False 是**有出入口的开放路段**: 车从 s=0 进、从 s=length 出 (消失),
    不再沿弧长绕回。要"疏->密->散"真实演化 (而不是控制器凭空增删) 就必须用它。
    """
    name: str = 'main'
    waypoints: tuple = ()                 # ((x,y), ...) 道路中心线折点 (m)
    lateral_offset_m: float = 0.0         # 车道相对中心线的横向偏移 (m)
    speed_limit_mps: float = 33.3
    closed: bool = True


@dataclass
class ScenarioConfig:
    """一个连续交通场景。

    density_profile 给出 (时刻 s, 相对密度 0~1) 折点, 线性插值: 0=稀疏, 1=拥堵。
    密度这个旋钮接到哪里, 取决于 population_mode:

      * 'density' (闭合环, 出厂场景): 同时决定 ①期望速度
        v0(d)=限速·[1−(1−congestion_speed_factor)·d]; ②每车道目标车辆数
        n(d)=环长/IDM 稳态间距。环上没有真实出入口, 所以车数是**控制器直接给的**
        (台账 self.ledger 会如实记下每一次凭空 in/out) —— 这叫"密度调控仿真"。
      * 'fixed'   (受控基准): 车数固定为 fixed_counts, 全程零增删, 密度就按每车道
        N/L 计量。用来回答"没有凭空增删时, 车流自己会怎么样"。
      * 'flow'    (开放路段): 车数由**入口到达率**给 —— 时距 h(d) 在
        [min_headway_s(堵), max_headway_s(疏)] 之间插值, 车从下游端真实离场,
        于是 N(t+dt)−N(t) = N_in − N_out 每一项都有物理来源。

    车头时距为什么在 'flow' 里才是输入: 在环上它若被当成 IDM 的期望时距, 会与
    "环上放得下几辆车"打架 (26 m/s × 4 s = 104 m, 491 m 的环放不下) -> 永久压缩队列。
    """

    name: str = 'unnamed'
    duration_s: float = 40.0              # 会话长度: 必须容得下密度平台 (见 scenarios 头部)
    dt_traffic: float = 1e-3
    lanes: tuple = ()                     # tuple[LaneConfig, ...]
    # (时刻 s, 相对密度 0~1) 折点; 出厂场景一律用 simulation/scenarios.py 的 DENSITY_CYCLE
    density_profile: tuple = ((0.0, 0.10), (5.0, 0.10), (13.0, 0.95), (22.0, 1.00),
                              (28.0, 0.15), (40.0, 0.15))
    ego_lane: int = 0                     # 受害者雷达所在车道索引
    ego_start_s: float = 20.0             # 受害者在车道折线上的起始弧长 (m)
    ego_speed_mps: float = 25.0           # 受害者车辆的目标速度 (IDM 的 v0, 不乘拥堵因子)
    congestion_speed_factor: float = 0.35  # 密度=1 时**期望**速度乘子 (只作用于普通车流)
    # IDM (Treiber, Hennecke & Helbing 2000, cond-mat/0002177; 参数取作者给出的推荐值:
    # T=1.5 s, s0=2.0 m, delta=4, a/b 在 0.3~1.4 / 2.0~3.0 区间按车辆类型取)。
    # 之前用 a=2.5, T=1.1 比推荐值激进, 会让跟驰间距偏小、拥堵段的车头时距不真实。
    idm: dict = field(default_factory=lambda: {
        'a_max': 1.4, 'b_comfort': 2.0, 's0': 2.0, 'T': 1.5, 'delta': 4.0})
    interference_enabled: bool = True
    radar_cfg: InterfererRadarConfig = field(default_factory=InterfererRadarConfig)
    vehicle_cfg: VehicleConfig = field(default_factory=VehicleConfig)
    # 弱目标与强干扰车同时存在: 在指定弧长上放一个低 RCS 目标 (行人/摩托车量级)
    weak_targets: tuple = ()              # ((lane, s_m, speed_mps, rcs_m2), ...)
    # 可控的确定性车辆: (lane, s0, v0, rcs_m2, 是否装雷达)。随机车流不能保证
    # "会话窗口内一定有一次近距会车", 关键实验场景需要一个跑不掉的相遇。
    extra_vehicles: tuple = ()
    # ---------------- P2: 三种车流来源与两个几何附加项 ----------------
    initial_counts: tuple | None = None  # None preserves old seeding; explicit counts exclude extra vehicles/ego
    arrival_lanes: tuple | None = None   # None allows all open lanes; otherwise only these lane indices admit arrivals
    population_mode: str = 'density'      # 'density' | 'fixed' | 'flow'
    fixed_counts: tuple = ()              # 'fixed' 时每车道车辆数 (不含受害者)
    min_headway_s: float = 1.2            # 'flow' 入口到达时距: 拥堵端
    max_headway_s: float = 4.0            # 'flow' 入口到达时距: 稀疏端
    # 冲突区 ((x, y, r_m), ...): **只用于度量** —— 数穿模事件 + 铺点卫生 (不许凭空把车放在
    # 箱子里)。本模型不建模让行: 试过"接受间隙比较"与"时间窗预约"两轮, 都不满足验收,
    # 数字与根因见 simulation/traffic.py 模块头。
    conflicts: tuple = ()
    # 急刹判定倍数: |a| > b_comfort × 本值 记一次 hard_brake (纯跟驰模型下应当极少)。
    hard_brake_factor: float = 2.5
    occlusion_enabled: bool = False       # 几何遮挡 (两档敏感性对比的第二档)
