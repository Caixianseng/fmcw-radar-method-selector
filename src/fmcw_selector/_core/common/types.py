# common/types.py
"""贯穿各层的数据结构。字段含义以注释里的单位/形状为准。

真值 (TruthRecord) 只允许 evaluation/ 与离线标定读取; sensing/detection/
reconstruction/control 只能看到 ChirpFrame.observed 与其自身可算出的量。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class TargetTruth:
    """一个散射中心的真值 (仅离线)。"""
    vehicle_id: int
    range_m: float
    radial_speed_mps: float     # 逼近为正
    rcs_m2: float
    beat_hz: float
    power_w: float
    in_band: bool
    # 视线被别的车身挡住 (simulation/occlusion.py): 本 chirp 里它**没有回波**, 但车还在
    # 场上。不区分"被挡住"与"探测算法没查到"就会把几何问题算成算法失败。
    occluded: bool = False
    power_physical_w: Optional[float] = None  # nominal link budget before AGC/LPF
    power_agc_w: Optional[float] = None       # same nominal power after common AGC


def target_observable(t) -> bool:
    """本 chirp 里这个散射中心**应当**有可检测回波: 在通带内, 且没被别的车身挡住。

    评价层的两处统计必须用同一个判据: `evaluation/processor.frame_targets` (Pd 的分母)
    过去只看 `in_band`, 而 `evaluation/offline.target_peak_loss_db` 看
    `in_band and not occluded` —— 一旦某条路径给出 occluded=True 且 in_band=True
    (测试桩就是这么构造的), 同一个目标会在一处进分母、在另一处不算, Pd 与目标峰损失
    就不是同一批目标。当前 receiver 把遮挡目标一律登记成 in_band=False, 所以实跑不触发;
    把这个判断收成一个函数是为了让它**不可能**再分叉。
    """
    return bool(t.in_band and not getattr(t, 'occluded', False))


@dataclass
class InterfererTruth:
    """一个干扰源的贡献 (仅离线)。"""
    radar_id: int
    vehicle_id: int
    range_m: float
    power_w: float
    beat_slope_hz_s: float      # d f_beat / dt
    beat_hz_center: float
    duty: float                 # 本 chirp 内落在通带里的时间占比
    inband_mask: np.ndarray     # bool[N], 几何/通带有效窗
    occluded: bool = False
    power_physical_w: Optional[float] = None  # nominal link budget before AGC/LPF
    power_agc_w: Optional[float] = None       # same nominal power after common AGC


#: 真值标签 schema 版本。改动任何一个 mask_* 的定义时必须升这一位, 并**同时**用同一批
#: 数组重算新旧算法的结果 —— 只替新标签改分母会让新旧算法不在一个口径上比 (M0-C)。
#: v2: 新增 `mask_if_support` / `mask_startup_invalid` 两条独立支持集;
#:     `mask_significant` 的定义原样冻结 (旧报告的 IoU 仍然可比)。
LABEL_SCHEMA_VERSION = 'labels-v2'


@dataclass
class TruthRecord:
    clean: np.ndarray                   # complex[N] 仅目标回波 (含其多普勒), 无噪声无干扰
    interf: np.ndarray                  # complex[N] 干扰分量之和
    noise: np.ndarray                   # complex[N] 热噪声实现
    mask_inband: np.ndarray             # bool[N] 任一干扰落在通带内的几何窗
    mask_significant: np.ndarray        # bool[N] 局部干扰功率 >= 底噪*10^margin 的显著污染
    noise_power_w: float                # 与下面三条曲线同一 AGC 口径的单点噪声功率 (W)
    snr_db: Optional[float]             # 输入 SNR: 10log10(P_clean/P_noise); off 模式无定义 -> None
    inr_db: Optional[float]             # 输入 INR: 10log10(P_interf/P_noise)
    sinr_db_in: Optional[float]         # 输入 SINR (分量功率口径): 10log10(P_clean/(P_interf+P_noise))
    targets: list = field(default_factory=list)
    interferers: list = field(default_factory=list)
    n_in_scene: int = 0                   # 本 chirp 场上车辆数 (不含本车): 干扰源集合的大小
    unrecoverable: bool = False         # 无有效观测/削顶/不可辨识的标记 (由底座判定)
    # ---- 2026-10-11 v3: 噪声模式与解耦后的参考功率 ----
    # `noise_mode`: 'off'/'scaled'/'physical' (实际注入规则);
    # `noise_power_ref_w`: 固定的分量门限与干扰标签基准 (同一 AGC 口径), 与模式无关;
    # off 模式实测噪声=0, SNR/INR 置 None, 评价改用重建误差/检出/损伤/虚警。
    noise_mode: str = 'scaled'
    noise_power_ref_w: float = 0.0
    # ---- 2026-10-04 M0-C: 三种"支持集"是三种不同的东西, 压成一个标签就会互相遮蔽 ----
    # `mask_inband` 是**射频域**判据: |f_beat| <= lpf_cutoff, 也就是模拟抗混叠过渡带的
    # 顶端 (增益仍为 1 的那条边)。而 `_anti_alias` 在 lpf_cutoff~lpf_stop_edge 之间给
    # 0<增益<1, 数字 LPF 还会把能量展宽。于是通带外**确实**存在高于底噪的 IF 干扰
    # (实跑: 缺口两侧各 59/68 点, 边沿处 |interf|^2 达噪声的 67~229 倍)。
    # `mask_significant` 过去与 mask_inband 求交, 把这部分整段切掉 —— 定位器如果正确地
    # 把拖尾标出来反而会被记成"标签外误删"。所以两条分开存, 各自单独可查:
    mask_if_support: Optional[np.ndarray] = None
    #   ^ bool[N] 与 mask_significant 同一 AGC 口径、同一滑窗判据, 但**不**与 RF 窗求交。
    mask_startup_invalid: Optional[np.ndarray] = None
    #   ^ bool[N] 每 chirp 零状态启动的数字滤波器尚未进入稳态的那一段。既不是干净也不是
    #     干扰, 是前端状态无效; 削顶与不可辨识另由 adc_clipped_frac / unrecoverable 表达。
    # ---- 2026-10-04 M1-A: 前端连续状态的两条"公共尺度"残差 (仅非默认前端模式填充) ----
    # 只在 ``radar.frontend_mode='continuous'`` 时非 None; 旧模式保持 None, 让旧表/旧
    # 数组的语义完全不变。两条都与 clean/interf/noise 同一 AGC 尺度, 且满足分解恒等式
    #   observed = clean + interf + noise + frontend_residual + clip_residual
    # (分解是在线性滤波器上按 T(x,s)=T(x,0)+ZIR(s) 做的, 不把状态响应平摊进分量)。
    frontend_residual: Optional[np.ndarray] = None
    #   ^ 本 chirp 起始时刻前端状态 (含 idle 期间演化) 造成的**确定性线性响应**。
    #     可预测 (状态已知时), 不是噪声也不是干扰; 但把它当“目标回波”看就是幻觉。
    clip_residual: Optional[np.ndarray] = None
    #   ^ 满量程削顶后的**非线性残差** (削顶后输出 − 削顶前处理输出)。信息已毁,
    #     不得通过"线性分量相加"把它伪装成可恢复 (方案 §四.1)。
    label_schema: str = LABEL_SCHEMA_VERSION


@dataclass
class ChirpFrame:
    """受害者雷达一个 chirp 的观测量。"""
    session_chirp: int                  # 全局 chirp 序号
    frame_index: int                    # 帧序号 = session_chirp // frame_chirps
    chirp_in_frame: int
    t0: float                           # 本 chirp 起始时刻 (s, 会话绝对时间)
    observed: np.ndarray                # complex[N] 经 LPF (与可选削顶) 后的差拍数据
    agc_scale: float = 1.0             # 物理量 -> 本帧数据 的乘子 (乘 agc_scale 得归一化值)
    truth: TruthRecord = None
    adc_clipped_frac: float = 0.0


@dataclass
class InterferenceState:
    """感知层输出: 多维干扰状态 + 置信度 (技术方案第二节)。

    全部为在线可算量, 不含真值。
    """
    anomaly_db: float = 0.0             # 帧内能量异常分数的 p99 (dB), 相对稳健背景
    anomaly_trace: Optional[np.ndarray] = None   # complex/real[N] 逐点异常分数 (dB)
    inr_est_db: float = -80.0           # 估计干噪比 (dB), 只在有掩码时给值
    inr_conf: float = 0.0               # 上式的置信度 0~1
    noise_w: float = 0.0                # 背景功率估计 (W 口径, 归一化后)
    contaminated_frac: float = 0.0      # rho: 判定污染样本占比
    longest_gap: int = 0                # L_max: 最长连续缺口 (点)
    gap_edges: tuple = ()               # 缺口 (start, end) 列表
    edge_left: bool = False             # 缺口贴 chirp 开头
    edge_right: bool = False            # 缺口贴 chirp 结尾
    n_clean: int = 0                    # 干净样本数
    longest_clean: int = 0              # 最长连续干净段 (点)
    target_peak_margin_db: float = 0.0  # 最强目标峰相对本底的裕度 (dB)
    target_range_m: float = np.nan      # 最强峰对应距离 (m)
    target_conf: float = 0.0            # 目标可见度置信度 (含历史轨迹支持)
    localization_conf: float = 0.0      # 掩码可靠度 (检测器自报 + 两两一致性)
    par_db: float = 0.0                 # 峰均比 (dB)
    fine_used: bool = False             # 是否触发了按需精细感知
    extra: dict = field(default_factory=dict)


#: 检测上下文 schema 版本 (P0)。加字段或改 fit_exclude 判据必须升位, 旧结果不得按
#: 新标准解读 (docs/干扰检测与重建_分阶段修改方案_2026-10-04.md §2)。
DETECTION_SCHEMA_VERSION = 'detection-context-v1'


@dataclass
class DetectionResult:
    mask: np.ndarray                    # bool[N], True = 已确认污染, **唯一**授权重写输出的掩码
    confidence: float = 0.0             # 定位置信度 0~1
    name: str = ''
    cost_s: float = 0.0
    detail: dict = field(default_factory=dict)
    candidate_mask: Optional[np.ndarray] = None  # Evidence, not authority to rewrite.
    unknown_mask: Optional[np.ndarray] = None    # Not established clean or polluted.
    status: str = ''                            # clean/detected/uncertain/out_of_scope
    reason: str = ''
    # ---- 2026-10-04 P0: "允许重写" 与 "允许训练" 分开 (方案 §2) ----
    # 旧接口只有 mask: 重建器普遍拿 ~mask 当干净训练集, 而 ~mask 里还压着候选、未知区
    # 和滤波边界保护区。R4 实测: 只把训练掩码换成正确的, 缺口 NMSE 从 1.617 降到 0.0036。
    # fit_exclude 是"禁止进入拟合"的位置; 有效训练集是 ~fit_exclude, 不是 ~write_mask。
    fit_exclude_mask: Optional[np.ndarray] = None
    boundary_guard_mask: Optional[np.ndarray] = None  # 已确认污染两侧的保护环, 只禁止训练, 不授权重写
    startup_guard_mask: Optional[np.ndarray] = None   # 公开前端配置推出的启动无效区
    #: 二维时频证据 (D2/D3): freqs/starts/nperseg/noverlap/window/hop/excess_db/
    #: candidate_tf/structure_tf/energy_tf/trusted/config。数组不参与 JSON detail,
    #: 由 debugging.support 的检测上下文往返单独持久化。
    tf_evidence: dict = field(default_factory=dict)
    schema_version: str = DETECTION_SCHEMA_VERSION

    def __post_init__(self):
        self.mask = np.asarray(self.mask, dtype=bool)
        self.candidate_mask = self.mask.copy() if self.candidate_mask is None else np.asarray(self.candidate_mask, dtype=bool)
        self.unknown_mask = np.zeros_like(self.mask) if self.unknown_mask is None else np.asarray(self.unknown_mask, dtype=bool)
        if self.mask.ndim != 1 or self.candidate_mask.shape != self.mask.shape or self.unknown_mask.shape != self.mask.shape:
            raise ValueError('Detection masks must have the same 1D shape')
        self.boundary_guard_mask = (np.zeros_like(self.mask) if self.boundary_guard_mask is None
                                    else np.asarray(self.boundary_guard_mask, dtype=bool))
        self.startup_guard_mask = (np.zeros_like(self.mask) if self.startup_guard_mask is None
                                   else np.asarray(self.startup_guard_mask, dtype=bool))
        if self.boundary_guard_mask.shape != self.mask.shape or self.startup_guard_mask.shape != self.mask.shape:
            raise ValueError('Detection guard masks must have the same 1D shape')
        # 初始规则: fit_exclude 至少包含 write、候选、未知、边界保护与启动保护。显式传入的
        # fit_exclude 也**并上**保护区 —— 保护区是规则不是提示, 不能被调用方覆盖掉。
        base_exclude = self.mask | self.candidate_mask | self.unknown_mask
        if self.fit_exclude_mask is None:
            self.fit_exclude_mask = base_exclude
        else:
            self.fit_exclude_mask = np.asarray(self.fit_exclude_mask, dtype=bool)
            if self.fit_exclude_mask.shape != self.mask.shape:
                raise ValueError('fit_exclude_mask must have the same 1D shape')
            if np.any(base_exclude & ~self.fit_exclude_mask):
                raise ValueError('fit_exclude_mask 不能漏掉 write/candidate/unknown; '
                                 '允许训练的区域必须显式排除它们')
        self.fit_exclude_mask = (self.fit_exclude_mask | self.boundary_guard_mask
                                 | self.startup_guard_mask)
        if not isinstance(self.tf_evidence, dict):
            raise ValueError('tf_evidence must be a dict of arrays/metadata')
        if not self.status:
            self.status = 'detected' if self.mask.any() else 'uncertain'
        if self.status not in ('clean', 'detected', 'uncertain', 'out_of_scope'):
            raise ValueError('Invalid detection status')
        if self.status == 'clean' and (self.mask.any() or self.candidate_mask.any() or self.unknown_mask.any()):
            raise ValueError('Clean status conflicts with detection evidence')

    @property
    def write_mask(self) -> np.ndarray:
        """允许修改正式输出的位置。与 `mask` 是同一个数组, 单列出来是为了让
        "重写授权"与"训练排除"在代码里不可能被当成同一件事。"""
        return self.mask

    @property
    def training_mask(self) -> np.ndarray:
        """允许作为拟合观测的位置 = ~fit_exclude。别再把它叫"干净区": 它只是
        "没有被证据排除", 不等于已确认无污染。"""
        return ~self.fit_exclude_mask

    def to_arrays(self) -> dict:
        """检测上下文的数组部分 (NPZ 往返用); tf_evidence 递归展平。

        嵌套字典用 `键__子键` 展平 (D3 的 long/short 两套网格), 数组不塞进 JSON。
        """
        out = {'det_mask': self.mask, 'det_candidate_mask': self.candidate_mask,
               'det_unknown_mask': self.unknown_mask,
               'det_fit_exclude_mask': self.fit_exclude_mask,
               'det_boundary_guard_mask': self.boundary_guard_mask,
               'det_startup_guard_mask': self.startup_guard_mask}

        def add(prefix, d):
            for k, v in d.items():
                if isinstance(v, np.ndarray):
                    out[prefix + str(k)] = v
                elif isinstance(v, dict):
                    add(prefix + str(k) + '__', v)

        add('tf_', self.tf_evidence)
        return out

    def context_meta(self) -> dict:
        """检测上下文的标量元数据 (数组一律不进 JSON)。"""
        def meta(d):
            out = {}
            for k, v in d.items():
                if isinstance(v, np.ndarray):
                    continue
                if isinstance(v, dict):
                    out[k] = meta(v)
                elif isinstance(v, (np.floating, np.integer)):
                    out[k] = v.item()
                elif isinstance(v, (str, int, float, bool)) or v is None:
                    out[k] = v
                elif isinstance(v, (list, tuple)):
                    out[k] = [x.item() if isinstance(x, (np.floating, np.integer)) else x
                              for x in v if not isinstance(x, np.ndarray)]
            return out
        return {'name': self.name, 'status': self.status, 'reason': self.reason,
                'confidence': float(self.confidence), 'schema_version': self.schema_version,
                'tf_evidence_meta': meta(self.tf_evidence)}


#: `ReconstructionResult.status` 的封闭取值。加一个新状态就等于放宽一次口径,
#: 所以名单与它的判据写在一起, 不许重构器自创字符串。
#: 分成三档的理由: "没数据可用" (cold_start) 会随时间自己变好, "模型不适用"
#: (out_of_scope) 换个方法可能有救, "物理不可恢复" (unrecoverable) 换谁都没救。
#: 把这三件事合并成一个 failed 布尔, 就没法回答"这次失败该不该算进算法的能力"。
RECON_STATUSES = (
    'verified_reconstruction',   # 掩码内全部写入, 且拟合由**当前 chirp** 的可信观测支持并通过该方法自己的留出检验
    'history_prediction',        # 全部写入, 但当前 chirp 没贡献可信观测, 只由过去 chirp 外推 -> 不证明目标此刻仍在
    'open_loop_fill',            # 全部写入, 但该方法没有定义独立留出检验 (R1/R2/AC 的幅度校正) -> 不许称"已验证"
    'partial',                   # 只写入了一部分掩码内样本, 其余按统一输出层切除
    'cold_start',                # 历史/观测还没攒够, 属可用性问题, 不是算法质量问题
    'out_of_scope',              # 模型前提不成立: 定阶失败、拟合不通过、参考窗不足、无法自检
    'unrecoverable',             # 信息已经没了, 换谁都没救: ADC 削顶、前端启动盲区、
    #   不可辨识, 以及**置零策略**主动把掩码内样本删掉 (reason 区分是哪一种)。
    'no_op',                     # 空掩码或主动直通 -> **永远不得计入"重构成功"**
)

#: 结果 schema 版本。status 语义或字段集变化时要升这一位, 让旧结果不会被新评价误读。
#: v2 (P0): 增加 `validation` 证据块; `verified_reconstruction` 从"方法自称有留出检验"
#: 收紧为"该方法明确给出 pass + 成对改进为正" (方案 §3)。
RECON_SCHEMA_VERSION = 'recon-status-v2'

#: `ReconstructionResult.validation['validation_status']` 的封闭取值。
#:   pass         —— 外层连续块上成对改进为正且参考充分; 这是**唯一**允许 verified 的证据
#:   fail         —— 外层检验可做但成对改进不为正 (预测无收益); 不得写为已验证
#:   not_testable —— 参考不足/留出视界短于实际缺口/模型不可估; 不代表"数学上不可恢复"
VALIDATION_STATUSES = ('pass', 'fail', 'not_testable')


@dataclass
class ReconstructionResult:
    y: np.ndarray                       # complex[N], 掩码外必须与输入逐位相同
    filled_frac: float = 0.0            # 掩码内**被写入**的比例。`zero` 也写 (写 0),
    #   所以它是"没留成原观测"的比例, 不是"把信息恢复回来"的比例 (审计 2026-10-02)
    failed: bool = False                # 重构失败标志 (不能以"输出了数字"代表成功)
    name: str = ''
    cost_s: float = 0.0
    detail: dict = field(default_factory=dict)
    # ---- 2026-10-04 M0-A: "写满了" 与 "恢复了" 是两件事 ----------------------
    # 原缺陷: `zero` 在掩码内写 0, 于是 filled_frac=1.0 且 failed=False —— 置零和
    # 真正恢复在同一个字段上不可区分 (实测打印见 debugging/reconstruct_case.py:
    # "zero: ... write/API=1.000, failed/API=False")。status 就是把这两件事拆开。
    status: str = ''                    # RECON_STATUSES 之一; '' = 重构器未申报, 不得当作已恢复
    reason: str = ''                    # 机器可分组的原因码 (拒绝时必填)
    # 两张子掩码回答"**这帧输出凭什么**", 与 status 的"凭得够不够硬"是两个维度:
    #   verified_mask   当前 chirp 里确实有可信观测支撑这个填充 (够不够硬看 status)
    #   prediction_mask 当前 chirp 没贡献可信观测, 数值全部来自过去 chirp 的外推
    # 两者互斥。训练可信掩码另由重构器自己表达 (见 cross_chirp.observe 的 invalid)。
    verified_mask: Optional[np.ndarray] = None
    prediction_mask: Optional[np.ndarray] = None
    #: 2026-10-06 P0: 掩码内**未恢复**的样本 (保留原观测, 不置零)。"失败如实上报"
    #: 的载体: 后续评价必须能区分 已恢复 / 未解决污染, 不能把未解决当干净结果。
    unresolved_mask: Optional[np.ndarray] = None
    evidence_age_s: float = np.nan      # 最近一条**参与拟合**的可信观测距本 chirp 的时间 (s)
    prediction_span_s: float = np.nan   # 拟合窗口的时间基线 (s); 非外推为 0
    # ---- 2026-10-04 P0: 独立验证证据 (方案 §3) ----
    # 至少含: validation_status (pass/fail/not_testable), heldout_length, actual_gap_length,
    # baseline_error, candidate_error, paired_improvement, reference_count, model_order,
    # noise_profile, selection_protocol, context, regime。没有这一块 (或不是 pass) 时
    # 不许报 verified_reconstruction —— "留出误差有限"不再等于"恢复正确"。
    validation: dict = field(default_factory=dict)

    def __post_init__(self):
        self.y = np.asarray(self.y)
        n = self.y.size
        for f in ('verified_mask', 'prediction_mask'):
            v = getattr(self, f)
            v = np.zeros(n, dtype=bool) if v is None else np.asarray(v, dtype=bool)
            if v.shape != (n,):
                raise ValueError(f'{f} 必须是长度 {n} 的一维 bool 数组')
            setattr(self, f, v)
        # 一个样本不能同时"已验证"和"仅预测" —— 这两句话关于同一批样本互相矛盾。
        if np.any(self.verified_mask & self.prediction_mask):
            raise ValueError(f'{self.name}: verified_mask 与 prediction_mask 重叠, '
                             f'同一批样本不能既是已验证恢复又是历史外推')
        if self.status and self.status not in RECON_STATUSES:
            raise ValueError(f'{self.name}: 未知 status {self.status!r}, 只允许 {RECON_STATUSES}')
        vs = (self.validation or {}).get('validation_status')
        if vs is not None and vs not in VALIDATION_STATUSES:
            raise ValueError(f'{self.name}: 未知 validation_status {vs!r}, '
                             f'只允许 {VALIDATION_STATUSES}')
        # 只有已经提供验证证据块的方法才受"必须 pass"约束 (R3/R4 本批接入);
        # 尚未接入的方法 (R5/R6/跨 chirp) 保留它们自己的内部检验, 由后续批次迁移。
        if self.validation and self.status == 'verified_reconstruction' and vs != 'pass':
            raise ValueError(f'{self.name}: verified_reconstruction 必须带 '
                             f"validation_status='pass', 收到 {vs!r}; "
                             f'"留出误差有限"不等于通过验证')

    @property
    def is_verified(self) -> bool:
        """只有这一档可以对外说"该缺口内的信号被恢复出来了"。"""
        return self.status == 'verified_reconstruction'

    @property
    def is_history_prediction(self) -> bool:
        return self.status == 'history_prediction'

    @property
    def supports_target_presence(self) -> bool:
        """这一帧的输出能不能当作"目标此刻仍然存在"的证据。

        历史外推**不能**: 交叉审计 (2026-10-03 §3) 的小反例里, 目标已消失与目标仍在
        两种输入产生逐位相同的输出, 内部留出误差 2.2e-5, 因而任何"留出通过"都不足以
        证实目标延续 —— 当前 chirp 根本没被观测到。检测/跟踪侧读这个标志, 不读 failed。
        """
        return self.status in ('verified_reconstruction', 'open_loop_fill')


@dataclass
class Action:
    """一个动作 = 一对完整选择 (定位层级, 重构层级)。"""
    key: str
    detector: str                         # 'none' | 'd0' | 'd1' | 'd2' | 'd3'
    reconstructor: str                    # 'none' | 'r1' | 'r2' | 'r3' | 'r4' | 'r5'

    @property
    def is_bypass(self) -> bool:
        return self.detector == 'none' and self.reconstructor == 'none'


@dataclass
class ChirpOutcome:
    """闭环单 chirp 结果, 供在线反馈与离线评价共用。"""
    frame: ChirpFrame
    state: InterferenceState
    action: Action
    detection: Optional[DetectionResult]
    recon: Optional[ReconstructionResult]
    t_sense: float = 0.0
    t_detect: float = 0.0
    t_recon: float = 0.0
    t_total: float = 0.0
    deadline_hit: bool = False
    quality_score: float = 0.0
    detail: dict = field(default_factory=dict)
