# configs/budget.py
"""时延与计算预算配置。

实测耗时是 PC 上 Python 的数, accelerator_time_scale 把它等比换算到目标硬件口径,
这样调度逻辑能在 PC 上验证, 而不会把"75 us chirp 时长"错当成任意实现的端到端截止期。
计算耗时 / 缓冲时延 / 排队时延三者在 evaluation/latency.py 里分开统计。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BudgetConfig:
    accelerator_time_scale: float = 1.0    # 目标硬件耗时 = 实测 * scale
    budget_per_chirp_s: float = 2.0e-3     # 平均计算预算 (目标硬件口径, 每秒)
    deadline_factor: float = 4.0           # 单次硬截止 = factor * chirp_period (目标口径)
    queue_gain: float = 1.0                # 虚拟队列积压 -> 时间惩罚的增益
    lambda_max: float = 5.0
    # warmup_chirps 已删: 全仓无人读取, 真正生效的是下面的 warmup_executions。
    explore_prob: float = 0.0              # 低频探索概率 (开启时其耗时照常计入预算)
    # p95_over_budget_is_fail 已删: "P95 超预算即不可用"是无条件实现的 (control/budget.py::fits), 这个开关从来没有被读过。
    # ---- 冷启动与"锁死" (评审 2026-09-29 第 4/12 条) ------------------------------
    # 一个部件的**头几次**执行含 import / 分页 / 缓存未热的固定开销 (实测某次会话第一帧的
    # bypass 就要 38 ms, 稳态只有 0.14 ms)。旧写法把这些样本一起喂进分位数, 于是"起步阶段
    # 偶发几次超时"就把那个动作永久挡在门外, 而门外的动作再也拿不到新样本证明它已恢复 ——
    # 三次独立重跑在 2 ms 档给出 3.19 / 3.14 / 1.70 dB 的差别走的正是这条路径。
    #   * warmup_executions: 每个部件前 N 次执行只记账、不进分位数;
    #   * reprobe_prob: 以这个概率把"当前预测超预算"的动作再试一次 (代价照计), 让被排除的
    #     动作仍能拿到新样本。0 = 关 (与旧行为一致)。
    warmup_executions: int = 3
    reprobe_prob: float = 0.0
    reprobe_max_factor: float = 4.0        # 只重探"预测耗时 <= factor × 预算"的动作
