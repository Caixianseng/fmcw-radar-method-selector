# FMCW radar method selector

四车道 FMCW 雷达互干扰仿真与基于观测特征的 MLP 方法选择器。此仓库只保留论文当前主干实现，不含历史实验、调参工程、测试工程、论文修订档及大规模中间数据。

## 核心流程

```text
四车道交通与独立雷达时钟
  → 2 GHz 复包络 / Blackman / 传播与散射 / 干扰和噪声
  → IQ 混频 / 实际 FIR 低通 / 降采样至 50 MHz
  → 32 维观测特征
  → MLP 预测三个候选的负 SINR 与恢复风险
  → 按预测代价选择 MP、C-AR 或 JSS
  → 执行选中方法，输出恢复的复数 ADC
```

本文的中心是**依据当前受干扰观测的结构和可恢复性选择互补的恢复方法**。MP、C-AR、JSS 的适用条件不同；选择器不是一个直接端到端生成波形的网络。

## 安装与快速查看

建议 Python 3.11，优先使用 `requirements.txt` 中与冻结模型一致的版本。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
fmcw-selector demo --out outputs/demo
```

双击 `outputs/demo/report.html`：查看三个论文示例的四车道记录、实际选中方法、六联时频图、处理前后包络、距离谱和 SINR/NMSE。HTML 图片内嵌，可单独复制。三个输入分别展示 MLP 选择 MP、C-AR、JSS，均为真实保存的同条 ADC；未修改输入或为了美化另行归一化。

## 产生新信号与执行选择器

```bash
fmcw-selector simulate --seed 97028 --windows 0.01 20 39 --chirps 8 --out outputs/session
# 几何扰动场景
fmcw-selector simulate --seed 98000 --geometry --chirps 8 --out outputs/geometry
```

每会话固定六台干扰雷达，另有自车雷达；七辆其他车保留运动与目标散射身份。每会话以 1/3 概率在六台中随机指定至多一个近斜率源，扫频时间取 74、74.5、74.9、75.1、75.5、76 μs，均不等于自车 75 μs；其压力时钟仅在 0.01 s 初始化一次，以单程传播时延补偿后到达晚 1 μs。普通源仍使用 25/175 μs 扫频与独立持续时钟。该事件概率是仿真配置概率，不是实车近斜率发生率。

| 参数 | 设置 |
|---|---|
| 道路 / 载频 / 带宽 | 四车道 / 77 GHz / 300 MHz |
| 自车 sweep / 周期 | 75 μs / 100 μs |
| 波形采样率 / ADC | 2 GHz / 50 MHz |
| 每 chirp ADC 点数 | 3750，复数 I/Q |
| 接收滤波 | 实际 FIR，通带20 MHz，阻带边缘23.5 MHz |
| 干扰合成距离门限 | 1000 m，与混频后的低通通带不同 |
| 噪声 | `scaled / −40 dB`，人为低噪声对照 |

77 GHz 采用复包络等效建模，不以 2 GHz 直接采样 77 GHz 实载波。传播相位、时延、散射、独立时钟和滤波作用均保留。

## 直接处理自己的 ADC

```python
import numpy as np
from fmcw_selector import process

with np.load("examples/mp.npz") as z:
    import json
    x = z["observed"]
    public = json.loads(z["public_parameters"].item())
y, info = process(x, public)
print(info["method"], info["predicted_cost"] if "predicted_cost" in info else info["cost"])
```

`process` 只读取观测及公开接收参数，先选择，再执行一个方法。MP 显式失败时按当前实现回退到 C-AR。`process_all` 为离线比较才执行三个候选。干净目标、源标签和噪声真值不参与特征与推理。

## 候选与特征

* **MP**：复指数模型，利用留出的连续观测块比较阶数，再用可用观测重估极点与复幅度；可能漏掉弱谱线。
* **C-AR**：稳健慢 Chirplet 抵消后，执行公开包络补偿、验证约束的时频双向复数 AR。默认固定为 `robust_validated`。
* **JSS**：目标音调与慢变干扰的稳健联合拟合，支撑补充，保留固定10%稳健残差。单主导慢变源与可分性仍是其边界。

32维输入包括幅度/能量、频谱/时频结构、小 Hankel 子空间、非零曲率和 JSS 初始化支撑探针。前28维以观测结构为主，最后4维是方法相关的初始化探针，不能全部称为普适雷达特征。完整名称见 `src/fmcw_selector/data/metadata.json`。

冻结 MLP 使用32维输入、两个 ReLU 隐层及9个线性输出：三个负 SINR、三个大误差风险分数、三个弱分量失真风险分数。代价为 `预测负SINR + 2 × 大误差风险分数`；弱分量风险头参与训练，当前直接排序权重为0。这些分数未经概率校准，不应解释为方法选择概率。

## 离线训练

```bash
fmcw-selector dataset --seeds 25000:25191 --chirps 8 --out datasets/development.npz
fmcw-selector train --dataset datasets/development.npz --out outputs/my_selector.joblib
# 冻结后只用于独立评价，勿根据留出结果调参
fmcw-selector simulate --seed 99000 --model outputs/my_selector.joblib --out outputs/holdout
```

训练按会话隔离，排序后的前75%会话训练、后25%验证；标准化器只在训练样本上拟合。比较两种预声明 MLP 架构，验证平均 SINR 精确并列时按枚举顺序选择。此入口供主干重新训练使用；捆绑的论文模型来自既有开发流程，不会被覆盖。历史 C-AR 组合筛选及相对旧模型的准入检查属于已完成的开发过程，不包含在此运行工程中；详见 [复现说明](docs/reproduction.md)。

## 评价与输出

主指标是固定目标参考误差 SINR：`10log10(P_target / (P_output_minus_target + 1e-12 P_target))`。不作幅相对齐，误差包含残余干扰、噪声与目标失真；它不是对非线性输出作物理分量分离。线性 NMSE 与每 chirp SINR 同时报告，算术平均 SINR 与平均线性 NMSE 的排序可能不同。保留弱谱分量的复系数保真诊断，不把谱峰保留直接当作实车检测率。

生成的数据、报告和用户重训模型默认位于忽略目录。执行时会产生完整复数 ADC、恢复波形、特征、实际选择、原始九输出派生分数、诊断及评分。当前算法计算量不支持100 μs周期的实时时序承诺。

## 目录

```text
src/fmcw_selector/api.py       观测推理与三方法调度
src/fmcw_selector/scene.py     四车道与独立时钟接收链
src/fmcw_selector/training.py  会话隔离离线训练
src/fmcw_selector/report.py    波形、距离谱和时频 HTML
src/fmcw_selector/_core/      迁移的固定数值算法与仿真模块
src/fmcw_selector/data/       可移植冻结模型与元数据
examples/                     三条实际论文 ADC 示例
docs/                         复现边界、代码来源与核对记录
```

没有捆绑未使用候选、MCA–SALSA、独立 Chirplet 动作、测试工程、第三方论文 PDF、论文 DOCX、旧模型或全部留出波形。`_core/anti_interference/methods.py` 保留现有检测器和 STFT 数值辅助函数的源码，部分辅助实现与早期 CFAR 共用，但不注册为选择器候选。

来源与论文实现关系见 [方法来源说明](docs/method_sources.md)。此私有仓库未指定开源许可证。

发布时数值配对核对见 [核对记录](docs/verification.json)，当前完整留出队列的必要结果摘要见 [论文结果摘要](docs/paper_results.json)。三条演示样例不用于估计总体平均性能。
