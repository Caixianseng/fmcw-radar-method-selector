# 数值模块来源与论文关系

此仓库使用原项目的 Python 实现，不包含第三方论文 PDF 或作者仓库快照。

* MP依据 Wang、Ding、Yarovoy 的 *Matrix-Pencil Approach-Based Interference Mitigation for FMCW Radar Systems*（IEEE TMTT，2021）复指数/矩阵铅笔重建思路；连续观测块验证、阶数比较及失败保护为本项目工程扩展，不能称为论文逐行复现。
* Chirplet 部分参考 Correas-Serrano 与 González-Huici 的 *Sparse Reconstruction of Chirplets for Automotive FMCW Radar Interference Mitigation*（ICMIM，2019，DOI: 10.1109/ICMIM.2019.8726758）。本项目使用连续慢曲率搜索及公开窗/FIR模型，并非原文矩形二参数字典的原样复现。
* C-AR 将观测拟合的 Chirplet 抵消与时频双向复数 AR结合，含公开包络补偿与验证预测；这是当前工程组合。
* JSS 为本项目的确定性稳健联合拟合实现；不声称复现 VEM/SBL 或 MCA–SALSA。模型假设和可分性限制见源码及 README。
* SINR沿用项目中明确记录的 Wang (2022) *CFAR-Based Interference Mitigation for FMCW Automotive Radar Systems* 式(9)的固定参考误差定义，具体公式、数值保护与解释均保存在 `_core/sinr_metrics.py`。

NumPy、SciPy、scikit-learn、joblib、threadpoolctl、Matplotlib 通过各自安装包提供，其版权和许可证随安装包保留。本项目没有重新分发其库源码。未替用户选定代码开源许可；仓库保持私有。
