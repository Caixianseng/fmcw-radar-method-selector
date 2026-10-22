# common/dsp.py
"""公共数字信号处理与掩码工具 (被 sensing/detection/reconstruction/evaluation 共用)。

设计约定:
  * 信号一律为复基差 (analytic) 一维数组, 长度 N = round(fs * T_chirp)。
  * 滤波器用 SOS + lfilter 状态在 chirp 内递推, 不再用整帧循环 FFT 低通
    (循环 FFT 会把 chirp 两端人为接起来, 掩码落在帧尾时会产生假环绕)。
    chirp 间不复用状态。这个决定的代价现在是被量出来并单独标成标签的, 不再是口头断言:
    本体制下 (fs=50 MHz, 通带 20 MHz, 3 阶椭圆) `filter_settle_samples` 在 1% 容差上
    得到 **17 点 = 340 ns**, 0.1% 容差 33 点 = 660 ns, 都远小于 25 µs 空闲 —— 所以上一
    句"模拟前端已重新稳定"在数字级成立。**注意它没有覆盖的部分**: 每 chirp 头 17 点
    不是稳态观测, 由 `TruthRecord.mask_startup_invalid` 单列; 而模拟抗混叠过渡带
    (lpf_cutoff~lpf_stop_edge) 泄漏进来的能量落在 RF 通带窗**之外**, 那部分由
    `TruthRecord.mask_if_support` 承载, 与上面的"空闲足够长"无关。连续模拟前端的
    idle/LO/ADC 状态归 M1, 这里只保证数字级口径自洽。
  * 门限类量统一写成"相对参考本底的 dB 裕量", 不假装是恒定虚警率的理论门限。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.signal import ellip, ellipord, sosfilt, windows

_STRUCT_CACHE: dict = {}
_SOS_CACHE: dict = {}


# ----------------------------------------------------------------- 基本量
def power(x: np.ndarray) -> np.ndarray:
    """瞬时功率 |x|^2, 与 x 同形状 (复数输入返回实数)。"""
    return (x.real ** 2 + x.imag ** 2).astype(np.float64, copy=False)


def from_db(x) -> np.ndarray:
    return 10.0 ** (np.asarray(x, dtype=float) / 10.0)


# ----------------------------------------------------------------- 滤波
def design_lpf(fs: float, cutoff: float, stop_edge: float,
               pass_ripple_db: float = 0.5, stop_atten_db: float = 40.0):
    """椭圆低通 SOS: 通带 cutoff, 阻带自 stop_edge 起衰减 stop_atten_db。

    fs=50 MHz、通带 20 MHz、阻带 23.5 MHz 时约 6~8 阶; Butterworth 在这个
    过渡带比值 (1.175) 下要 20 阶以上, 数值上不现实, 故用等纹波椭圆。
    """
    key = (fs, cutoff, stop_edge, pass_ripple_db, stop_atten_db)
    if key not in _SOS_CACHE:
        wp, ws = cutoff / (fs / 2.0), min(stop_edge / (fs / 2.0), 0.999)
        n, wn = ellipord(wp, ws, pass_ripple_db, stop_atten_db, analog=False)
        _SOS_CACHE[key] = ellip(n, pass_ripple_db, stop_atten_db, wn, output='sos')
    return _SOS_CACHE[key]


def apply_lpf(x: np.ndarray, sos) -> np.ndarray:
    """SOS 递推低通, 复数安全, chirp 内单次通过 (零状态启动)。"""
    return sosfilt(sos, x, axis=0)


def filter_settle_samples(sos, rel_tol: float = 0.01, probe: int = 4096) -> int:
    """零状态启动的数字滤波器多久进入稳态 (采样点数)。

    判据用**阶跃响应**而不是冲激响应: 3 阶椭圆在这里的冲激能量尾巴很长 (99% 能量落在
    约 210 点之后), 但"这一段输出能不能当稳态观测用"问的是幅度偏离稳态多少, 不是能量
    还剩多少。返回 k 表示第 k 点起偏离一直保持在 rel_tol 以内 (取最后一次越界之后,
    所以椭圆滤波器的振铃不会被"某一点刚好过零"骗过去)。
    """
    if not 0.0 < rel_tol < 1.0:
        raise ValueError('rel_tol 必须落在 (0,1) 内')
    u = sosfilt(np.asarray(sos, dtype=float), np.ones(int(probe)))
    ss = float(np.median(u[3 * probe // 4:]))
    if abs(ss) < 1e-30:
        return int(probe)                     # 直流增益为 0: 无稳态可言, 整段保守标记
    inside = np.abs(u - ss) <= rel_tol * abs(ss)
    out = np.flatnonzero(~inside)
    return int(out[-1] + 1) if out.size else 0


# ----------------------------------------------------------------- 时频
def stft(x: np.ndarray, nperseg: int, noverlap: int, detrend_mean: bool = True):
    """返回 (freqs[Hz], starts[采样点索引], Z[complex, nfreq x ntime])。

    只在信号内部开窗，不做周期延拓；末窗补到帧尾，返回实际索引。
    去均值在加窗前，频率单位cycles/sample（乘fs为Hz）。
    """
    x = np.asarray(x)
    step = max(1, nperseg - noverlap)
    n = x.size
    if n < nperseg:
        return np.empty(0), np.empty(0, dtype=int), np.empty((0, 0), dtype=complex)
    idx = np.arange(0, n-nperseg+1, step, dtype=int)
    if idx[-1] != n-nperseg:idx=np.r_[idx,n-nperseg]
    win = windows.hann(nperseg, sym=False)
    view=np.lib.stride_tricks.sliding_window_view(x,nperseg)[idx]
    # Standard STFT convention: detrend each data segment BEFORE windowing.
    if detrend_mean:view=view-view.mean(axis=1,keepdims=True)
    seg=view*win
    z = np.fft.fft(seg, n=nperseg, axis=1).T
    freqs = np.fft.fftfreq(nperseg)          # cycles/sample; 乘 fs 得 Hz
    return freqs, idx, z.astype(complex, copy=False)


# ----------------------------------------------------------------- 掩码
def segments(mask: np.ndarray):
    """bool 掩码 → [(start, end), ...], end 不含。"""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    e = np.diff(np.concatenate(([0], m.view(np.int8), [0])))
    return list(zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist()))


def longest_run(mask: np.ndarray) -> int:
    """最长连续 True 的长度 (点)。"""
    return max((b - a for a, b in segments(mask)), default=0)


def dilate(mask: np.ndarray, k: int) -> np.ndarray:
    """一维对称膨胀 k 点 (两侧各扩 k)。"""
    m = np.asarray(mask, dtype=bool)
    if k <= 0 or not m.any():
        return m.copy()
    if k not in _STRUCT_CACHE:
        _STRUCT_CACHE[k] = np.ones(2 * k + 1, dtype=bool)
    return binary_dilation(m, structure=_STRUCT_CACHE[k])


def point_iou_on(mask: np.ndarray, truth: np.ndarray) -> float:
    """点级 IoU = TP/(TP+FP+FN)。两者皆空时返回 nan 而不是 1。

    旧实现把"掩码与真值都为空"记 1.0, 干净帧一多就把均值抬到没有意义的程度;
    返回 nan 让上层必须显式只在有干扰的帧上统计。
    """
    m = np.asarray(mask, dtype=bool)
    t = np.asarray(truth, dtype=bool)
    tp = int(np.sum(m & t))
    fp = int(np.sum(m & ~t))
    fn = int(np.sum(~m & t))
    if tp + fp + fn == 0:
        return float('nan')
    return tp / (tp + fp + fn)


def diagonal_average(h: np.ndarray) -> np.ndarray:
    """Hankel 化 / 反对角平均: 把任意矩阵压回一维序列 (SSA 的重构步)。"""
    h = np.asarray(h)
    m, n = h.shape
    rows, cols = np.indices((m, n))
    k = rows + cols
    out = np.zeros(m + n - 1, dtype=h.dtype)
    cnt = np.zeros(out.size, dtype=np.int64)
    np.add.at(out, k.ravel(), h.ravel())
    np.add.at(cnt, k.ravel(), 1)
    np.divide(out, cnt, out=out, where=cnt > 0)
    return out


# ----------------------------------------------------------------- CFAR / 背景
def cfar_1d(level: np.ndarray, train: int, guard: int, margin_db: float,
            os_frac: float = 0.5) -> np.ndarray:
    """一维 OS-CFAR 型检测: 检测量 > 参考本底 * 10^(margin_db/10), 全向量化。

    level     一维检测量 (功率, 线性)
    train     每侧参考单元数 N (参考窗共 2N 个, 跳过中间 2*guard+1 个)
    guard     每侧保护单元数
    os_frac   有序参考单元取第 ceil(os_frac*N_valid) 小，N_valid不包括CUT/保护单元
    边缘仅用真实有效参考，不复制被检端点或环绕；dB裕量是经验门限，不是解析PFA。
    """
    v = np.asarray(level, dtype=float)
    n = v.size
    if n == 0 or train <= 0:
        return np.zeros(n, dtype=bool)
    if v.ndim != 1 or guard < 0 or not 0 < os_frac <= 1:
        raise ValueError('1D level, guard>=0 and os_frac in (0,1] required')
    offs=np.r_[np.arange(-train-guard,-guard),np.arange(guard+1,train+guard+1)]
    idx=np.arange(n)[:,None]+offs[None,:]
    valid=(idx>=0)&(idx<n);count=valid.sum(axis=1)
    cells=np.where(valid,v[np.clip(idx,0,n-1)],np.inf)
    rank=np.maximum(0,np.ceil(os_frac*count).astype(int)-1)
    ref=np.take_along_axis(np.sort(cells,axis=1),rank[:,None],axis=1)[:,0]
    return (count>0)&(v>ref*from_db(margin_db))



def mode_background(values: np.ndarray, bins: int = 64) -> float:
    """背景众数估计 (dB 域直方图峰值), 密集干扰下比均值稳健。

    对 dB 量做众数而不是对线性功率做均值: 干扰把右尾拉长, 均值会被抬走。
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return -120.0
    lo, hi = np.percentile(v, [1.0, 99.0])
    if hi - lo < 1e-9:
        return float(lo)
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))


def sliding_mean_power(x: np.ndarray, win: int) -> np.ndarray:
    """长度为 win 的**尾部**滑动平均功率 (前缀和差分, O(N)), 复数输入取 |x|^2。

    out[i] = mean(p[max(0,i-win+1) : i+1]) —— 头部不足一窗时按实际点数取均值。
    以前这里头部写的是 `sum(p[i:i+win])/(i+1)`: 前视窗的**和**除以后视窗的**点数**,
    量纲都不对 (i=0 时是整窗能量总和, 最多高估 win 倍 ≈ +20 dB @ win=100), 方向还与
    身体部分相反。它在 `simulation/receiver.py` 里构造**真值** `mask_significant`,
    后果是"污染刚进入的那 ~win 个点"被提前标成显著污染 —— 缺口左侧可用观测被夸大成
    "从第 0 点就开始"。钉住它的测试: tests/test_target_protection.py::LocalWindowHead。
    """
    p = power(x) if np.iscomplexobj(x) else np.asarray(x, dtype=float) ** 2
    if win <= 1:
        return p
    c = np.concatenate(([0.0], np.cumsum(p)))
    i = np.arange(p.size)
    lo = np.maximum(0, i - win + 1)
    return (c[i + 1] - c[lo]) / (i - lo + 1)


