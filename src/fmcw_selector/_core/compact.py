"""Frozen, compact 32-feature FMCW selector; four unchanged recovery methods.

The first 28 descriptors are observation statistics and FMCW structure probes.
The last four are explicitly JSS initialization/support probes. They stop
before joint fitting or recovered output synthesis and are not universal
radar features. No full 56-feature extraction is executed by this module.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import numpy as np
import joblib
from fmcw_selector._core.anti_interference.features import (
    extract_features as _base_features, FEATURE_NAMES as BASE_NAMES,
)
from fmcw_selector._core.anti_interference.methods import _stft
from fmcw_selector._core.features_enhanced import multiple_ridges, EXTRA_FEATURE_NAMES as ENHANCED_NAMES
from fmcw_selector._core.features_subspace import extract_subspace_only, EXTRA_FEATURE_NAMES as SUBSPACE_NAMES
from fmcw_selector._core.features_near_slope import near_slope_descriptors, EXTRA_FEATURE_NAMES as CURVATURE_NAMES
from fmcw_selector._core.features_specialist_support import specialist_support_descriptors, EXTRA_FEATURE_NAMES as SUPPORT_NAMES

ROOT = Path(__file__).resolve().parent
VERSION = 'four_method_jss_v2_compact32_negative_sinr_v1'
COLUMNS_1BASED = (3,4,6,7,9,10,12,13,16,18,23,25,26,27,28,29,30,
                  42,43,44,45,46,47,48,49,50,51,52,53,54,55,56)
BASE_COLUMNS_1BASED = COLUMNS_1BASED[:17]
FEATURE_NAMES = (tuple(BASE_NAMES[i-1] for i in BASE_COLUMNS_1BASED)
                 + tuple(ENHANCED_NAMES[-3:]) + tuple(SUBSPACE_NAMES)
                 + tuple(CURVATURE_NAMES) + tuple(SUPPORT_NAMES))


def feature_names():
    return FEATURE_NAMES


def extract(observed, public):
    """Compute only the selected groups from observed complex ADC/public setup.

    Base extraction is reused for exact historical definitions; its omitted
    scalar statistics are inexpensive. Removed AR4 probes are never calculated;
    MP initial-row-count geometry is not returned or used as a model input.
    The Hankel routine still uses the existing observation detector and known
    public window, as documented.
    """
    x = np.asarray(observed, np.complex128)
    if x.ndim != 1 or len(x) < 32 or not np.isfinite(x).all():
        raise ValueError('Finite complex one-dimensional ADC, at least32 samples, required')
    fs = float(public['sample_rate_hz'])
    band = float(public['max_beat_hz'])
    base = _base_features(x, fs, band)[np.array(BASE_COLUMNS_1BASED)-1]
    scale = float(np.max(np.abs(x)))
    normalized = x/scale if scale else x.copy()
    _, _, spectrum, _ = _stft(normalized, fs)
    ridge = np.asarray(multiple_ridges(normalized, spectrum, fs, band)[:3])
    subspace = extract_subspace_only(x, public)
    curvature = near_slope_descriptors(x, public)[0]
    support = specialist_support_descriptors(x, public)[0]
    result = np.concatenate((base, ridge, subspace, curvature, support))
    if result.shape != (32,) or not np.isfinite(result).all():
        raise ValueError('Compact32 feature contract violated')
    return result

