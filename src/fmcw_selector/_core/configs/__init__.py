# configs/__init__.py
"""系统配置聚合入口。

用法:
    from fmcw_selector._core.configs import SystemConfig
    cfg = SystemConfig()                 # 全默认
    cfg = SystemConfig(budget=BudgetConfig(accelerator_time_scale=0.01))
    cfg.radar.n_samples                  # 3750 点/chirp
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fmcw_selector._core.configs.algorithms import ActionConfig, DetectionConfig, ReconstructionConfig, SensingConfig
from fmcw_selector._core.configs.budget import BudgetConfig
from fmcw_selector._core.configs.radar import RadarConfig
from fmcw_selector._core.configs.traffic import ScenarioConfig

__all__ = ['RadarConfig', 'ScenarioConfig', 'SensingConfig', 'DetectionConfig',
           'ReconstructionConfig', 'ActionConfig', 'BudgetConfig', 'SystemConfig']


@dataclass
class SystemConfig:
    radar: RadarConfig = field(default_factory=RadarConfig)
    sensing: SensingConfig = field(default_factory=SensingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    @property
    def fs(self) -> float:
        return self.radar.sample_rate

    @property
    def n_samples(self) -> int:
        return self.radar.n_samples
