"""Active traffic: one sparse four-lane open road, ego in lane 3.

Historical definitions live in simulation.archived.scenarios_legacy. Explicit
legacy names remain callable for stored experiments/tests, never active defaults.
Lane numbers in figures are 1..4; Python indices are 0..3.
"""
from fmcw_selector._core.configs.traffic import LaneConfig, ScenarioConfig, VehicleConfig

LANE_W=3.5
SESSION_S=40.
DT_TRAFFIC=.005

def four_lane(duration_s=SESSION_S, seed=0, **kw):
    if duration_s <= 0:
        raise ValueError('Positive duration required')
    length=max(2000.,500.+20.*duration_s+250.)
    x=(-5.25,-1.75,1.75,5.25)
    lanes=tuple(LaneConfig(
        name=f'lane{i+1}_'+('opposing' if i<2 else 'same_direction'),
        waypoints=((x[i],length),(x[i],0.)) if i<2 else ((x[i],0.),(x[i],length)),
        speed_limit_mps=22. if i<2 else (20. if i==2 else 28.),
        closed=False) for i in range(4))
    # Initial sparse ordinary traffic; subsequent vehicles enter only at endpoints.
    initial=((0,length-850.,22.,12.,True),(0,length-1450.,22.,12.,True),
             (1,length-1150.,21.,12.,True),(1,length-1750.,21.,12.,True),
             (3,200.,28.,12.,True),(3,400.,28.,12.,True),(3,680.,28.,12.,True))
    values=dict(name='four_lane',duration_s=float(duration_s),dt_traffic=DT_TRAFFIC,
        lanes=lanes,population_mode='flow',initial_counts=(0,0,0,0),
        arrival_lanes=(0,1,3),density_profile=((0.,0.),(float(duration_s),0.)),
        congestion_speed_factor=1.,min_headway_s=16.,max_headway_s=16.,
        ego_lane=2,ego_start_s=500.,ego_speed_mps=20.,
        extra_vehicles=initial,vehicle_cfg=VehicleConfig(has_radar_prob=.5),
        conflicts=(),weak_targets=(),occlusion_enabled=False)
    values.update(kw)
    return ScenarioConfig(**values)

SCENARIOS={'four_lane':four_lane}


def four_lane_continuous(duration_s=SESSION_S, seed=0, **kw):
    """四车道连续交通扩展版 (2026-10-06 用户新计划): 几何不变, 车流更连续/更杂。

    - 车头时距范围 6~22 s (原 16 s 固定) -> 进出场时机连续变化;
    - lane4 同向车速 16~32 m/s: 慢车 (16-19, 自车 20 接近/超越) 与快车 (26-32, 远离)
      同时存在, 相对运动随几何连续变化;
    - RCS 混合 (8~18) + **弱目标** (摩托/行人 0.5~1.2) 同窗共存;
    - 多雷达车 (has_radar_prob=.6): 每部雷达自带时钟/波形/相位演化 (assign_radar,
      波形参数按范围连续, 不逐 chirp 重抽), 同窗可同时有 2+ 干扰;
    - 不堵车/无红绿灯: 密度折线保持平坦 (与 four_lane 相同语义)。
    """
    if duration_s <= 0:
        raise ValueError('Positive duration required')
    base = four_lane(duration_s=duration_s, seed=seed, **kw)
    length = max(2000., 500. + 20. * duration_s + 250.)
    initial = ((0, length - 700., 22., 15., True), (0, length - 1100., 23., 9., False),
               (0, length - 1500., 21., 12., True), (0, length - 1900., 22., 18., False),
               (1, length - 900., 21., 8., True), (1, length - 1300., 20., 14., True),
               (1, length - 1700., 22., 11., False),
               (3, 150., 30., 12., True), (3, 320., 17., 10., True),
               (3, 520., 28., 9., True), (3, 760., 19., 13., True),
               (3, 980., 32., 8., False), (3, 1250., 26., 16., True))
    values = dict(extra_vehicles=initial, min_headway_s=6., max_headway_s=22.,
                  vehicle_cfg=VehicleConfig(has_radar_prob=.6),
                  weak_targets=((2, 260., 18., 0.8), (2, 900., 19., 1.2),
                                (3, 1500., 24., 0.5)))
    import dataclasses
    return dataclasses.replace(base, **values)


SCENARIOS['four_lane_continuous'] = four_lane_continuous


def four_lane_six_radars(duration_s=SESSION_S, seed=0, **kw):
    """固定六台雷达：一台自车、五台独立干扰源，四车道车辆持续移动。"""
    from dataclasses import replace
    base = four_lane(duration_s=duration_s, seed=seed, **kw)
    length = max(2500., 2000.+30.*duration_s)
    x = (-5.25,-1.75,1.75,5.25)
    lanes = tuple(LaneConfig(
        name=f'lane{i+1}_'+('opposing' if i<2 else 'same_direction'),
        waypoints=((x[i],length),(x[i],0.)) if i<2 else ((x[i],0.),(x[i],length)),
        speed_limit_mps=22. if i<2 else (20. if i==2 else 28.),
        closed=False) for i in range(4))
    # 三辆对向车先后驶近；两辆邻车道同向车产生持续变化的相对距离。
    # 全部车辆在40s内保持在道路上，无入口补车，雷达身份集合固定。
    initial = ((0,length-1000.,22.,12.,True),
               (0,length-1400.,22.,12.,True),
               (1,length-1850.,21.,12.,True),
               (3,380.,28.,12.,True),
               (3,550.,28.,12.,True))
    return replace(base, name='four_lane_six_radars', lanes=lanes,
                   population_mode='fixed', fixed_counts=(2,1,0,2),
                   initial_counts=(0,0,0,0), arrival_lanes=(),
                   extra_vehicles=initial, weak_targets=(),
                   vehicle_cfg=VehicleConfig(has_radar_prob=1.))


SCENARIOS['four_lane_six_radars'] = four_lane_six_radars


def _radar_population_streams(seed):
    """Count and subset streams independent of traffic, clock and noise draws."""
    import numbers
    import numpy as np
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError('A nonnegative integer session seed is required')
    count_seed, subset_seed = np.random.SeedSequence([int(seed), 0xC07]).spawn(2)
    return np.random.default_rng(count_seed), np.random.default_rng(subset_seed)


def interferer_count_for_seed(seed):
    """Uniform draw in 0..7, made once for a reproducible session."""
    count_rng, _ = _radar_population_streams(seed)
    return int(count_rng.integers(0, 8))


def four_lane_random_radars(duration_s=SESSION_S, seed=0, **kw):
    """Seven moving scatterers; one persistent random subset carries 0..7 radars.

    The ego radar is additional to this count. Turning off an interfering radar
    does not remove its vehicle or target echo. All seven vehicles remain on the
    open road over the default 40 s session; source clocks are assigned once by
    TrafficFlow, exactly as in four_lane_six_radars.
    """
    from dataclasses import replace
    base = four_lane_six_radars(duration_s=duration_s, seed=seed, **kw)
    length = base.lanes[2].waypoints[-1][1]
    pool = ((0, length-1000., 22., 12.),
            (0, length-1400., 22., 12.),
            (1, length-1850., 21., 12.),
            (3, 380., 28., 12.),
            (3, 550., 28., 12.),
            (1, length-1000., 21., 12.),
            (2, 800., 18., 12.))
    count_rng, subset_rng = _radar_population_streams(seed)
    count = int(count_rng.integers(0, 8))
    equipped = set(int(i) for i in subset_rng.choice(len(pool), size=count, replace=False))
    initial = tuple((*vehicle, i in equipped) for i, vehicle in enumerate(pool))
    return replace(base, name='four_lane_random_radars',
                   fixed_counts=(2, 2, 1, 2), extra_vehicles=initial,
                   vehicle_cfg=VehicleConfig(has_radar_prob=0.))


SCENARIOS['four_lane_random_radars'] = four_lane_random_radars

def make_scenario(name='four_lane', **kw):
    if name in SCENARIOS:
        return SCENARIOS[name](**kw)
    raise KeyError(f'Unknown scenario {name}; choose from {sorted(SCENARIOS)}')
