"""Independent-clock traffic generation using the paper's high-rate receiver."""
from dataclasses import asdict
from copy import deepcopy
import numpy as np


def generate(seed=97028, windows=(.01,20.,39.), chirps=8, geometry=False):
    from ._core.mixed_scene import build_session, diagnostic_for_frame, apply_vehicle_radar_overrides
    from ._core.simulation.session import Run
    from ._core.configs.radar import C
    session=build_session(int(seed))
    if geometry:
        rng=np.random.default_rng(np.random.SeedSequence([int(seed),0x47454F]))
        for vehicle in session.flow.vehicles:
            if vehicle.is_ego:continue
            vehicle.s+=float(rng.uniform(-60,60))
            vehicle.v=float(np.clip(vehicle.v+rng.uniform(-2,2),12,30))
            vehicle.rcs*=float(2**rng.uniform(-1,1))
        special=session.mixed_metadata['special_vehicle_id']
        if special is not None:
            ego,states=session.flow.snapshot(.01)
            state=next(s for s in states if s.vid==special)
            distance=float(np.linalg.norm(state.pos-ego.pos))
            apply_vehicle_radar_overrides(session,{str(special):dict(t_start=float((.01+1e-6-distance/C)%100e-6))})
    radar=session.cfg.radar
    public=dict(sample_rate_hz=float(radar.sample_rate),max_beat_hz=float(radar.operating_max_beat),
                chirp_duration_s=float(radar.chirp_duration),chirp_slope_hz_per_s=float(radar.slope),
                simulation_sample_rate_hz=float(session.receiver.sim_fs),transmit_bandwidth_hz=float(radar.bandwidth),
                lpf_cutoff_hz=20e6,lpf_stop_edge_hz=23.5e6,lpf_stop_atten_db=40.)
    records=[];arrays={k:[] for k in ('observed','clean','interference','noise')}
    for wi,start in enumerate(windows):
        for ci,frame in enumerate(session.run(Run(f'w{wi}',float(start),int(chirps)))):
            values=dict(observed=frame.observed,clean=frame.truth.clean,
                        interference=frame.truth.interf,noise=frame.truth.noise)
            for key,value in values.items():arrays[key].append(np.array(value,copy=True))
            info=deepcopy(diagnostic_for_frame(session,frame))
            records.append(dict(seed=int(seed),window_s=float(start),chirp=ci,
                time_s=float(frame.t0),session_chirp=int(frame.session_chirp),
                configured_extreme_event=bool(session.mixed_metadata['configured_extreme_event']),
                target_paths=len(frame.truth.targets),scene=info))
    return {k:np.stack(v) for k,v in arrays.items()},public,records
