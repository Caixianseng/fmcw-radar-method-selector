"""Six-radar scenes with a one-time configured-probability near-slope event around the immutable receiver snapshot.

Only this adapter changes scene identities; the receiver and mitigation methods
remain the archived files. Ground-truth diagnostics below are offline only.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from functools import partial
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
from fmcw_selector._core.configs import SystemConfig
from fmcw_selector._core.configs.radar import C
from fmcw_selector._core.inspect_receiver import InspectReceiver
from fmcw_selector._core.simulation.propagation import aspect
from fmcw_selector._core.simulation.scenarios import four_lane_random_radars
from fmcw_selector._core.simulation.session import TrafficSession

SCENE_VERSION = 'six_active_radars_bernoulli_probability_probe_persistent_clock_v1'
VEHICLE_IDS = tuple(range(1001, 1008))
ALLOWED_SWEEP_US = (74., 74.5, 74.9, 75.1, 75.5, 76.)


def load_settings():
    return json.loads((ROOT / 'mixed_settings.json').read_text(encoding='utf-8'))


def _apply_config(config, overrides):
    fields = asdict(config)
    for key, value in overrides.items():
        if key not in fields:
            raise ValueError(f'Unknown config field: {key}')
        setattr(config, key, tuple(value) if isinstance(fields[key], tuple) else value)


def apply_vehicle_radar_overrides(session, overrides):
    """Apply requested persistent waveform/clock fields by stable vehicle ID.

    This entrance runs once after the entire latent catalog has been drawn.
    It does not draw random numbers or change traffic/RCS/specular phases.
    Values use ChirpTimeline field names and SI units. It deliberately rejects
    changes to IDs, power, carrier, or framed timing in this experiment.
    """
    permitted = {'bandwidth', 'sweep_duration', 'slope_sign', 'chirp_period',
                 't_start', 'drift_ppm'}
    lookup = {v.vid: v for v in session.flow.vehicles if not v.is_ego}
    for vid_key, fields in overrides.items():
        vid = int(vid_key)
        if vid not in lookup or lookup[vid].radar is None:
            raise ValueError(f'No latent radar assigned to V{vid}')
        if set(fields) - permitted:
            raise ValueError(f'Unsupported override fields: {set(fields) - permitted}')
        timeline = replace(lookup[vid].radar, **fields)
        if not all(np.isfinite(float(v)) for v in fields.values()):
            raise ValueError('Non-finite vehicle radar override')
        if (timeline.bandwidth <= 0 or timeline.sweep_duration <= 0
                or timeline.sweep_duration >= timeline.chirp_period
                or timeline.slope_sign not in (-1., 1.) or timeline.t_start < 0):
            raise ValueError('Invalid persistent waveform/clock override')
        lookup[vid].radar = timeline


def draw_scene_assignment(seed, force_event=None, force_special_id=None, force_sweep_us=None):
    """Draw the active subset and event once in independent random namespaces.

    This is independent of traffic/property/noise RNGs and is cheap enough for
    distribution audits. The event means a configured near-slope radar, not a
    promise that the 1000 m propagation gate allows its signal at every window.
    Force arguments are explicit validation/gallery interventions only.
    """
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('A nonnegative integer seed is required')
    settings = load_settings()
    probability = float(settings['extreme_event_probability'])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('Scene event probability must be finite in [0, 1]')
    active_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x36414354]))
    active_ids = sorted(int(v) for v in active_rng.permutation(VEHICLE_IDS)[:6])
    event_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x4D495845]))
    event_uniform = float(event_rng.random())
    sampled_event = bool(event_uniform < probability)
    # Draw candidates even in ordinary sessions: paired validation never shifts RNGs.
    candidate_id = int(event_rng.choice(active_ids))
    candidate_sweep = float(event_rng.choice(ALLOWED_SWEEP_US))
    if force_event is not None and not isinstance(force_event, (bool, np.bool_)):
        raise ValueError('force_event must be bool or None')
    event = sampled_event if force_event is None else bool(force_event)
    special_id = candidate_id if event else None
    sweep_us = candidate_sweep if event else None
    if force_special_id is not None:
        if not event or int(force_special_id) != force_special_id or int(force_special_id) not in active_ids:
            raise ValueError('Forced special identity must be one of the six active radars in an event session')
        special_id = int(force_special_id)
    if force_sweep_us is not None:
        value = float(force_sweep_us)
        if not event or value not in ALLOWED_SWEEP_US or value == 75.:
            raise ValueError('Forced sweep requires an event and one of six allowed unequal-to-75 us sweeps')
        sweep_us = value
    return dict(seed=int(seed), event_probability=probability,
        event_uniform=event_uniform, sampled_event=sampled_event,
        configured_extreme_event=event, configured_event=event,
        active_vehicle_ids=active_ids, special_vehicle_id=special_id,
        sweep_us=sweep_us, total_count=6,
        assignment_intervention=bool(force_event is not None or force_special_id is not None or force_sweep_us is not None))


def build_session(seed, force_event=None, force_special_id=None, force_sweep_us=None):
    """Build seven scatterers, enable six radars, optionally replace exactly one.

    All radar identities and target phases are allocated before the independent
    subset/event choice is applied. Every clock is initialized once; pressure
    alignment of the special source is conditioned at 0.01 s only. Its future
    arrival evolves with vehicle motion, and ordinary clocks are left untouched.
    """
    assignment = draw_scene_assignment(seed, force_event, force_special_id, force_sweep_us)
    settings = load_settings()
    baseline = json.loads((ROOT / 'baseline_settings.json').read_text(encoding='utf-8'))
    special = settings['special_interferer']
    reference = float(special['reference_time_s'])
    scen = four_lane_random_radars(duration_s=float(baseline.get('duration_s', 40.)), seed=int(seed))
    scen = replace(scen, name='four_lane_six_radars_probability_probe',
                   extra_vehicles=tuple((*v[:4], True) for v in scen.extra_vehicles))
    _apply_config(scen.radar_cfg, baseline.get('interferer_overrides', {}))
    cfg = SystemConfig()
    _apply_config(cfg.radar, baseline.get('radar_overrides', {}))
    _apply_config(cfg.radar, {'noise_mode': settings['noise_mode'],
                            'noise_scale_db': float(settings['noise_scale_db']),
                            'interference_max_range_m': float(settings['interference_max_range_m'])})
    generation = dict(baseline.get('signal_generation', {}))
    generation['noise_seed'] = int(seed)
    generation['simulation_sample_rate_hz'] = 2e9
    session = TrafficSession(scen, cfg, int(seed),
                            receiver_factory=partial(InspectReceiver, generation=generation))
    all_vehicles = {v.vid: v for v in session.flow.vehicles if not v.is_ego}
    if sorted(all_vehicles) != list(VEHICLE_IDS):
        raise AssertionError('Expected exactly seven stable moving scatterer identities')
    latent_catalog = {str(vid): asdict(v.radar) for vid, v in sorted(all_vehicles.items())}
    vehicle_phases = {str(vid): float(v.specular_phase) for vid, v in sorted(all_vehicles.items())}
    ego, states = session.flow.snapshot(reference)
    special_id = assignment['special_vehicle_id']
    overrides = {}
    reference_range = reference_delay = None
    if special_id is not None:
        special_state = next(st for st in states if st.vid == special_id)
        reference_range = float(np.linalg.norm(special_state.pos - ego.pos))
        reference_delay = reference_range / C
        duration = assignment['sweep_us'] * 1e-6
        period = float(special['pressure_period_s'])
        arrival = reference + float(special['pressure_arrival_offset_s'])
        epoch = float((arrival - reference_delay) % period)
        overrides[str(special_id)] = dict(sweep_duration=duration,
            bandwidth=float(special['bandwidth_hz']), slope_sign=float(special['slope_sign']),
            chirp_period=period, t_start=epoch,
            drift_ppm=float(special['pressure_drift_ppm']))
        apply_vehicle_radar_overrides(session, overrides)
    configured_catalog = {str(vid): asdict(v.radar) for vid, v in sorted(all_vehicles.items())}
    active_ids = assignment['active_vehicle_ids']
    for vid, vehicle in all_vehicles.items():
        if vid not in active_ids:
            vehicle.radar = None
    session.flow._radar_vids = set(active_ids)
    session.t_cursor = reference
    metadata = dict(assignment,
        scene_version=SCENE_VERSION, profile='pressure' if special_id is not None else 'ordinary',
        reference_time_s=reference, reference_range_m=reference_range,
        reference_one_way_delay_s=reference_delay,
        pressure_arrival_offset_us=float(special['pressure_arrival_offset_s'])*1e6 if special_id is not None else None,
        reference_range_gate_allows=bool(cfg.radar.interference_range_allows(reference_range)) if special_id is not None else None,
        ordinary_activation_order=[vid for vid in active_ids if vid != special_id],
        ordinary_nested_order=[vid for vid in active_ids if vid != special_id],
        latent_catalog=latent_catalog, configured_catalog=configured_catalog,
        vehicle_specular_phases=vehicle_phases, vehicle_radar_overrides=overrides,
        clock_initialization_count=1, generation=generation,
        conditioning_note=f'Session Bernoulli({assignment["event_probability"]:.17g}) configuration event; if selected, one uniformly chosen active source has a one-time reference-arrival-conditioned pressure clock. Actual propagation and filtered interference remain separate diagnostics.')
    session.mixed_metadata = metadata
    # Compatibility with archived metrics/export helpers; no old experiment file is changed.
    session.near_parallel_metadata = metadata
    return session


def _clip_affine_band(lo, hi, intercept, slope, cutoff):
    if hi <= lo:
        return None
    if abs(slope) < 1e-18:
        return (lo, hi) if abs(intercept) <= cutoff else None
    roots = sorted(((-cutoff-intercept)/slope, (cutoff-intercept)/slope))
    left, right = max(lo, roots[0]), min(hi, roots[1])
    return (left, right) if right > left else None


def source_timing(session, t0, ego, st):
    """Independent closed-form overlap/passband calculation, no ideal filtering."""
    tl, rc = st.radar, session.cfg.radar
    if tl is None:
        raise ValueError('Timing diagnostics require an enabled source')
    r, closing, _, _ = aspect(ego, st)
    delay, fd = r / C, closing / rc.wavelength
    end = float(t0) + rc.chirp_duration
    first = max(0, int(tl.chirp_index(float(t0)-delay-tl.sweep_duration))-1)
    last = max(first, int(tl.chirp_index(end-delay))+1)
    overlaps, intervals, sweeps = [], [], []
    for index in range(first, last+1):
        tx_start = float(tl.chirp_start(index))
        d = tx_start + delay - float(t0)
        lo, hi = max(0., d), min(rc.chirp_duration, d+tl.sweep_duration)
        if hi <= lo:
            continue
        intercept = (rc.fc - rc.bandwidth/2) - tl.f0 + tl.slope*d - fd
        slope = rc.slope - tl.slope
        interval = _clip_affine_band(lo, hi, intercept, slope, rc.lpf_cutoff)
        overlaps.append([lo, hi])
        if interval is not None:
            intervals.append(list(interval))
        sweeps.append(dict(source_chirp_index=index, tx_start_s=tx_start,
            arrival_offset_us=d*1e6, overlap_us=[lo*1e6, hi*1e6],
            intercept_hz=intercept, beat_slope_hz_s=slope,
            beat_at_overlap_edges_hz=[intercept+slope*lo, intercept+slope*hi],
            passband_interval_us=None if interval is None else [v*1e6 for v in interval]))
    # Keep a future/previous-arrival descriptor even when no sweep overlaps.
    at_start = int(tl.chirp_index(float(t0)-delay))
    next_index = max(0, tl.first_index_at_or_after(float(t0)-delay))
    return dict(vehicle_id=int(st.vid), range_m=float(r), closing_mps=float(closing),
        one_way_delay_s=float(delay), doppler_hz=float(fd), waveform=asdict(tl),
        actual_period_s=float(tl.period), idle_s=float(tl.chirp_period-tl.sweep_duration),
        source_index_at_victim_start=at_start,
        source_arrival_at_victim_start_us=float((tl.chirp_start(at_start)+delay-float(t0))*1e6),
        next_arrival_offset_us=float((tl.chirp_start(next_index)+delay-float(t0))*1e6),
        arrival_offset_us=float(sweeps[0]['arrival_offset_us']) if sweeps else float((tl.chirp_start(next_index)+delay-float(t0))*1e6),
        acquisition_overlap_fraction=float(sum(b-a for a,b in overlaps)/rc.chirp_duration),
        theoretical_passband_fraction=float(sum(b-a for a,b in intervals)/rc.chirp_duration),
        theoretical_passband_intervals_us=[[a*1e6,b*1e6] for a,b in intervals],
        sweeps=sweeps, range_gate_allows=bool(rc.interference_range_allows(r)))


def _state_dict(st):
    return dict(vehicle_id=int(st.vid), lane=int(st.lane)+1,
                x_m=float(st.pos[0]), y_m=float(st.pos[1]),
                vx_mps=float(st.vel[0]), vy_mps=float(st.vel[1]),
                rcs_m2=float(st.rcs), is_ego=bool(st.is_ego),
                specular_phase_rad=float(st.specular_phase),
                radar=None if st.radar is None else asdict(st.radar))


def diagnostic_for_frame(session, frame, validate_waveform=False):
    """Offline timing, source powers and geometric truth for the current frame.

    Must be called before advancing to another frame: InspectReceiver stages are
    mutable current-frame buffers. Returns only JSON-safe Python scalar/list data.
    Set validate_waveform for a phase-increment test against actual mixed samples.
    """
    rc, receiver = session.cfg.radar, session.receiver
    ego, states = session.flow.snapshot(frame.t0)
    ids = list(receiver.components['interferer']['ids'])
    lookup = {vid: j for j, vid in enumerate(ids)}
    source_records = []
    stage = receiver.stages
    u = np.asarray(stage['time_s'])
    for st in states:
        if st.radar is None:
            continue
        row = source_timing(session, frame.t0, ego, st)
        if st.vid in lookup:
            j, tl = lookup[st.vid], st.radar
            mixed = stage['interferer_mixed'][j]
            filtered = receiver.components['interferer']['filtered'][j]
            pre = float(np.mean(np.abs(mixed)**2))
            post = float(np.mean(np.abs(filtered)**2))
            emitted = frame.t0 + u - row['one_way_delay_s']
            starts = tl.chirp_start(tl.chirp_index(emitted))
            arrival_offsets = starts + row['one_way_delay_s'] - frame.t0
            analytic = ((rc.fc-rc.bandwidth/2)-tl.f0
                        + tl.slope*arrival_offsets + (rc.slope-tl.slope)*u
                        - row['doppler_hz'])
            row.update(mixed_before_lpf_power_w=pre, adc_power_w=post,
                       fir_power_change_db=float(10*np.log10(post/pre)) if pre > 0 and post > 0 else None,
                       analytic_vs_generated_beat_max_error_hz=float(np.max(np.abs(analytic-stage['interferer_beat_hz'][j]))),
                       sampled_theoretical_passband_fraction=float(frame.truth.interferers[
                           next(i for i, x in enumerate(frame.truth.interferers) if x.vehicle_id==st.vid)].duty))
            if validate_waveform:
                actual = np.angle(mixed[1:]*np.conj(mixed[:-1]))*receiver.sim_fs/(2*np.pi)
                expected = .5*(analytic[1:]+analytic[:-1])
                active = stage['interferer_rx_active'][j]
                valid = (active[1:] & active[:-1]
                         & (tl.chirp_index(emitted[1:]) == tl.chirp_index(emitted[:-1]))
                         & (np.abs(mixed[1:]) > np.max(np.abs(mixed))*1e-4)
                         & (np.abs(mixed[:-1]) > np.max(np.abs(mixed))*1e-4))
                error = actual[valid]-expected[valid]
                error = (error+receiver.sim_fs/2)%receiver.sim_fs-receiver.sim_fs/2
                row['mixed_phase_increment_max_error_hz'] = float(np.max(np.abs(error))) if error.size else None
                row['mixed_phase_increment_rmse_hz'] = float(np.sqrt(np.mean(error**2))) if error.size else None
        else:
            row.update(mixed_before_lpf_power_w=0., adc_power_w=0., fir_power_change_db=None,
                       analytic_vs_generated_beat_max_error_hz=None,
                       sampled_theoretical_passband_fraction=0.)
        source_records.append(row)
    special_id = session.mixed_metadata['special_vehicle_id']
    special = next((r for r in source_records if r['vehicle_id']==special_id), None)
    configured_event = bool(session.mixed_metadata['configured_extreme_event'])
    return dict(session_chirp=int(frame.session_chirp), t0_s=float(frame.t0),
                ego=_state_dict(ego), vehicles=[_state_dict(st) for st in states],
                sources=source_records, interference_sources=source_records, special=special,
                configured_extreme_event=configured_event, configured_event=configured_event,
                special_vehicle_id=special_id, special_range_gate_allows=bool(special['range_gate_allows']) if special is not None else False,
                special_theoretical_passband_fraction=float(special['theoretical_passband_fraction']) if special is not None else 0.,
                special_adc_power_w=float(special['adc_power_w']) if special is not None else 0.,
                radar_count=len(source_records), vehicle_count=len(states),
                target_paths=len(frame.truth.targets),
                adc_sum_max_abs_error=float(np.max(np.abs(
                    frame.observed-frame.truth.clean-frame.truth.interf-frame.truth.noise))),
                source_sum_max_abs_error=float(np.max(np.abs(
                    frame.truth.interf-receiver.components['interferer']['filtered'].sum(axis=0)))))
