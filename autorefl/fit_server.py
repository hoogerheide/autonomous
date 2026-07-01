"""Standalone aiohttp fit server for AutoRefl.

Runs DreamFit + MPMapper Q-profile calculation and returns results as a
single HDF5 byte stream. Deploy on an HPC node; point FitClient at it.

Usage::

    python -m autorefl.fit_server [--host 0.0.0.0] [--port 5100]
"""

import argparse
import asyncio
import io
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import concurrent.futures
import multiprocessing

import dill
import h5py
import numpy as np
from aiohttp import web

from bumps.fitters import DreamFit, MonitorRunner
from bumps.initpop import generate
from .simulation import calc_expected_R

logger = logging.getLogger(__name__)


# ── Q-profile + derived worker (module level so ProcessPoolExecutor can pickle it) ────

_qprof_problem = None          # populated by _qprof_init in each worker process
_qprof_molgroups_index = None  # first molgroups model index, or None
_qprof_requested_labels = []   # derived labels to compute, or []


def _qprof_init(shared_bytes, molgroups_index=None, requested_labels=None):
    global _qprof_problem, _qprof_molgroups_index, _qprof_requested_labels
    _qprof_problem = dill.loads(shared_bytes[:])
    _qprof_molgroups_index = molgroups_index
    _qprof_requested_labels = requested_labels or []


def _qprof_worker(point):
    """Evaluate Q-profiles and (optionally) derived quantities at one draw point.

    setp + chisq_str are called exactly once, so derived extraction piggybacks
    on the already-evaluated model state at no extra cost.

    Returns (qprof_list, derived_dict) where derived_dict maps label→float.
    """
    mlist = list(_qprof_problem.models)

    # single setp + chisq_str updates all models
    _qprof_problem.setp(point)
    _qprof_problem.chisq_str()

    qprof = [
        calc_expected_R(m, *newvar,
                        oversampling=_qprof_problem.oversampling,
                        resolution=_qprof_problem.resolution)
        for m, newvar in zip(mlist, _qprof_problem.calcTdTLdL)
    ]

    derived = {}
    if _qprof_requested_labels and _qprof_molgroups_index is not None:
        model = mlist[_qprof_molgroups_index]
        for layer in _collect_molgroups_layers(model):
            iresults = {'parameters': {}}
            for group in [layer.base_group] + layer.add_groups + layer.overlay_groups:
                iresults = group._molgroup.fnWriteResults2Dict(iresults, group.name)
                iresults[group.name].update(group._stored_profile['referencepoints'])
            for group_key, props in iresults.items():
                if group_key == 'parameters':
                    continue
                if isinstance(props, dict):
                    for prop_key, val in props.items():
                        key = f'{group_key}.{prop_key}'
                        if key in _qprof_requested_labels:
                            derived[key] = float(val)

    return qprof, derived


# ── Molgroups derived parameter helpers ─────────────────────────────────────

def _is_molgroups_experiment(model) -> bool:
    # Use MRO names rather than hasattr(_molgroups_layers): that attribute is assigned
    # in __init__ but is not a declared dataclass field, so dill drops it on round-trip.
    return any(cls.__name__ in ('MolgroupsExperiment', 'MolgroupsMixedExperiment')
               for cls in type(model).__mro__)


def _collect_molgroups_layers(model) -> list:
    """Return list of MolgroupsLayer objects from a model after dill round-trip.

    _molgroups_layers is dropped by pickle (non-field __init__ attribute), so we
    reconstruct from model.sample, which is a declared dataclass field and survives.

    MolgroupsExperiment:      model.sample.molgroups_layer
    MolgroupsMixedExperiment: model.parts[i].sample.molgroups_layer for each part

    # TODO: replace with a native FitProblem/Experiment derived-parameters API when available.
    """
    layers = []
    if hasattr(model, 'parts'):
        for p in model.parts:
            if hasattr(p, 'sample') and hasattr(p.sample, 'molgroups_layer'):
                layers.append(p.sample.molgroups_layer)
    elif hasattr(model, 'sample') and hasattr(model.sample, 'molgroups_layer'):
        layers.append(model.sample.molgroups_layer)
    return layers


def _eval_derived_one(problem, model_index: int, pt: np.ndarray) -> dict:
    """Evaluate all molgroups derived quantities at a single parameter vector.

    Returns a flat {"group.property": scalar} dict.
    """
    problem.setp(pt)
    model = list(problem.models)[model_index]
    model.update()
    model.nllf()

    result = {}
    for layer in _collect_molgroups_layers(model):
        iresults = {'parameters': {}}
        for group in [layer.base_group] + layer.add_groups + layer.overlay_groups:
            iresults = group._molgroup.fnWriteResults2Dict(iresults, group.name)
            iresults[group.name].update(group._stored_profile['referencepoints'])
        for group_key, props in iresults.items():
            if group_key == 'parameters':
                continue
            if isinstance(props, dict):
                for prop_key, val in props.items():
                    result[f'{group_key}.{prop_key}'] = float(val)
    return result


def _discover_derived(problem) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Discover molgroups derived parameters and evaluate them over a prior population.

    Returns
    -------
    labels : list[str]
        Derived parameter names.
    prior_scales : np.ndarray, shape (D,)
        Empirical max-min range over the prior population (for par_scale).
    native_prior_draws : np.ndarray, shape (N_valid, npars)
        The native LHS prior draw points that evaluated successfully.
    derived_prior_draws : np.ndarray, shape (N_valid, D)
        Derived values at each of those native draw points.

    Returns four empty objects if no molgroups models are found.
    """
    empty = [], np.array([]), np.empty((0, len(problem.getp()))), np.empty((0, 0))

    molgroups_model_indices = [
        i for i, m in enumerate(problem.models) if _is_molgroups_experiment(m)
    ]
    if not molgroups_model_indices:
        return empty

    pop_size = 200
    try:
        prior_pts = generate(problem, init='lhs', pop=pop_size, use_point=False)
    except Exception:
        return empty

    model_index = molgroups_model_indices[0]
    labels = None
    valid_native = []
    all_vals = []
    last_exc = None
    for pt in prior_pts:
        try:
            d = _eval_derived_one(problem, model_index, pt)
        except Exception as e:
            last_exc = e
            continue
        if labels is None:
            labels = list(d.keys())
        valid_native.append(pt)
        all_vals.append([d[k] for k in labels])

    if not labels or not all_vals:
        if last_exc is not None:
            raise RuntimeError(
                f'_discover_derived: all {len(prior_pts)} prior evaluations failed. '
                f'Last exception: {last_exc!r}'
            )
        return empty

    derived_arr = np.array(all_vals)          # (N_valid, D)
    native_arr  = np.array(valid_native)      # (N_valid, npars)
    prior_scales = derived_arr.max(axis=0) - derived_arr.min(axis=0)
    prior_scales = np.where(prior_scales == 0, 1.0, prior_scales)
    return labels, prior_scales, native_arr, derived_arr


# ── Derived draws worker (module level for ProcessPoolExecutor) ──────────────

_derived_problem = None
_derived_model_index = None


def _derived_init(shared_bytes, model_index: int):
    global _derived_problem, _derived_model_index
    _derived_problem = dill.loads(shared_bytes[:])
    _derived_model_index = model_index


def _derived_worker(pt):
    return _eval_derived_one(_derived_problem, _derived_model_index, pt)


def _calc_derived_sync(problem, draw_points: np.ndarray, requested_labels: List[str],
                       manager) -> np.ndarray:
    """Evaluate derived quantities at each draw point. Returns N×D array."""
    molgroups_model_indices = [
        i for i, m in enumerate(problem.models) if _is_molgroups_experiment(m)
    ]
    if not molgroups_model_indices or not requested_labels:
        return np.empty((len(draw_points), 0))

    model_index = molgroups_model_indices[0]
    shared_bytes = manager.Array("B", dill.dumps(problem))
    with concurrent.futures.ProcessPoolExecutor(
        initializer=_derived_init, initargs=(shared_bytes, model_index)
    ) as executor:
        results = list(executor.map(_derived_worker, draw_points))

    return np.array([[r.get(lbl, np.nan) for lbl in requested_labels] for r in results])


@dataclass
class ServerState:
    job_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    is_busy: bool = False
    fitter: Optional[DreamFit] = None   # retained between calls for warm-start
    mp_manager: Optional[multiprocessing.managers.SyncManager] = None
    # populated by /setup, cleared by /reset
    derived_labels: List[str] = field(default_factory=list)
    derived_prior_scales: np.ndarray = field(default_factory=lambda: np.array([]))


routes = web.RouteTableDef()


@routes.get('/status')
async def get_status(request: web.Request) -> web.Response:
    state: ServerState = request.app['state']
    return web.json_response({
        'is_busy': state.is_busy,
        'has_state': state.fitter is not None and state.fitter.state is not None,
    })


@routes.post('/setup')
async def post_setup(request: web.Request) -> web.Response:
    """One-time setup: discover derived parameters, compute prior scales, return prior draws.

    Multipart fields:
        problem — dill-serialised bumps FitProblem bytes

    Response: HDF5 with:
        attrs['native_labels']  — JSON list[str] from problem.labels()
        attrs['derived_labels'] — JSON list[str] of molgroups derived parameter names
        attrs['prior_scales']   — JSON list[float], one per derived label
        setup/native_draws      — (N, npars) prior draw points that evaluated successfully
        setup/derived_draws     — (N, D) derived values at those points (empty if no derived)
    """
    state: ServerState = request.app['state']
    try:
        reader = await request.multipart()
        problem_bytes = None
        async for part in reader:
            if part.name == 'problem':
                problem_bytes = await part.read()

        if problem_bytes is None:
            return web.json_response({'error': 'missing problem field'}, status=400)

        problem = dill.loads(problem_bytes)

        derived_labels, prior_scales, native_draws, derived_draws = \
            await asyncio.to_thread(_discover_derived, problem)
        state.derived_labels = derived_labels
        state.derived_prior_scales = prior_scales

        buf = io.BytesIO()
        with h5py.File(buf, 'w') as f:
            f.attrs['native_labels']  = json.dumps(list(problem.labels()))
            f.attrs['derived_labels'] = json.dumps(derived_labels)
            f.attrs['prior_scales']   = json.dumps(prior_scales.tolist())
            if native_draws.size > 0:
                f.create_dataset('setup/native_draws',  data=native_draws,  compression='gzip')
            if derived_draws.size > 0:
                f.create_dataset('setup/derived_draws', data=derived_draws, compression='gzip')
        buf.seek(0)

        return web.Response(
            body=buf.read(),
            content_type='application/x-hdf5',
            headers={'Content-Disposition': 'attachment; filename="setup.h5"'},
        )

    except Exception as e:
        logger.exception('Setup failed: %s', e)
        return web.json_response({'error': str(e)}, status=500)


@routes.post('/reset')
async def post_reset(request: web.Request) -> web.Response:
    """Clear warm-start chain and derived parameter state.

    Call before starting a new experiment to ensure a cold start.
    """
    state: ServerState = request.app['state']
    state.fitter = None
    state.derived_labels = []
    state.derived_prior_scales = np.array([])
    logger.info('Server state reset.')
    return web.json_response({'status': 'ok'})


@routes.post('/derived_draws')
async def post_derived_draws(request: web.Request) -> web.Response:
    """Evaluate molgroups derived quantities at posterior draw points.

    Multipart fields:
        problem          — dill-serialised bumps FitProblem bytes
        draw_points      — JSON: 2-D array (ndraw × npars)
        requested_labels — JSON: list[str] of derived parameter names to return

    Response: HDF5 with dataset ``derived/draws`` (ndraw × D) and string
    attribute ``derived/labels``.
    """
    state: ServerState = request.app['state']
    try:
        reader = await request.multipart()
        fields = {}
        problem_bytes = None
        async for part in reader:
            if part.name == 'problem':
                problem_bytes = await part.read()
            else:
                fields[part.name] = (await part.read()).decode()

        if problem_bytes is None:
            return web.json_response({'error': 'missing problem field'}, status=400)

        problem = dill.loads(problem_bytes)
        draw_points = np.array(json.loads(fields['draw_points']))
        requested_labels = json.loads(fields.get('requested_labels', '[]'))

        derived_arr = await asyncio.to_thread(
            _calc_derived_sync, problem, draw_points, requested_labels, state.mp_manager
        )

        buf = io.BytesIO()
        with h5py.File(buf, 'w') as f:
            f.create_dataset('derived/draws', data=derived_arr, compression='gzip')
            f.attrs['derived_labels'] = json.dumps(requested_labels)
        buf.seek(0)

        return web.Response(
            body=buf.read(),
            content_type='application/x-hdf5',
            headers={'Content-Disposition': 'attachment; filename="derived.h5"'},
        )

    except Exception as e:
        logger.exception('derived_draws failed: %s', e)
        return web.json_response({'error': str(e)}, status=500)


def _run_dream(problem, fit_options: dict, fitter: Optional[DreamFit]) -> DreamFit:
    """Synchronous DreamFit run. Intended to be called in a thread.

    Pass an existing fitter for warm-start (its state is resumed); pass None
    for a cold start. Returns the fitter so the caller can retain it.
    """
    if fitter is None or fitter.state is None:
        fitter = DreamFit(problem)
    else:
        # Re-attach to new problem object (deserialized each call)
        fitter.problem = problem

    monitors = MonitorRunner(monitors=[], problem=problem)
    opts = {
        'samples':  fit_options.get('samples', 5000),
        'burn':     fit_options.get('burn', 1000),
        'pop':      fit_options.get('pop', 10),
        'init':     fit_options.get('init', 'lhs'),
        'thin':     fit_options.get('thin', 1),
        'alpha':    fit_options.get('alpha', 0.001),
        'outliers': fit_options.get('outliers', 'iqr'),
        'trim':     fit_options.get('trim', True),
        'steps':    fit_options.get('steps', 0),
    }
    fitter.solve(monitors, mapper=lambda p: list(map(problem.nllf, p)), **opts)
    return fitter


def _calc_qprofiles_sync(problem, draw_points: np.ndarray, calc_tdtldl, oversampling: int,
                         resolution: str, manager,
                         requested_derived_labels: List[str] = []) -> Tuple[list, np.ndarray]:
    """Q-profile (and optionally derived quantity) calculation via ProcessPoolExecutor.

    Returns (qprofs, derived_arr) where:
        qprofs      — list of (ndraw, nQ) arrays, one per model
        derived_arr — (ndraw, D) array; empty if requested_derived_labels is []
    """
    setattr(problem, 'calcTdTLdL', calc_tdtldl)
    setattr(problem, 'oversampling', oversampling)
    setattr(problem, 'resolution', resolution)

    molgroups_index = next(
        (i for i, m in enumerate(problem.models) if _is_molgroups_experiment(m)), None
    ) if requested_derived_labels else None

    shared_bytes = manager.Array("B", dill.dumps(problem))
    with concurrent.futures.ProcessPoolExecutor(
        initializer=_qprof_init,
        initargs=(shared_bytes, molgroups_index, requested_derived_labels),
    ) as executor:
        results = list(executor.map(_qprof_worker, draw_points))

    nmodels = len(calc_tdtldl)
    qprofs = [np.array([r[0][i] for r in results]) for i in range(nmodels)]

    if requested_derived_labels:
        derived_arr = np.array(
            [[r[1].get(lbl, np.nan) for lbl in requested_derived_labels] for r in results]
        )
    else:
        derived_arr = np.empty((len(draw_points), 0))

    return qprofs, derived_arr


def _build_hdf5(dream_state, qprofs: list,
                derived_arr: np.ndarray = None,
                derived_labels: List[str] = []) -> bytes:
    """Serialise MCMCDraw state + qprofs (+ optional derived draws) into HDF5 bytes."""
    buf = io.BytesIO()
    with h5py.File(buf, 'w') as f:
        # --- chains (nsteps × nchains × npars) ---
        points, logp, _ = dream_state.chains()
        f.create_dataset('chains/points', data=points, compression='gzip')
        f.create_dataset('chains/logp', data=logp, compression='gzip')

        # --- best ---
        best_x, best_logp = dream_state.best()
        f.create_dataset('best/x', data=best_x)
        f.attrs['best_logp'] = float(best_logp)

        # --- draw (thinned posterior for entropy + FOM) ---
        draw = dream_state.draw()
        f.create_dataset('draw/points', data=draw.points, compression='gzip')
        f.create_dataset('draw/logp', data=draw.logp, compression='gzip')

        # --- Q-profiles: one dataset per model ---
        for i, qp in enumerate(qprofs):
            f.create_dataset(f'qprofs/model_{i}', data=qp, compression='gzip')
        f.attrs['nmodels'] = len(qprofs)

        # --- derived draws (optional) ---
        if derived_arr is not None and derived_arr.size > 0:
            f.create_dataset('derived/draws', data=derived_arr, compression='gzip')
            f.attrs['derived_labels'] = json.dumps(derived_labels)

    buf.seek(0)
    return buf.read()


@routes.post('/fit')
async def post_fit(request: web.Request) -> web.Response:
    """Run DreamFit + Q-profile calculation.

    Multipart fields:
        problem                  — dill-serialised bumps FitProblem bytes
        fit_options              — JSON object (samples, burn, pop, init, thin, alpha, steps)
        warm_start               — "true" or "false"
        calc_tdtldl              — JSON: list of M lists, each [T, dT, L, dL] as flat arrays
        oversampling             — integer string
        resolution               — "normal" or "uniform"
        requested_derived_labels — JSON: list[str] of derived labels to compute (optional)
    """
    state: ServerState = request.app['state']

    async with state.job_lock:
        state.is_busy = True
        try:
            reader = await request.multipart()
            fields = {}
            problem_bytes = None

            async for part in reader:
                if part.name == 'problem':
                    problem_bytes = await part.read()
                else:
                    fields[part.name] = (await part.read()).decode()

            if problem_bytes is None:
                return web.json_response({'error': 'missing problem field'}, status=400)

            problem = dill.loads(problem_bytes)
            fit_options = json.loads(fields.get('fit_options', '{}'))
            warm_start = fields.get('warm_start', 'false').lower() == 'true'
            oversampling = int(fields.get('oversampling', '11'))
            resolution = fields.get('resolution', 'normal')
            requested_derived_labels = json.loads(fields.get('requested_derived_labels', '[]'))

            raw_coords = json.loads(fields.get('calc_tdtldl', '[]'))
            calc_tdtldl = [
                tuple(np.array(arr) for arr in model_coords)
                for model_coords in raw_coords
            ]

            fitter_in = state.fitter if warm_start else None

            logger.info('Starting DreamFit (warm_start=%s, samples=%s)',
                        warm_start, fit_options.get('samples', 5000))

            fitter = await asyncio.to_thread(
                _run_dream, problem, fit_options, fitter_in
            )
            state.fitter = fitter
            dream_state = fitter.state

            logger.info('DreamFit complete. Calculating Q-profiles%s for %d draw points.',
                        ' + derived' if requested_derived_labels else '',
                        dream_state.draw().points.shape[0])

            qprofs, derived_arr = await asyncio.to_thread(
                _calc_qprofiles_sync, problem, dream_state.draw().points,
                calc_tdtldl, oversampling, resolution, state.mp_manager,
                requested_derived_labels,
            )

            hdf5_bytes = await asyncio.to_thread(
                _build_hdf5, dream_state, qprofs, derived_arr, requested_derived_labels
            )

            logger.info('Returning HDF5 (%d bytes)', len(hdf5_bytes))
            return web.Response(
                body=hdf5_bytes,
                content_type='application/x-hdf5',
                headers={'Content-Disposition': 'attachment; filename="fit_result.h5"'},
            )

        except Exception as e:
            logger.exception('Fit failed: %s', e)
            return web.json_response({'error': str(e)}, status=500)
        finally:
            state.is_busy = False


@routes.post('/qprofiles')
async def post_qprofiles(request: web.Request) -> web.Response:
    """Calculate Q-profiles for an arbitrary set of draw points (no fitting).

    Used by AutoReflBase.initial_points() to get the prior Q-profile ensemble
    without running DREAM.

    Multipart fields:
        problem      — dill-serialised bumps FitProblem bytes
        draw_points  — JSON: 2-D array (npoints × npars)
        calc_tdtldl  — JSON: list of M lists, each [T, dT, L, dL] as flat arrays
        oversampling — integer string
        resolution   — "normal" or "uniform"
    """
    try:
        reader = await request.multipart()
        fields = {}
        problem_bytes = None

        async for part in reader:
            if part.name == 'problem':
                problem_bytes = await part.read()
            else:
                fields[part.name] = (await part.read()).decode()

        if problem_bytes is None:
            return web.json_response({'error': 'missing problem field'}, status=400)

        problem = dill.loads(problem_bytes)
        draw_points = np.array(json.loads(fields['draw_points']))
        oversampling = int(fields.get('oversampling', '11'))
        resolution = fields.get('resolution', 'normal')

        raw_coords = json.loads(fields.get('calc_tdtldl', '[]'))
        calc_tdtldl = [
            tuple(np.array(arr) for arr in model_coords)
            for model_coords in raw_coords
        ]

        state: ServerState = request.app['state']
        qprofs, _ = await asyncio.to_thread(
            _calc_qprofiles_sync, problem, draw_points, calc_tdtldl, oversampling, resolution, state.mp_manager
        )

        buf = io.BytesIO()
        with h5py.File(buf, 'w') as f:
            for i, qp in enumerate(qprofs):
                f.create_dataset(f'qprofs/model_{i}', data=qp, compression='gzip')
            f.attrs['nmodels'] = len(qprofs)
        buf.seek(0)

        return web.Response(
            body=buf.read(),
            content_type='application/x-hdf5',
            headers={'Content-Disposition': 'attachment; filename="qprofs.h5"'},
        )

    except Exception as e:
        logger.exception('qprofiles failed: %s', e)
        return web.json_response({'error': str(e)}, status=500)


async def _on_startup(app: web.Application) -> None:
    state: ServerState = app['state']
    state.mp_manager = multiprocessing.Manager()
    logger.info('Multiprocessing manager started')


async def _on_cleanup(app: web.Application) -> None:
    state: ServerState = app['state']
    if state.mp_manager is not None:
        state.mp_manager.shutdown()
        logger.info('Multiprocessing manager shut down')


def build_app() -> web.Application:
    app = web.Application(client_max_size=512 * 1024 * 1024)  # 512 MB upload limit
    app['state'] = ServerState()
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    app.add_routes(routes)
    return app


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='AutoRefl fit server')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5100)
    args = parser.parse_args()
    web.run_app(build_app(), host=args.host, port=args.port)
