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
from typing import Optional

import dill
import h5py
import numpy as np
from aiohttp import web

from .inference import DreamFitPlus, _MP_calc_qprofile
from bumps.mapper import MPMapper

logger = logging.getLogger(__name__)


@dataclass
class ServerState:
    job_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    is_busy: bool = False
    # Last chain population (nchains × npars) for warm-starting the next fit.
    last_population: Optional[np.ndarray] = None


routes = web.RouteTableDef()


@routes.get('/status')
async def get_status(request: web.Request) -> web.Response:
    state: ServerState = request.app['state']
    return web.json_response({
        'is_busy': state.is_busy,
        'has_population': state.last_population is not None,
    })


def _run_dream(problem, fit_options: dict, initial_population: Optional[np.ndarray]) -> 'bumps.dream.state.MCMCDraw':
    """Synchronous DreamFit run. Intended to be called in a thread."""
    fitter = DreamFitPlus(problem)
    opts = {
        'samples': fit_options.get('samples', 5000),
        'burn':    fit_options.get('burn', 1000),
        'pop':     fit_options.get('pop', 10),
        'init':    fit_options.get('init', 'lhs'),
        'thin':    fit_options.get('thin', 1),
        'alpha':   fit_options.get('alpha', 0.001),
        'outliers': fit_options.get('outliers', 'IQR'),
        'trim':    fit_options.get('trim', True),
        'steps':   fit_options.get('steps', 0),
    }
    fitter.solve(initial_population=initial_population, **opts)
    return fitter.state


def _calc_qprofiles_sync(problem, draw_points: np.ndarray, calc_tdtldl, oversampling: int, resolution: str) -> list:
    """Synchronous MPMapper Q-profile calculation. Intended to be called in a thread."""
    setattr(problem, 'calcTdTLdL', calc_tdtldl)
    setattr(problem, 'oversampling', oversampling)
    setattr(problem, 'resolution', resolution)

    mapper = MPMapper.start_mapper(problem, None, cpus=0)
    try:
        mappercalc = lambda pts: MPMapper.pool.map(_MP_calc_qprofile, ((MPMapper.problem_id, p) for p in pts))
        res = mappercalc(draw_points)
    finally:
        MPMapper.stop_mapper(mapper)
        MPMapper.pool = None

    nmodels = len(calc_tdtldl)
    qprofs = [np.array([r[i] for r in res]) for i in range(nmodels)]
    return qprofs


def _build_hdf5(dream_state, qprofs: list) -> bytes:
    """Serialise MCMCDraw state + qprofs into an in-memory HDF5 and return bytes."""
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

    buf.seek(0)
    return buf.read()


@routes.post('/fit')
async def post_fit(request: web.Request) -> web.Response:
    """Run DreamFit + Q-profile calculation.

    Multipart fields:
        problem      — dill-serialised bumps FitProblem bytes
        fit_options  — JSON object (samples, burn, pop, init, thin, alpha, steps)
        warm_start   — "true" or "false"
        calc_tdtldl  — JSON: list of M lists, each [T, dT, L, dL] as flat arrays
        oversampling — integer string
        resolution   — "normal" or "uniform"
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

            raw_coords = json.loads(fields.get('calc_tdtldl', '[]'))
            calc_tdtldl = [
                tuple(np.array(arr) for arr in model_coords)
                for model_coords in raw_coords
            ]

            initial_population = state.last_population if warm_start else None

            logger.info('Starting DreamFit (warm_start=%s, samples=%s)',
                        warm_start, fit_options.get('samples', 5000))

            dream_state = await asyncio.to_thread(
                _run_dream, problem, fit_options, initial_population
            )

            _, chains, _ = dream_state.chains()
            state.last_population = chains[-1, :, :]

            logger.info('DreamFit complete. Calculating Q-profiles for %d draw points.',
                        dream_state.draw().points.shape[0])

            qprofs = await asyncio.to_thread(
                _calc_qprofiles_sync, problem, dream_state.draw().points,
                calc_tdtldl, oversampling, resolution
            )

            hdf5_bytes = await asyncio.to_thread(_build_hdf5, dream_state, qprofs)

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

        qprofs = await asyncio.to_thread(
            _calc_qprofiles_sync, problem, draw_points, calc_tdtldl, oversampling, resolution
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


def build_app() -> web.Application:
    app = web.Application(client_max_size=512 * 1024 * 1024)  # 512 MB upload limit
    app['state'] = ServerState()
    app.add_routes(routes)
    return app


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='AutoRefl fit server')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5100)
    args = parser.parse_args()
    web.run_app(build_app(), host=args.host, port=args.port)
