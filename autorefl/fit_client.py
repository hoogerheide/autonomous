"""HTTP client for the AutoRefl fit server (autorefl.fit_server).

Replaces Refl1DClient usage in AutoReflBase: instead of serialising via the
bumps webview WebSocket protocol, FitClient sends the problem over HTTP and
receives HDF5 bytes containing both the DREAM state and Q-profiles.
"""

import io
import json
import logging
from typing import List, Optional, Tuple

import aiohttp
import dill
import h5py
import numpy as np

logger = logging.getLogger(__name__)

# aiohttp default is 1 MB; fits can be large
_RESPONSE_LIMIT = 512 * 1024 * 1024  # 512 MB


def _encode_tdtldl(calc_tdtldl) -> str:
    """Serialise per-model (T,dT,L,dL) tuples to JSON."""
    return json.dumps([
        [arr.tolist() for arr in model_coords]
        for model_coords in calc_tdtldl
    ])


def _read_qprofs(f: h5py.File) -> List[np.ndarray]:
    nmodels = int(f.attrs['nmodels'])
    return [f[f'qprofs/model_{i}'][()] for i in range(nmodels)]


class FitClient:
    """Thin async HTTP client for the autorefl fit server.

    Parameters
    ----------
    host:
        Hostname or IP of the fit server.
    port:
        TCP port the fit server is listening on.
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 5100) -> None:
        self.base_url = f'http://{host}:{port}'
        self._session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> None:
        connector = aiohttp.TCPConnector()
        self._session = aiohttp.ClientSession(
            connector=connector,
            connector_owner=True,
        )

    async def disconnect(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.disconnect()

    def _session_or_raise(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError('FitClient not connected — call connect() first.')
        return self._session

    async def post_fit(
        self,
        problem,
        fit_options: dict,
        calc_tdtldl,
        oversampling: int,
        resolution: str,
        warm_start: bool = False,
        timeout_s: float = 7200.0,
    ) -> Tuple[dict, List[np.ndarray]]:
        """Run DREAM fit + Q-profile calculation on the remote server.

        Returns
        -------
        fit_fields : dict
            Keys: ``chains`` (nsteps × nchains × npars), ``logp`` (nsteps × nchains),
            ``best_x`` (npars,), ``best_logp`` (float),
            ``draw_points`` (ndraw × npars), ``draw_logp`` (ndraw,).
        qprofs : list[np.ndarray]
            One array per model, shape (ndraw, nQ).
        """
        session = self._session_or_raise()

        problem_bytes = dill.dumps(problem)

        data = aiohttp.FormData()
        data.add_field('problem', problem_bytes, content_type='application/octet-stream')
        data.add_field('fit_options', json.dumps(fit_options))
        data.add_field('warm_start', 'true' if warm_start else 'false')
        data.add_field('calc_tdtldl', _encode_tdtldl(calc_tdtldl))
        data.add_field('oversampling', str(oversampling))
        data.add_field('resolution', resolution)

        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with session.post(
            f'{self.base_url}/fit',
            data=data,
            timeout=timeout,
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f'Fit server returned {resp.status}: {body}')
            hdf5_bytes = await resp.read()

        return _parse_fit_hdf5(hdf5_bytes)

    async def post_qprofiles(
        self,
        problem,
        draw_points: np.ndarray,
        calc_tdtldl,
        oversampling: int,
        resolution: str,
        timeout_s: float = 600.0,
    ) -> List[np.ndarray]:
        """Calculate Q-profiles for arbitrary draw points (no fitting).

        Used by ``AutoReflBase.initial_points()``.

        Returns
        -------
        qprofs : list[np.ndarray]
            One array per model, shape (npoints, nQ).
        """
        session = self._session_or_raise()

        problem_bytes = dill.dumps(problem)

        data = aiohttp.FormData()
        data.add_field('problem', problem_bytes, content_type='application/octet-stream')
        data.add_field('draw_points', json.dumps(draw_points.tolist()))
        data.add_field('calc_tdtldl', _encode_tdtldl(calc_tdtldl))
        data.add_field('oversampling', str(oversampling))
        data.add_field('resolution', resolution)

        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with session.post(
            f'{self.base_url}/qprofiles',
            data=data,
            timeout=timeout,
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f'Fit server /qprofiles returned {resp.status}: {body}')
            hdf5_bytes = await resp.read()

        buf = io.BytesIO(hdf5_bytes)
        with h5py.File(buf, 'r') as f:
            return _read_qprofs(f)

    async def is_alive(self) -> bool:
        session = self._session_or_raise()
        try:
            async with session.get(f'{self.base_url}/status', timeout=aiohttp.ClientTimeout(total=5)) as resp:
                return resp.status == 200
        except Exception:
            return False


def _parse_fit_hdf5(hdf5_bytes: bytes) -> Tuple[dict, List[np.ndarray]]:
    """Deserialise the HDF5 response from /fit into Python objects."""
    buf = io.BytesIO(hdf5_bytes)
    with h5py.File(buf, 'r') as f:
        fit_fields = {
            'chains':       f['chains/points'][()],
            'logp':         f['chains/logp'][()],
            'best_x':       f['best/x'][()],
            'best_logp':    float(f.attrs['best_logp']),
            'draw_points':  f['draw/points'][()],
            'draw_logp':    f['draw/logp'][()],
        }
        qprofs = _read_qprofs(f)
    return fit_fields, qprofs
