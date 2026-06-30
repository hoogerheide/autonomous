"""Minimal integration test for autorefl fit_server + FitClient.

Run the server in one terminal:
    python -m autorefl.fit_server

Then run this script:
    python test_fit_server.py

What it tests:
  1. /status — server is alive
  2. /qprofiles — prior Q-profile ensemble (initial_points path)
  3. /fit cold start — DREAM fit + qprofiles in one call
  4. /fit warm start — second fit reuses server's chain population
"""

import asyncio
import numpy as np
from bumps.names import FitProblem, Parameter
from refl1d.names import SLD, Slab, Experiment

from autorefl.fit_client import FitClient
from autorefl.instrument import MAGIK


# ── build a dead-simple 1-model problem ────────────────────────────────────

def make_problem():
    """Single-slab film on Si in D2O, no data files required."""
    from refl1d.probe import NeutronProbe

    d2o   = SLD(name='d2o',  rho=6.3)
    film  = SLD(name='film', rho=2.0)
    si    = SLD(name='si',   rho=2.07)

    film_thickness = Parameter(name='thickness', value=100.0).range(50, 200)
    film_roughness = Parameter(name='roughness', value=5.0).range(1, 15)

    sample = si | Slab(material=film, thickness=film_thickness, interface=film_roughness) | d2o

    # MAGIK: monochromatic, L=5 Å. Convert Q to angle so _set_TLR works.
    L_ang = 5.0
    Q = np.linspace(0.01, 0.25, 40)
    T = np.degrees(np.arcsin(Q * L_ang / (4 * np.pi)))
    dT = 0.01 * T
    R  = np.ones_like(Q) * 1e-3
    dR = R * 0.05

    probe = NeutronProbe(T=T, dT=dT, L=L_ang*np.ones_like(T), dL=0.01*L_ang*np.ones_like(T),
                         data=(R, dR), back_reflectivity=False)
    probe.background.range(-1e-7, 1e-5)
    probe.intensity.range(0.9, 1.1)

    model = Experiment(sample=sample, probe=probe)
    return FitProblem([model])


async def main():
    problem = make_problem()
    instr   = MAGIK()

    measQ = np.linspace(0.01, 0.25, 20)
    x     = measQ                         # MAGIK: x == Q
    calc_tdtldl = [instr.Q2TdTLdL(measQ, x, measQ)]
    oversampling = 5
    resolution   = instr.resolution

    # tiny fit so the test is fast
    fit_options = {
        'samples': 200,   # 200 draws total
        'burn':    50,
        'pop':     8,
        'init':    'lhs',
        'alpha':   0.001,
        'steps':   0,
    }

    # draw_points for qprofiles test: small LHS population
    from bumps.initpop import generate
    draw_points = generate(problem, init='lhs', pop=-50, use_point=False)

    async with FitClient(host='127.0.0.1', port=5100) as client:

        # 1. status
        alive = await client.is_alive()
        assert alive, 'Server not reachable — is fit_server running?'
        print('[1/4] /status OK')

        # 2. qprofiles (initial_points path)
        qprofs = await client.post_qprofiles(
            problem=problem,
            draw_points=draw_points,
            calc_tdtldl=calc_tdtldl,
            oversampling=oversampling,
            resolution=resolution,
        )
        assert len(qprofs) == 1,                      'expected 1 model'
        assert qprofs[0].shape[0] == len(draw_points), 'wrong ndraws'
        assert qprofs[0].shape[1] == len(measQ),       'wrong nQ'
        print(f'[2/4] /qprofiles OK — shape {qprofs[0].shape}')

        # 3. cold fit
        fit_fields, qprofs_fit = await client.post_fit(
            problem=problem,
            fit_options=fit_options,
            calc_tdtldl=calc_tdtldl,
            oversampling=oversampling,
            resolution=resolution,
            warm_start=False,
        )
        ndraws = fit_fields['draw_points'].shape[0]
        assert fit_fields['best_x'].shape[0] == len(problem.getp()), 'wrong npars'
        assert len(qprofs_fit) == 1,            'expected 1 model'
        assert qprofs_fit[0].shape[0] == ndraws, 'qprof draw dim mismatch'
        print(f'[3/4] /fit cold OK — {ndraws} draws, best_logp={fit_fields["best_logp"]:.2f}')

        # 4. warm fit (server should have population from step 3)
        fit_fields2, _ = await client.post_fit(
            problem=problem,
            fit_options=fit_options,
            calc_tdtldl=calc_tdtldl,
            oversampling=oversampling,
            resolution=resolution,
            warm_start=True,
        )
        print(f'[4/4] /fit warm OK — best_logp={fit_fields2["best_logp"]:.2f}')

    print('\nAll checks passed.')


if __name__ == '__main__':
    asyncio.run(main())
