"""Integration tests for autorefl fit_server + FitClient.

Run the server in one terminal:
    python -m autorefl.fit_server

Then run this script:
    python test_fit_server.py

What it tests:
  1. /status — server is alive
  2. /setup — returns native labels, empty derived labels for plain Experiment
  3. /qprofiles — prior Q-profile ensemble (initial_points path)
  4. /fit cold start — DREAM fit + qprofiles in one call
  5. /fit warm start — second fit reuses server's chain population
  6. /reset — clears warm-start state; subsequent cold fit succeeds
"""

import asyncio
import numpy as np
from bumps.names import FitProblem, Parameter
from bumps.initpop import generate
from refl1d.names import SLD, Slab, Experiment
from refl1d.probe import NeutronProbe

from autorefl.fit_client import FitClient
from autorefl.instrument import MAGIK


# ── build a dead-simple 1-model problem ────────────────────────────────────

def make_problem():
    """Single-slab film on Si in D2O, no data files required."""
    d2o  = SLD(name='d2o',  rho=6.3)
    film = SLD(name='film', rho=2.0)
    si   = SLD(name='si',   rho=2.07)

    film_thickness = Parameter(name='thickness', value=100.0).range(50, 200)
    film_roughness = Parameter(name='roughness', value=5.0).range(1, 15)

    sample = si | Slab(material=film, thickness=film_thickness, interface=film_roughness) | d2o

    L_ang = 5.0
    Q  = np.linspace(0.01, 0.25, 40)
    T  = np.degrees(np.arcsin(Q * L_ang / (4 * np.pi)))
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
    x     = measQ
    calc_tdtldl = [instr.Q2TdTLdL(measQ, x, measQ)]
    oversampling = 5
    resolution   = instr.resolution

    fit_options = {
        'samples': 200,
        'burn':    50,
        'pop':     8,
        'init':    'lhs',
        'alpha':   0.001,
        'steps':   0,
    }

    draw_points = generate(problem, init='lhs', pop=-50, use_point=False)

    async with FitClient(host='127.0.0.1', port=5100) as client:

        # 1. status
        alive = await client.is_alive()
        assert alive, 'Server not reachable — is fit_server running?'
        print('[1/6] /status OK')

        # 2. setup — plain Experiment has no molgroups layers
        setup = await client.post_setup(problem)
        assert setup['native_labels'] == list(problem.labels()), 'native labels mismatch'
        assert setup['derived_labels'] == [], 'expected no derived labels for plain Experiment'
        assert setup['prior_scales'] == [], 'expected no prior scales for plain Experiment'
        print(f'[2/6] /setup OK — {len(setup["native_labels"])} native labels, 0 derived')

        # 3. qprofiles
        qprofs = await client.post_qprofiles(
            problem=problem,
            draw_points=draw_points,
            calc_tdtldl=calc_tdtldl,
            oversampling=oversampling,
            resolution=resolution,
        )
        assert len(qprofs) == 1,                       'expected 1 model'
        assert qprofs[0].shape[0] == len(draw_points),  'wrong ndraws'
        assert qprofs[0].shape[1] == len(measQ),         'wrong nQ'
        print(f'[3/6] /qprofiles OK — shape {qprofs[0].shape}')

        # 4. cold fit
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
        assert len(qprofs_fit) == 1,             'expected 1 model'
        assert qprofs_fit[0].shape[0] == ndraws,  'qprof draw dim mismatch'
        print(f'[4/6] /fit cold OK — {ndraws} draws, best_logp={fit_fields["best_logp"]:.2f}')

        # 5. warm fit
        fit_fields2, _ = await client.post_fit(
            problem=problem,
            fit_options=fit_options,
            calc_tdtldl=calc_tdtldl,
            oversampling=oversampling,
            resolution=resolution,
            warm_start=True,
        )
        print(f'[5/6] /fit warm OK — best_logp={fit_fields2["best_logp"]:.2f}')

        # 6. reset then cold fit — confirms chain was cleared
        await client.post_reset()
        fit_fields3, _ = await client.post_fit(
            problem=problem,
            fit_options=fit_options,
            calc_tdtldl=calc_tdtldl,
            oversampling=oversampling,
            resolution=resolution,
            warm_start=False,
        )
        print(f'[6/6] /reset + cold fit OK — best_logp={fit_fields3["best_logp"]:.2f}')

    print('\nAll checks passed.')


if __name__ == '__main__':
    asyncio.run(main())
