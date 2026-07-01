"""Integration test for derived parameter endpoints: /setup, /reset, /fit with derived.

Requires molgroups to be installed in the same environment.

Run the server in one terminal:
    python -m autorefl.fit_server

Then run this script:
    python test_fit_server_derived.py

What it tests:
  1. /status — server is alive
  2. /setup — discovers molgroups derived labels and prior scales
  3. /fit with requested_derived_labels — derived draws computed in the same worker
     pass as qprofs (one setp per draw point, not two)
  4. /derived_draws — fallback endpoint still works for ad-hoc evaluation
  5. /reset — clears server state; subsequent /setup returns fresh labels
"""

import asyncio
import numpy as np
from bumps.names import FitProblem, Parameter
from bumps.initpop import generate
from refl1d.names import SLD, Slab
from refl1d.probe import NeutronProbe

from molgroups import components as cmp
from molgroups.refl1d_interface import (
    SolidSupportedBilayer,
    MolgroupsLayer,
    MolgroupsStack,
    MolgroupsExperiment,
)

from autorefl.fit_client import FitClient
from autorefl.instrument import MAGIK


# ── minimal molgroups problem (single contrast, no data files) ───────────────

def make_probe(rho_bulk: float) -> NeutronProbe:
    L_ang = 5.0
    Q  = np.linspace(0.01, 0.20, 30)
    T  = np.degrees(np.arcsin(Q * L_ang / (4 * np.pi)))
    dT = 0.01 * T
    R  = np.ones_like(Q) * 1e-3
    dR = R * 0.05
    probe = NeutronProbe(
        T=T, dT=dT,
        L=L_ang * np.ones_like(T), dL=0.01 * L_ang * np.ones_like(T),
        data=(R, dR), back_reflectivity=False,
    )
    probe.background.range(-1e-7, 1e-5)
    probe.intensity.range(0.9, 1.1)
    return probe


def make_problem() -> FitProblem:
    """Single-contrast ssBLM on silicon, no data files required."""

    # shared structural parameters
    vf_bilayer = Parameter(name='volume fraction bilayer', value=0.9).range(0.5, 1.0)
    l_lipid1   = Parameter(name='inner acyl chain thickness', value=11.0).range(8, 18)
    l_lipid2   = Parameter(name='outer acyl chain thickness', value=11.0).range(8, 18)
    l_sub      = Parameter(name='submembrane thickness', value=10.0).range(0, 30)
    sigma      = Parameter(name='bilayer roughness', value=5.0).range(0.5, 9)
    global_rough = Parameter(name='substrate roughness', value=5.0).range(2, 9)
    d_siox     = Parameter(name='siox thickness', value=10.0).range(5, 30)

    # materials
    d2o     = SLD(name='d2o',     rho=6.3)
    silicon = SLD(name='silicon', rho=2.07)
    siox    = SLD(name='siox',    rho=4.1)
    d2o.rho.range(5.5, 6.36)
    siox.rho.range(3.0, 4.8)

    # substrate slab stack (no molgroups layer yet)
    layer_si   = Slab(material=silicon, thickness=0.0, interface=global_rough)
    layer_siox = Slab(material=siox, thickness=d_siox, interface=global_rough)
    substrate  = layer_si | layer_siox

    # single DOPC bilayer
    DOPC = cmp.Lipid(name='DOPC', headgroup=cmp.pc, tails=2 * [cmp.oleoyl], methyls=[cmp.methyl])

    blm = SolidSupportedBilayer(
        name='bilayer',
        overlap=20.0,
        lipids=[DOPC],
        inner_lipid_nf=[1.0],
        outer_lipid_nf=[1.0],
        rho_substrate=siox.rho,
        l_siox=0.0,
        vf_bilayer=vf_bilayer,
        l_lipid1=l_lipid1,
        l_lipid2=l_lipid2,
        l_submembrane=l_sub,
        substrate_rough=global_rough,
        sigma=sigma,
    )

    mollayer = MolgroupsLayer(
        base_group=blm,
        thickness=150.0,
        contrast=d2o,
        name='bilayer layer d2o',
    )

    sample = MolgroupsStack(substrate=substrate, molgroups_layer=mollayer, name=mollayer.name)
    probe  = make_probe(rho_bulk=6.3)

    model = MolgroupsExperiment(sample=sample, probe=probe, dz=0.5, step_interfaces=False)
    return FitProblem([model])


async def main():
    problem = make_problem()
    instr   = MAGIK()

    measQ = np.linspace(0.01, 0.20, 15)
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

    async with FitClient(host='127.0.0.1', port=5100) as client:

        # 1. status
        alive = await client.is_alive()
        assert alive, 'Server not reachable — is fit_server running?'
        print('[1/5] /status OK')

        # 2. setup — should discover molgroups derived labels and return prior draws
        setup = await client.post_setup(problem)
        native_labels   = setup['native_labels']
        derived_labels  = setup['derived_labels']
        prior_scales    = setup['prior_scales']
        native_draws    = setup['native_draws']
        derived_draws   = setup['derived_draws']

        assert native_labels == list(problem.labels()), 'native labels mismatch'
        assert len(derived_labels) > 0, \
            'Expected derived labels from MolgroupsExperiment — got none. ' \
            'Check that molgroups is installed and fnWriteResults2Dict returns data.'
        assert len(prior_scales) == len(derived_labels), \
            f'prior_scales length {len(prior_scales)} != derived_labels length {len(derived_labels)}'
        assert all(s > 0 for s in prior_scales), \
            'All prior scales should be positive (max-min over LHS prior population)'
        assert native_draws.shape[1] == len(native_labels), 'native_draws column count mismatch'
        assert derived_draws.shape == (native_draws.shape[0], len(derived_labels)), \
            f'derived_draws shape mismatch: {derived_draws.shape}'

        print(f'[2/5] /setup OK — {native_draws.shape[0]} valid prior draws')
        print(f'      native labels ({len(native_labels)}): {native_labels}')
        print(f'      derived labels ({len(derived_labels)}):')
        for lbl, scale in zip(derived_labels, prior_scales):
            print(f'        {lbl:50s}  prior scale={scale:.4g}')

        # 3. cold fit with derived labels — server computes qprofs + derived in one pass
        n_request = min(3, len(derived_labels))
        requested = derived_labels[:n_request]

        fit_fields, _ = await client.post_fit(
            problem=problem,
            fit_options=fit_options,
            calc_tdtldl=calc_tdtldl,
            oversampling=oversampling,
            resolution=resolution,
            warm_start=False,
            requested_derived_labels=requested,
        )
        draw_pts = fit_fields['draw_points']
        ndraws   = draw_pts.shape[0]
        assert 'derived_draws' in fit_fields, \
            'derived_draws missing from fit_fields — server did not return derived data'
        derived_arr = fit_fields['derived_draws']
        assert derived_arr.shape == (ndraws, n_request), \
            f'Expected derived_draws shape ({ndraws}, {n_request}), got {derived_arr.shape}'
        assert not np.any(np.isnan(derived_arr)), \
            'NaN values in derived draws — model evaluation failed for some draw points'

        print(f'[3/5] /fit with derived OK — {ndraws} draws, best_logp={fit_fields["best_logp"]:.2f}')
        print(f'      requested: {requested}')
        for i, lbl in enumerate(requested):
            col = derived_arr[:, i]
            print(f'        {lbl}: median={np.median(col):.4g}  std={np.std(col):.4g}')

        # 4. /derived_draws fallback — same labels, same draw points, should match
        derived_arr2 = await client.post_derived_draws(
            problem=problem,
            draw_points=draw_pts,
            requested_labels=requested,
        )
        assert derived_arr2.shape == derived_arr.shape, \
            f'/derived_draws shape {derived_arr2.shape} != /fit derived shape {derived_arr.shape}'
        # values won't be bit-identical (different process workers), but medians should be close
        for i, lbl in enumerate(requested):
            med1, med2 = np.median(derived_arr[:, i]), np.median(derived_arr2[:, i])
            assert abs(med1 - med2) < 0.1 * abs(med1) + 1e-12, \
                f'Median mismatch for {lbl!r}: /fit={med1:.4g} vs /derived_draws={med2:.4g}'
        print(f'[4/5] /derived_draws fallback OK — shape {derived_arr2.shape}, medians match')

        # 5. reset — clears server state; /setup on same problem should rediscover same labels
        await client.post_reset()
        setup2 = await client.post_setup(problem)
        assert setup2['derived_labels'] == derived_labels, \
            'After reset, /setup should rediscover the same labels for the same problem'
        print('[5/5] /reset + re-setup OK')

    print('\nAll checks passed.')


if __name__ == '__main__':
    asyncio.run(main())
