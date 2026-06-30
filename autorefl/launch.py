import asyncio
import datetime
import copy
import os
import json
import time

import numpy as np

from autorefl.remote.nicedata import NICECampaignTask, Signaller
from autorefl.remote.monitor import SocketServer, buttonhandler, emit_history

from autorefl.autorefl import AutoReflExperiment
from autorefl.calibration import calibrate_intensity


class AutoReflLauncher:
    """Autonomous reflectometry optimization loop (asyncio-native).

    Lifecycle::

        task = NICECampaignTask(instr.trajectoryMotors(), filename)
        launcher = AutoReflLauncher(exp, client, signals, maxtime, pathname,
                                    nice_task=task, nice_api=api)
        await asyncio.gather(
            asyncio.to_thread(api.serve_tasks, task),
            launcher.run(),
        )

    When ``nice_task`` is None the loop still works in simulation — ``measure_step``
    can be overridden directly with any async callable before calling ``run()``.
    """

    def __init__(self,
                 exp: AutoReflExperiment,
                 client,                      # Refl1DClient
                 signals: Signaller,
                 maxtime: float,
                 pathname: str,
                 nice_task: NICECampaignTask | None = None,
                 nice_api=None,
                 intensity_template=None,     # reductus template dict; None → use simulation fallback
                 intensity_files=None) -> None:        # list[FileInfo] from IntensityDatabase
        self.exp = exp
        self.client = client
        self.signals = signals
        self.maxtime = maxtime
        self.pathname = pathname
        self.nice_task = nice_task
        self.nice_api = nice_api
        self.intensity_template = intensity_template
        self.intensity_files = intensity_files
        self._stop = asyncio.Event()

        # Async callable: (points: List[List[MeasurementPoint]]) -> Dict[int, List[DataPoint]]
        if nice_task is not None:
            self.measure_step = nice_task.measure
        else:
            self.measure_step = None

    def stop(self) -> None:
        self._stop.set()
        if self.nice_task is not None:
            self.nice_task.stop()
        # unblock any signal-based wait in case caller holds old-style signals
        self.signals.global_start.set()
        self.signals.measurement_queue_updated.set()
        self.signals.first_measurement_complete.set()

    def _load_data(self, data: dict) -> None:
        """Merge step-keyed DataPoint dict returned by measure_step into exp.steps."""
        for step_id, points in data.items():
            if step_id < len(self.exp.steps):
                self.exp.steps[step_id].points = points

    async def run(self) -> None:
        socket_task = asyncio.create_task(SocketServer().serve())

        # register stop/start button callbacks
        buttonhandler.start_callbacks.append(self.signals.global_start.set)
        buttonhandler.stop_callbacks.append(self.stop)

        await asyncio.sleep(1)
        await self.signals.global_start.wait()

        if self.intensity_template is not None and self.intensity_files:
            print('AutoLauncher: calibrating intensity from scan files')
            await asyncio.to_thread(
                calibrate_intensity, self.exp.instrument, self.intensity_template, self.intensity_files
            )
        else:
            print('AutoLauncher: no intensity calibration provided, using simulation fallback')

        print('AutoLauncher: calculating initial points')
        points, init_qprofs = await self.exp.initial_points(self.client)

        total_t = 0.0
        k = 0
        warm_start = False

        while total_t < self.maxtime and not self._stop.is_set():

            self.exp.add_step([])
            print('AutoLauncher: Step %i, total time so far: %.1f s' % (k, total_t))

            data = await self.measure_step(points)
            self._load_data(data)

            # track latest instrument position for FOM planning
            if self.nice_task is not None and self.nice_task.last_x is not None:
                self.exp.instrument.x = self.nice_task.last_x

            if self._stop.is_set():
                break

            print('AutoLauncher: fitting data')
            await self.exp.fit_step(self.client, warm_start=warm_start)
            warm_start = True

            step = self.exp.steps[-1]
            print('AutoLauncher: final chi-squared:', step.final_chisq)

            # emit posterior R(Q) credible intervals
            await emit_history('autorefl_profiles', self.update_plot_profiles(step.qprofs))
            # emit reduced R(Q) data
            await emit_history('autorefl_data', self.update_plot_data())
            # emit convergence trace (entropy + chisq across all steps so far)
            convergence = [
                {'step': i, 'dH': s.dH, 'dH_marg': s.dH_marg, 'chisq': s.final_chisq}
                for i, s in enumerate(self.exp.steps)
                if s.dH is not None
            ]
            await emit_history('autorefl_convergence', json.dumps(convergence))

            print('AutoLauncher: calculating FOM')
            points = await asyncio.to_thread(self.exp.take_step, allow_repeat=False)

            # emit FOM landscape for the step just completed
            step = self.exp.steps[-1]
            if step.foms is not None:
                fom_data = [
                    {'model': m, 'x': list(self.exp.x[m]), 'fom': list(fom.tolist())}
                    for m, fom in enumerate(step.foms)
                ]
                await emit_history('autorefl_fom', json.dumps(fom_data))

            print('AutoLauncher: saving')
            await asyncio.to_thread(self.exp.save, self.pathname + '/autoexp0.pickle')

            total_t = sum(pt.t + pt.movet for step in self.exp.steps for pt in step.points)
            k += 1

        socket_task.cancel()

    def update_plot_data(self) -> str:
        from autorefl.reduction import reduce
        plotdata = {'data': []}
        for specdata, bkgpdata, bkgmdata in self.exp.get_data():
            refl = reduce(specdata, bkgpdata, bkgmdata)
            if refl is not None:
                plotdata['data'].append({'x': list(refl.x), 'v': list(refl.v), 'dv': list(refl.dv)})
        return json.dumps(plotdata)

    def update_plot_profiles(self, allqprofs) -> str:
        from bumps.plotutil import form_quantiles
        plotdata = {'ci': []}
        for q, qprofs in zip(self.exp.measQ, allqprofs):
            _, ci = form_quantiles(qprofs, [68, 95])
            plotdata['ci'].append({
                'x': list(q),
                '68': {'lower': list(ci[0][0]), 'upper': list(ci[0][1])},
                '95': {'lower': list(ci[1][0]), 'upper': list(ci[1][1])},
            })
        return json.dumps(plotdata)

if __name__ == '__main__':

    # python -m autorefl.launch
    import os
    from autorefl.instrument import MAGIK, CANDOR
    from bumps.cli import load_model
    from autorefl.fit_client import FitClient
    from nice.remote import connect as nice_connect

    instr = MAGIK()

    modelfile = 'example_model/ssblm_d2o.py'
    model = load_model(modelfile)

    bestpars = 'example_model/ssblm_d2o_tosb0.par'
    bestp = np.array([float(line.split(' ')[-1]) for line in open(bestpars, 'r').readlines()]) if bestpars is not None else None

    sel = [10, 11, 12, 13, 14]

    qstep_max = 0.0024
    qmax = 0.25
    qmin = 0.008
    qstep = 0.0005
    dq = np.linspace(qstep, qstep_max, int(np.ceil(2 * (qmax - qmin) / (qstep_max + qstep))))
    measQ = (qmin - qstep) + np.cumsum(dq)

    exp = AutoReflExperiment('test', model, measQ, instr,
                             bestpars=bestp,
                             meas_bkg=3e-6,
                             eta=0.5,
                             npoints=6,
                             select_pars=sel,
                             min_meas_time=20.0,
                             oversampling=5,
                             fit_options={'burn': 1000, 'steps': 100, 'pop': 8})
    if instr.name == 'MAGIK':
        exp.x = exp.measQ
    elif instr.name == 'CANDOR':
        for i, measQ in enumerate(exp.measQ):
            x = []
            overlap = 0.90
            xrng = exp.instrument.qrange2xrange([min(measQ), max(measQ)])
            x.append(xrng[0])
            while x[-1] < xrng[1]:
                curq = exp.instrument.x2q(x[-1])
                curminq, curmaxq = np.min(curq), np.max(curq)
                newrng = exp.instrument.qrange2xrange([curminq + (curmaxq - curminq) * (1 - overlap), max(measQ)])
                x.append(newrng[0])
            x[-1] = xrng[1]
            exp.x[i] = np.array(x)

    fprefix = '%s_eta%0.2f_npoints%i' % (instr.name, exp.eta, exp.npoints)
    fn = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    pathname = fprefix + '_' + fn + '_testauto'
    os.makedirs(pathname, exist_ok=True)

    NICE_HOST = os.environ.get('NICE_HOST', 'localhost')
    NICE_CLIENT = os.environ.get('NICE_CLIENT', 'AutoRefl')

    async def main():
        signals = Signaller()
        client = FitClient(
            host=os.environ.get('AUTOREFL_FIT_HOST', '127.0.0.1'),
            port=int(os.environ.get('AUTOREFL_FIT_PORT', '5100')),
        )

        nice_api = await asyncio.to_thread(nice_connect, NICE_HOST, NICE_CLIENT)
        nice_task = NICECampaignTask(
            motors_to_move=instr.trajectoryMotors(),
            filename=fprefix,
        )

        await client.connect()
        try:
            launcher = AutoReflLauncher(exp, client, signals, maxtime=7200, pathname=pathname,
                                        nice_task=nice_task, nice_api=nice_api)
            await asyncio.gather(
                asyncio.to_thread(nice_api.serve_tasks, nice_task),
                launcher.run(),
            )
        finally:
            await client.disconnect()
            nice_api.end_serve()
            nice_api.disconnect()

    print("launching")
    asyncio.run(main())
