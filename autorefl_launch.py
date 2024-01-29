import datetime
import copy
import os
import sys
import time
import json
from queue import Empty
from threading import Event

import numpy as np

from remote.nicepath import nicepath
#sys.path.append(nicepath)
import nice.remote

from remote.util import StoppableThread
from remote.nicedata import Signaller, blocking, MeasurementThread, NICEMeasurementThread, NICEMeasurementDevice, SimMeasurementDevice
from remote.monitor import SocketMonitor, SocketServer, QueueMonitor, buttonhandler
from bumps.fitters import ConsoleMonitor

from autorefl.autorefl import AutoReflExperiment

class KeyboardInput(StoppableThread):

    def __init__(self, signals: Signaller, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.signals = signals

    def run(self):
        while not self.stopped():
            cmd = input()

            if cmd == 'stop':
                break
            elif cmd == 'stopfit':
                self.signals.measurement_queue_empty.set()
            else:
                print(f'{cmd} not recognized')

class AutoReflLauncher(StoppableThread):
    """
    Launch autonomous reflectometry calculator in separate thread
    """

    def __init__(self, exp: AutoReflExperiment,
                       signals: Signaller,
                       maxtime: float,
                       use_simulated_data: bool = False,
                       cli_args: dict = {'name': 'test'},
                       *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.exp = exp
        self.signals = signals
        self.maxtime = maxtime
        if False:
            self.measurementhandler = NICEMeasurementThread(
                task=NICEMeasurementDevice(signals,
                                        motors_to_move=exp.instrument.trajectoryMotors(),
                                        filename='test',
                                        use_simulated_data=use_simulated_data),
                host='localhost',
                name='AutoRefl'
            )
        else:
            self.measurementhandler = MeasurementThread(SimMeasurementDevice(signals))
        
        fprefix = '%s_eta%0.2f_npoints%i' % (self.exp.instrument.name, exp.eta, exp.npoints)
        fsuffix = '' if cli_args['name'] is None else cli_args['name']

        fn = copy.copy(datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
        self.pathname = fprefix + '_' + fn + '_' + fsuffix
        print(f'Making directory {self.pathname}')
        os.mkdir(self.pathname)

    def run(self):

        socketserver = SocketServer()
        socketserver.start()
        queuemonitor = QueueMonitor(self.signals.measurement_queue, socketserver.inqueue)

        # register callbacks
        self.measurementhandler.register_publish_callback(lambda data: socketserver.write(('set_current_measurement', data)))
        self.measurementhandler.register_publish_callback(lambda data: self._update_data() if data is not None else None)
        self.measurementhandler.register_publish_callback(lambda data: socketserver.write(('update_plot', self.update_plot_data())))
        self.measurementhandler.start()

        # register api callbacks
        buttonhandler.start_callbacks.append(self.signals.global_start.set)
        buttonhandler.stop_callbacks.append(self.stop)
        buttonhandler.terminate_count_callbacks.append(self.measurementhandler.task.terminate_count)

#        print(self.measurementhandler.name)
#        t1 = threading.Thread(target=lambda: api.run_task(self.measurementhandler), daemon=True)
#        t1.start()
        
        # clear fit output
        time.sleep(1)
        socketserver.write(('clear', None))

        # wait for global start
        # TODO: set to a Barrier that all child threads need to cross.
        self.signals.global_start.wait()
        self.signals.first_measurement_complete.set()

        # start measurement
        print('AutoLauncher: starting measurement')
        socketserver.write(('fit_update', 'Calculating initial points'))
        print('AutoLauncher: calculating initial points')
        points, init_qprofs = self.exp.initial_points()
        socketserver.write(('update_plot', self.update_plot_profiles(init_qprofs)))
        total_t = 0.0
        k = 0

        while (total_t < self.maxtime) & (not self.stopped()):

            # Add empty step
            self.exp.add_step([])

            socketserver.write(('new_step', {'step': k, 'text': 'Step: %i, Total time so far: %0.1f' % (k, total_t)}))
            print('AutoLauncher: Step: %i, Total time so far: %0.1f' % (k, total_t))

            print('AutoLauncher: updating queue')
            # wait for measurement to become complete, then add new points
            self.signals.measurement_queue.put(points)
            queuemonitor.update()
            self.signals.measurement_queue_updated.set()
            self.signals.first_measurement_complete.clear()
            self.signals.measurement_queue_empty.clear()
            
            # need to wait for measurements to be acquired
            # wait for new data to have been processed and added
            self.signals.first_measurement_complete.wait()

            self._update_data()     # blocking, can be slow

            socketserver.write(('update_plot', self.update_plot_data()))

            # start fitting
            if not self.stopped():
                print('AutoLauncher: fitting data')
                socketserver.write(('fit_update', 'Initializing fit...'))
                # fit the step. Blocks, but exits on stop or if measurement becomes idle
                stop_fit_criterion = lambda: (self.stopped() | self.signals.measurement_queue_empty.is_set())
                #stop_fit_criterion = lambda: self.stopped()
                #monitors = [ConsoleMonitor(self.exp.problem), SocketMonitor(self.exp.problem, socketserver.inqueue)]
                monitors = [SocketMonitor(self.exp.problem, socketserver.inqueue)]
                self.exp.fit_step(abort_test=stop_fit_criterion, monitors=monitors)

                socketserver.write(('fit_update', 'Final chi-squared: '+ self.exp.steps[-1].final_chisq))
                socketserver.write(('update_plot', self.update_plot_profiles(self.exp.steps[-1].qprofs)))

                if not self.stopped():
                    print('AutoLauncher: calculating FOM')
                    # update instrument position
                    try:
                        self.exp.instrument.x = self.signals.current_instrument_x.get_nowait()
                    except Empty:
                        # current position hasn't changed
                        pass

                    # calculate figure of merit and identify points for next step
                    # blocking but usually not slow so not decorated here
                    points = self.exp.take_step(allow_repeat=False)

            print('AutoLauncher: saving')
            self.exp.save(self.pathname + '/autoexp0.pickle')

            # update total time and step number
            
            total_t = sum(pt.t + pt.movet for step in exp.steps for pt in step.points)
            k += 1


        # also signals measurementhandler to stop
        socketserver.stop()
        self.stop()

    @blocking
    def _update_data(self) -> None:
        print('AutoLauncher: getting new data')
        
        # populate current step with new data
        #print(self.measurementhandler.data)

        data = self.measurementhandler.get_data()
        for i, step in enumerate(self.exp.steps):
            step.points = data.get(i, [])

    def update_plot_data(self) -> str:

        from autorefl.reduction import reduce

        plotdata = {'data': []}

        modeldata = self.exp.get_data()
        for specdata, bkgpdata, bkgmdata in modeldata:
            refl = reduce(specdata, bkgpdata, bkgmdata)
            if refl is not None:
                plotdata['data'].append({'x': list(refl.x),
                                 'v': list(refl.v),
                                 'dv': list(refl.dv),
                            })

        return json.dumps(plotdata)

    def update_plot_profiles(self, allqprofs) -> str:

        from bumps.plotutil import form_quantiles

        plotdata = {'ci': []}
        for q, qprofs in zip(self.exp.measQ, allqprofs):
            _, ci = form_quantiles(qprofs, [68, 95])

            plotdata['ci'].append({'x': list(q),
                                '68': {'lower': list(ci[0][0]),
                                        'upper': list(ci[0][1])},
                                '95': {'lower': list(ci[1][0]),
                                        'upper': list(ci[1][1])},
                            })

        return json.dumps(plotdata)

    def stop(self):
        print('AutoReflHandler: stopping')
        super().stop()
        self.signals.global_start.set()
        self.measurementhandler.stop()

if __name__ == '__main__':

    # python -m autorefl_launch
    from autorefl.instrument import MAGIK, CANDOR
    from bumps.cli import load_model

    instr = MAGIK()

    signaller = Signaller()
    #measuredata = MeasurementHandler(signaller, motors_to_move=instr.trajectoryMotors(), filename='test')

    modelfile = 'example_model/ssblm_d2o.py'
    model = load_model(modelfile)

    bestpars = 'example_model/ssblm_d2o_tosb0.par'
    bestp = np.array([float(line.split(' ')[-1]) for line in open(bestpars, 'r').readlines()]) if bestpars is not None else None

    # measurement background
 #   meas_bkg = args.meas_bkg if args.meas_bkg is not None else np.full(len(list(model.models)), 1e-5)

    # condition selection array
  #  sel = np.array(args.sel) if args.sel is not None else None
    sel = [10, 11, 12, 13, 14]

    # set measQ
    #qstep_max = args.qstep if args.qstep_max is None else args.qstep_max
    qstep_max = 0.0024
    qmax = 0.25
    qmin=0.008
    qstep = 0.0005
    #dq = np.linspace(args.qstep, qstep_max, int(np.ceil(2 * (args.qmax - args.qmin) / (qstep_max + args.qstep))))
    dq = np.linspace(qstep, qstep_max, int(np.ceil(2 * (qmax - qmin) / (qstep_max + qstep))))
    measQ = (qmin-qstep) + np.cumsum(dq)
    #measQ = [m.fitness.probe.Q for m in model.models]

    exp = AutoReflExperiment('test', model, measQ, instr,
                             bestpars=bestp,
                             meas_bkg=3e-6,
                             eta=0.5,
                             npoints=6,
                             select_pars=sel,
                             min_meas_time=20.0,
                             oversampling=5,
                             fit_options={'burn': 1000,
                                          'steps': 100,
                                          'pop': 8})
    if instr.name == 'MAGIK':
        exp.x = exp.measQ
    elif instr.name == 'CANDOR':
        for i, measQ in enumerate(exp.measQ):
            x = list()
            overlap = 0.90
            xrng = exp.instrument.qrange2xrange([min(measQ), max(measQ)])
            x.append(xrng[0])
            while x[-1] < xrng[1]:
                curq = exp.instrument.x2q(x[-1])
                curminq, curmaxq = np.min(curq), np.max(curq)
                newrng = exp.instrument.qrange2xrange([curminq + (curmaxq - curminq) * (1 - overlap), max(measQ)])
                x.append(newrng[0])
            x[-1] = xrng[1]
            x = np.array(x)
            exp.x[i] = x


    autolauncher = AutoReflLauncher(exp, signaller, 7200, use_simulated_data=True, cli_args={'name': 'testauto'})
    kinput = KeyboardInput(signaller)

    print("launching")
    autolauncher.start()
    #kinput.start()
    #kinput.join()
    # wait for stop signal from keyboard (just type "stop")
    #print('stop signal issued via keyboard')
    #autolauncher.stop()
    autolauncher.join()
