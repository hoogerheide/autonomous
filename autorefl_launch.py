import datetime
import copy
import os
import argparse
from queue import Empty

import numpy as np

from remote.util import StoppableThread
from remote.nicedata import Signaller

from autorefl.autorefl import AutoReflExperiment

class AutoReflLauncher(StoppableThread):
    """
    Launch autonomous reflectometry calculator in separate thread
    """

    def __init__(self, exp: AutoReflExperiment,
                       signals: Signaller,
                       maxtime: float,
                       cli_args: dict = {'name': 'test'},
                       *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.exp = exp
        self.signals = signals
        self.maxtime = maxtime
        self.measurementhandler = MeasurementHandler(signaller, motors_to_move=exp.instrument.trajectoryMotors(), filename='test')

        fprefix = '%s_eta%0.2f_npoints%i' % (self.exp.instrument.name, exp.eta, exp.npoints)
        fsuffix = '' if cli_args['name'] is None else cli_args['name']

        fn = copy.copy(datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
        self.pathname = fprefix + '_' + fn + '_' + fsuffix
        print(f'Making directory {self.pathname}')
        os.mkdir(self.pathname)

    def run(self):

        # start queue listener
        self.measurementhandler.start()

        # wait for global start
        # TODO: set to a Barrier that all child threads need to cross.
        self.signals.global_start.set()
        self.signals.first_measurement_complete.set()

        # start measurement
        print('AutoLauncher: starting measurement')
        self.exp.add_initial_step()
        self.measurementhandler.data[0] = exp.steps[0].points
        self.exp.fit_step(abort_test=lambda: (self.stopped() | self.signals.measurement_queue_empty.is_set()))
        points = self.exp.take_step(allow_repeat=False)
        total_t = 0.0
        k = 0
        print('AutoLauncher: finished initial step')
        while (total_t < self.maxtime) & (not self.stopped()):
            total_t += self.exp.steps[-1].meastime() + self.exp.steps[-1].movetime()
            k += 1
            print('AutoLauncher: Step: %i, Total time so far: %0.1f' % (k, total_t))

            # Add empty step
            exp.add_step([])
            if k not in self.measurementhandler.data.keys():
                self.measurementhandler.data[k] = []

            print('AutoLauncher: updating queue')
            # wait for measurement to become complete, then add new points
            self.signals.measurement_queue.put(points)
            self.signals.measurement_queue_updated.set()
            self.signals.first_measurement_complete.clear()
            self.signals.measurement_queue_empty.clear()
            
            # need to wait for measurements to be acquired

            print('AutoLauncher: getting new data')
            # wait for new data to have been processed and added
            self.signals.first_measurement_complete.wait()
            
            # populate current step with new data
            print(self.measurementhandler.data)
            self._update_data()

            print('AutoLauncher: fitting data')
            # fit the step. Blocks, but exits on stop or if measurement becomes idle
            self.exp.fit_step(abort_test=lambda: (self.stopped() | self.signals.measurement_queue_empty.is_set()))

            print('AutoLauncher: calculating FOM')
            # update instrument position
            try:
                self.exp.instrument.x = self.signals.current_instrument_x.get_nowait()
            except Empty:
                # current position hasn't changed
                pass

            # calculate figure of merit and identify points for next step
            points = self.exp.take_step(allow_repeat=False)

            print('AutoLauncher: saving')
            self.exp.save(self.pathname + '/autoexp0.pickle')

        self.measurementhandler.stop()

    def _update_data(self) -> None:

        for i, step in enumerate(self.exp.steps):
            step.points = self.measurementhandler.data[i]

    def stop(self):
        print('AutoReflHandler: stopping')
        super().stop()
        time.sleep(1)
        self.measurementhandler.stop()

if __name__ == '__main__':

    # python -m autorefl_launch
    import time

    from remote.nicedata import MeasurementHandler
    from autorefl.instrument import MAGIK
    from bumps.cli import load_model

    instr = MAGIK()

    signaller = Signaller()
    #measuredata = MeasurementHandler(signaller, motors_to_move=instr.trajectoryMotors(), filename='test')

    modelfile = 'example_model/ssblm_d2o.py'
    model = load_model(modelfile)

#    bestpars = args.pars
#    bestp = np.array([float(line.split(' ')[-1]) for line in open(bestpars, 'r').readlines()]) if bestpars is not None else None

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

    exp = AutoReflExperiment('test', model, measQ, instr, eta=0.5, npoints=6, select_pars=sel, fit_options={'burn': 50, 'steps': 10, 'pop': 2})
    exp.x = exp.measQ

    autolauncher = AutoReflLauncher(exp, signaller, 1000, cli_args={'name': 'testauto'})

    # TODO: make part of launcher thread

    def sigint_handler(signal, frame):
        print('KeyboardInterrupt is caught')
        print('Caught KeyboardInterrupt')
        autolauncher.stop()
        #queuedata.stop()

    #signal.signal(signal.SIGINT, sigint_handler)

    print("launching")
    try:
        #nicedata.start()
        autolauncher.start()
        #t = Timer(10, lambda: signaller.global_start.set())
        #t.start()
        #time.sleep(600)
        #print(autolauncher.measurementhandler.data)
        autolauncher.join()
        #time.sleep(1)
    except KeyboardInterrupt:
        print('Caught KeyboardInterrupt')
        autolauncher.stop()
    #for _ in range(data_queue.qsize()):
    #    print(data_queue.get())
