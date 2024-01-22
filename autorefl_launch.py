import datetime
import copy
import os
import sys
import time
from queue import Empty

import numpy as np

from remote.nicepath import nicepath
sys.path.append(nicepath)
import nice.remote as nice_remote

from remote.util import StoppableThread
from remote.nicedata import Signaller, blocking, MeasurementHandler
from remote.monitor import SocketMonitor, SocketServer, QueueMonitor
from bumps.fitters import ConsoleMonitor

from autorefl.autorefl import AutoReflExperiment

class KeyboardInput(StoppableThread):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def run(self):
        while not self.stopped():
            cmd = input()

            if cmd == 'stop':
                break
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
        self.measurementhandler = MeasurementHandler(signals, motors_to_move=exp.instrument.trajectoryMotors(), filename='test', use_simulated_data=use_simulated_data, name='MeasurementHandler')

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
        #fitmonitor = SocketMonitor(socketserver.inqueue)

        api = nice_remote.connect('localhost', 'AutoRefl')
        api.run_task(self.measurementhandler, wait=False)
#        print(self.measurementhandler.name)
#        t1 = threading.Thread(target=lambda: api.run_task(self.measurementhandler), daemon=True)
#        t1.start()

        # wait for global start
        # TODO: set to a Barrier that all child threads need to cross.
        self.signals.global_start.set()
        self.signals.first_measurement_complete.set()

        # clear fit output
        time.sleep(1)
        socketserver.write(('clear', None))

        # start measurement
        print('AutoLauncher: starting measurement')
        socketserver.write(('fit_update', 'Calculating initial points'))
        print('AutoLauncher: calculating initial points')
        points = self.exp.initial_points()
        total_t = 0.0
        k = 0

        while (total_t < self.maxtime) & (not self.stopped()):

            # Add empty step
            self.exp.add_step([])

            socketserver.write(('fit_update', 'Step: %i, Total time so far: %0.1f' % (k, total_t)))
            print('AutoLauncher: Step: %i, Total time so far: %0.1f' % (k, total_t))

            # create data placeholder (TODO: replace with set_default)
            if k not in self.measurementhandler.data.keys():
                self.measurementhandler.data[k] = []

            print('AutoLauncher: updating queue')
            # wait for measurement to become complete, then add new points
            self.signals.measurement_queue.put(points)
            queuemonitor.update()
            self.signals.measurement_queue_updated.set()
            self.signals.first_measurement_complete.clear()
            self.signals.measurement_queue_empty.clear()
            
            # need to wait for measurements to be acquired
            self._update_data()     # blocking, can be slow

            # start fitting
            if not self.stopped():
                print('AutoLauncher: fitting data')
                # fit the step. Blocks, but exits on stop or if measurement becomes idle
                #stop_fit_criterion = lambda: (self.stopped() | self.signals.measurement_queue_empty.is_set())
                stop_fit_criterion = lambda: self.stopped()
                #monitor = SocketMonitor(socketserver.inqueue)
                monitor = None
                self.exp.fit_step(abort_test=stop_fit_criterion, monitor=monitor)

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
            total_t += self.exp.steps[-1].meastime() + self.exp.steps[-1].movetime()
            k += 1


        # also signals measurementhandler to stop
        api.end_serve()
        api.disconnect()
        socketserver.stop()
        self.stop()

    @blocking
    def _update_data(self) -> None:
        print('AutoLauncher: getting new data')
        # wait for new data to have been processed and added
        self.signals.first_measurement_complete.wait()
        
        # populate current step with new data
        #print(self.measurementhandler.data)

        for i, step in enumerate(self.exp.steps):
            step.points = self.measurementhandler.data[i]

    def stop(self):
        print('AutoReflHandler: stopping')
        super().stop()

        # terminates count, which will update data
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

    exp = AutoReflExperiment('test', model, measQ, instr, bestpars=bestp, meas_bkg=3e-6, eta=0.5, npoints=6, select_pars=sel, min_meas_time=10.0, fit_options={'burn': 1000, 'steps': 500, 'pop': 8})
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
    kinput = KeyboardInput()

    print("launching")
    autolauncher.start()
    kinput.start()
    kinput.join()
    # wait for stop signal from keyboard (just type "stop")
    print('stop signal issued via keyboard')
    autolauncher.stop()
    autolauncher.join()
