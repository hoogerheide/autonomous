import time
from threading import Event, Timer, Condition
from util import StoppableThread, NICEInteractor
from queue import Queue

# temp
import numpy as np

# NICE import
import sys
from nicepath import nicepath
sys.path.append(nicepath)
import nice.datastream
import nice.core
import nice.writer

class Signaller:
    """
    Container class for all required events, conditions, and queues
    """

    def __init__(self) -> None:

        ### Events

        # Starts the measurement loop (may not be necessary)
        self.global_start = Event()

        # Stops the measurement loop
        self.global_stop = Event()

        # Signals that the measurement queue has been updated
        self.measurement_queue_updated = Event()

        # Signals that a new data point has been acquired
        self.new_data_acquired = Event()

        # Signals that a new trajectory definition has been acquired
        self.new_trajectory_acquired = Event()

        # Signals that the measurement queue is empty (more data needed)
        self.measurement_queue_empty = Event()

        # Signals that an analysis fit has converged (use if fitting process is split)
        self.fit_converged = Event()

        # Signals that the first measurement is complete after updating the measurement queue
        self.first_measurement_complete = Event()

        ### Queues

        # Current instrument position (initialized to None)
        self.current_instrument_x = Queue()
        self.current_instrument_x.put(None)

        # Measurement point queue (frequently flushed and updated after each analysis step)
        self.measurement_queue = Queue()

        # Data queue for temporarily storing new data points
        self.data_queue = Queue()

        # Trajectory queue for storing new trajectory definitions (should not be needed)
        self.trajectory_queue = Queue()

class DataListener(nice.datastream.NiceData):
    """
    Message processor for data stream.
    """

    def __init__(self, signals: Signaller) -> None:
        self.signals = signals

    def emit(self, message: bytes, current=None) -> None:
        #script_api.consolePrint("Hello there!")
        record = nice.writer.stream.loads(nice.writer.util.bytes_to_str(message))
        #script_api.consolePrint(record.keys())
        #print(message.decode()[:250])
        #print(record['command'])
        #print(f'DataListener: event_newdata {self.event_newdata.is_set()}')
        if record['command'] == 'Counts':
            self.signals.data_queue.put(record)
            print(f'DataListener: setting new_data_acquired event')
            self.signals.new_data_acquired.set()
        elif record['command'] == 'Open':
            self.signals.trajectory_queue.put(record)
            self.signals.new_trajectory_acquired.set()
        #elif record['command'] == 'End'
        #self.queue.put(record)

class NiceDataListener(NICEInteractor):
    """
    NICE data stream listener.
    Deprecated. Now part of MeasurementHandler
    """

    def __init__(self, signals: Signaller, host=None, port=None, *args, **kwargs):
        super().__init__(host=host, port=port, *args, **kwargs)
        self.signals = signals

    def run(self):
        # Connect to NICE
        api = nice.connect(**self.nice_connection)

        # Subscribe to data stream. All activity is handled in the DataListener.emit callback
        api.subscribe('data', DataListener(self.signals))

        # wait until thread is terminated
        self._stop_event.wait()

        # clean up
        api.close()
    
class DataQueueListener(StoppableThread):
    """
    Listens to data queue and gets the data
    """

    def __init__(self, signals: Signaller, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signals = signals
        self.data = []

    def run(self):

        # run forever until thread is stopped
        while not self.stopped():

            # wait for new data to come in
            self.signals.new_data_acquired.wait()

            print(f'DataQueueListener: new_data_acquired triggered')

            # event will be triggered upon stop as well
            if not self.stopped():
                #with self.signals.data_queue.mutex:
                for _ in range(self.signals.data_queue.qsize()):
                    record = self.signals.data_queue.get()
                    self.data.append(self._record_to_datapoint(record))
                    print(record)
            
                print(f'DataQueueListener: resetting new_data_acquired')

                # reset the event
                self.signals.new_data_acquired.clear()

            else:
                break

    def _record_to_datapoint(self, record):
        """Converts a NICE dictionary to a DataPoint object"""
        return record

    def stop(self):
        super().stop()

        # does this to achieve instant stopping. Ugly, but prevents having to wait on
        # more than one event
        self.signals.new_data_acquired.set()

# define end states
end_states = [
    nice.api.queue.CommandState.FINISHING,
    nice.api.queue.CommandState.FINISHED,
    nice.api.queue.CommandState.SKIPPED,
]

class StoppableNiceCounter(StoppableThread):

    def __init__(self, api, count_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count_args = count_args
        self.api = api
        self.running = False

    def run(self):

        self.running = True

        # Running count
        if not self.stopped():
            self.api.queue.wait_for(self.api.count(*self.count_args).UUID, end_states)

        self.running = False

    def stop(self):
        super().stop()
        if self.running:
            print('Interrupting count!')
            self.api.terminateCount()

class MeasurementHandler(NICEInteractor):
    """
    Threaded handler for measurement control
    """

    def __init__(self, signals: Signaller,
                 datalistener: DataListener,
                 host=None, port=None,
                 *args, **kwargs) -> None:
        
        super().__init__(host=host, port=port, *args, **kwargs)
        self.signals = signals
        self.counter = StoppableNiceCounter(None, None)
        self.datalistener = datalistener

    def _produce_random_point(self) -> None:

        # for testing only
        pt = [{'movements': ['detectorAngle', str(np.random.uniform(0, 15))],
             'count_time': np.random.uniform(0, 2) + 2,
             'intent': 'specular',
             'x': 1.0
            }]

        print(f'Producing random point {pt}')

        self.signals.measurement_queue.put(pt)
        self.signals.measurement_queue_updated.set()
        self.signals.measurement_queue_empty.clear()

    def run(self):

        # create NICE connection
        api = nice.connect(**self.nice_connection)

        # TODO: Enable for production
        #api.lock()

        # subscribe data listener
        api.subscribe('data', self.datalistener)

        # motor movements
        motors_to_move = ['detectorAngle']
        nmoves = 0

        # start trajectory
        api.startTrajectory(1, motors_to_move, 'test', 'entry', None, 'test_traj', True)

        # TODO: start 3 scans for spec, bkgp, bkgm
        # TODO: can we implement a system to look at existing background data and calculate
        # uncertainties on all putative measurement points? If expected uncertainty in measured
        # data is less than the interpolated uncertainty in the background, then don't bother
        # measuring the background
        api.startScan(1, motors_to_move, 'test', 'entry', None, 'test_traj', True)

        # blocking: will wait until configuration comes through
        self.signals.new_trajectory_acquired.wait()
        scaninfo = self.signals.trajectory_queue.get()

        filename = scaninfo['data']['trajectoryData.fileName']

        # wait for global start signal to come in
        self.signals.global_start.wait()

        print(f'MeasurementHandler: starting measurement loop')

        #### Waits on measurement queue ####
        while not self.stopped():

            # TESTING ONLY: produce and queue up new measurement point
            self._produce_random_point()

            # blocks until new measurement is available, but can be interrupted
            # using the meas_avail event
            #with self.signals.measurement_queue.mutex:
            #    if self.signals.measurement_queue.qsize() == 0:
            #        self.signals.measurement_queue_empty.set()
            
            print(f'Measurement queue update? {self.signals.measurement_queue_updated.is_set()}')
            self.signals.measurement_queue_updated.wait()

            # event will be triggered upon stop as well
            if not self.stopped():

                print('Getting queue value:')

                # Get a single measurement list (may include spec, back+, back-)
                meas_list = self.signals.measurement_queue.get()

                # For each point in the measurement list, measure
                for pt in meas_list:

                    print(f'MeasurementHandler: measuring {pt}')

                    if not self.stopped():

                        # blocking
                        self.signals.current_instrument_x.get()
                        self.signals.current_instrument_x.put(pt['x'])
                        api.queue.wait_for(api.move(pt['movements'], False).UUID, end_states)
                        nmoves += 1

                        # check again for stoppage after blocking call
                        if not self.stopped():

                            print(f'MeasurementHandler: counting')
                            self.counter = StoppableNiceCounter(api, (pt['count_time'], -1, -1, ''))

                            api.startCount(1, motors_to_move, 'test', 'entry', filename, 'test_traj', True)
                            #api.queue.wait_for(api.count(2.1, -1, -1, '').UUID, end_states)
                            self.counter.start()
                            self.counter.join()
                            self.counter.stop()
                            api.endCount(1, motors_to_move, 'test', 'entry', filename, 'test_traj')

                        print(f'MeasurementHandler: resetting event_ready')
                    
                    else:
                        break

                # signal that first measurement is complete (clear this when queue is updated)
                self.signals.first_measurement_complete.set()

            else:
                break

        api.endScan(1, motors_to_move, 'test', 'entry', filename, 'test_traj')
        api.endTrajectory(1, motors_to_move, 'test', 'entry', filename, 'test_traj')

        # TODO: enable for production
        #api.unlock()

        # clean up: note that this does NOT unlock the NICE queue
        api.close()

    def stop(self):
        print('MeasurementHandler: stopping')
        super().stop()

        # does this to achieve instant stopping. Ugly, but prevents having to wait on
        # more than one event. May have downstream effects if this event is ever used for something else
        self.signals.global_start.set()
        self.signals.measurement_queue_updated.set()
        self.signals.new_trajectory_acquired.set()
        self.counter.stop()


if __name__ == '__main__':

    signaller = Signaller()

    datalistener = DataListener(signaller)
    #nicedata = NiceDataListener(data_queue=data_queue, event_newdata=event_newdata, event_ready=event_ready)
    queuedata = DataQueueListener(signaller)

    measuredata = MeasurementHandler(signaller, datalistener)

    # TODO: make part of launcher thread

    try:
        #nicedata.start()
        queuedata.start()
        measuredata.start()
        t = Timer(10, lambda: signaller.global_start.set())
        t.start()
        time.sleep(60)
        measuredata.stop()
        time.sleep(1)
        #nicedata.stop()
        queuedata.stop()

        print(queuedata.data)
    except KeyboardInterrupt:
        #nicedata.stop()
        measuredata.stop()
        time.sleep(1)
        queuedata.stop()
    #for _ in range(data_queue.qsize()):
    #    print(data_queue.get())