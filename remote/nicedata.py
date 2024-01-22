from typing import List, Union, Dict

import time
from threading import Event, Timer, Condition
from remote.util import StoppableThread, NICEInteractor
from queue import Queue, Empty, Full

# temp
import numpy as np

from autorefl.datastruct import DataPoint, MeasurementPoint, Intent, data_attributes

# NICE import
import sys
from remote.nicepath import nicepath
sys.path.append(nicepath)
import nice
import nice.datastream
import nice.core
import nice.writer
from nice.remote import Task

def blocking(func):
    """
    Decorator for blocking functions that only executes the function if the class is not stopped
    """
    def check(self: StoppableThread, *args, **kwargs):
        if self.stopped():
            return
        else:
            return func(self, *args, **kwargs)

    return check

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

        # Signals that a new data point has been acquired
        self.new_data_processed = Event()

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
        self.current_instrument_x = Queue(maxsize=1)
        self.current_instrument_x.put(None)

        # Measurement point queue (frequently flushed and updated after each analysis step)
        self.measurement_queue = Queue()

        # Queue for storing information about current measurement
        self.current_measurement = Queue()

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
            print(record)
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
        self.connect()

        # Subscribe to data stream. All activity is handled in the DataListener.emit callback
        self.api.subscribe('data', DataListener(self.signals))

        # wait until thread is terminated
        self._stop_event.wait()

        # Disconnect
        self.disconnect()
        
class DataQueueListener(StoppableThread):
    """
    Listens to data queue and gets the data
    """

    def __init__(self, signals: Signaller, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signals = signals

        # data repository
        self.data: Dict[int, List[DataPoint]] = {}

    @blocking
    def _handle_data(self):
        """
        Handler for incoming data. Requires that signals.current_measurement and
            signals.data_queue have items in them to get.
        """

        # get desired measurement data
        basedata: MeasurementPoint = self.signals.current_measurement.get()

        # get measurement data
        data: dict = self.signals.data_queue.get()

        print(data)

        # combine into new DataPoint object
        datapoint: DataPoint = self._record_to_datapoint(basedata, data)

        self._get_current_list(basedata.step_id).append(datapoint)

    def run(self):

        # run forever until thread is stopped
        while not self.stopped():

            # wait for new data to come in
            # TODO: if multiple listeners required, a Condition would work better
            self.signals.new_data_acquired.wait()

            print(f'DataQueueListener: new_data_acquired triggered')

            self._handle_data()

            # reset the event
            self.signals.new_data_acquired.clear()
            self.signals.new_data_processed.set()
            print(f'DataQueueListener: resetting new_data_acquired')

    def _get_current_list(self, step_id) -> list:

        if step_id not in self.data.keys():
            self.data[step_id] = []
        
        return self.data[step_id]

    def _record_to_datapoint(self, basedatapoint: MeasurementPoint, data: dict):
        """Converts a NICE counts dictionary to a DataPoint object"""

        datapoint = basedatapoint.base
        basedata = list(datapoint.data)

        # TODO: counting device may be instrument-specific!
        cts = np.array(data['data']['counter.liveROI'], ndmin=1)
        basedata[data_attributes.index('N')] = cts
        datapoint.data = basedata

        # Shouldn't be necessary as interrupted counts will be ignored
        datapoint.t = data['data']['counter.liveTime']

        return datapoint

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

    def __init__(self, api, filePrefix, entry, time, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filePrefix = filePrefix
        self.entry = entry
        self.time = time
        self.api = api
        self.running = False

    def run(self):

        self.running = True

        # Running count
        if not self.stopped():
            self.api.measurement_count(filePrefix=self.filePrefix,
                                        entry=self.entry,
                                        presetTime=self.time)
        self.running = False

    def stop(self):
        super().stop()
        if self.running:
            print('Interrupting count!')
            self.api.terminateCount()

class MeasurementHandler(Task):
    """
    Threaded handler for measurement control
    """

    def __init__(self, signals: Signaller,
                 motors_to_move: List[str],
                 filename: str,
                 use_simulated_data: bool = False,
                 *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        self.signals = signals
        self.active_count = None
        self.api_locked = False
        self.use_simulated_data = use_simulated_data

        self.motors_to_move = motors_to_move + ['counter', 'pointDetector']
        self.filename = filename
        self._filename = filename
        
        # data repository
        self.data: Dict[int, List[DataPoint]] = {}
        self._stop_event = Event()

    def stopped(self):
        return self._stop_event.is_set()

    def clear(self):
        self._stop_event.clear()

    def _get_current_list(self, step_id) -> list:

        if step_id not in self.data.keys():
            self.data[step_id] = []
        
        return self.data[step_id]

    def _produce_random_point(self):

        # for testing only

        from autorefl.instrument import MAGIK

        instr = MAGIK()

        x = np.random.uniform(0.008, 0.25)
        intent = Intent.spec

        base = DataPoint(x, np.random.uniform(0, 2) + 2, 0,
                         (instr.T(x)[0], instr.dT(x)[0], instr.L(x)[0], instr.dL(x)[0],
                          None, instr.intensity(x)[0]),
                         intent=intent)
        pt = MeasurementPoint(1, 1, base, instr.trajectoryData(x, intent))

        base = DataPoint(x, np.random.uniform(0, 2) + 2, 0,
                         (instr.T(x)[0], instr.dT(x)[0], instr.L(x)[0], instr.dL(x)[0],
                          None, instr.intensity(x)[0]),
                         intent=intent)
        pt2 = MeasurementPoint(1, 2, base, instr.trajectoryData(x, intent))

        points = [[pt], [pt2]]

        print(f'Producing random point list {pt}')

        self.signals.measurement_queue.put(points)
        self.signals.measurement_queue_updated.set()
        self.signals.measurement_queue_empty.clear()

    @blocking
    def _measure_queue(self) -> None:

        print('MeasurementHandler: getting queue value')

        # Get the entire list of possible measurements
        complete_list: List[List[MeasurementPoint]] = self.signals.measurement_queue.get()

        # Reset queue update signal; will now run point_lists in complete_list until
        # the measurement queue is updated again
        self.signals.measurement_queue_updated.clear()

        # For each list of points (may include spec, or spec + backp + backm)
        for ptlistnum, point_list in enumerate(complete_list):

            print(f'MeasurementHandler: measuring point list {ptlistnum + 1} of {len(complete_list)}')
            #print(f'MeasurementHandler: queue update? {self.signals.measurement_queue_updated.is_set()}')

            if (not self.stopped()) & (not self.signals.measurement_queue_updated.is_set()):

                # measure all points in point list
                for ptnum, pt in enumerate(point_list):
                    
                    print(f'MeasurementHandler: measuring list {ptlistnum + 1}, point {ptnum + 1} of {len(point_list)}, step ID {pt.step_id}')
                    # TODO: Remove this if statement
                    if not self.use_simulated_data:
                        self._move_count(pt)
                    else:
                        self._get_current_list(pt.step_id).append(pt.base)


                if ptlistnum == 0:
                    # signal that first point list is complete, but only after the first set
                    # this allows the listener to clear the signal
                    self.signals.first_measurement_complete.set()

            else:
                print('MeasurementHandler: Queue has been updated, breaking')
                break

    @blocking
    def _move_count(self, pt: MeasurementPoint) -> None:

        # update current instrument position (for figure of merit calculation)
        try:
            self.signals.current_instrument_x.get_nowait()
        except Empty:
            # current position not defined
            pass
        finally:
            self.signals.current_instrument_x.put(pt.base.x)

        # blocking
        init_time = time.time()
        self.api.move(pt.movements)
        move_time = time.time() - init_time

        # add actual movement time
        pt.base.movet = move_time

        # blocking count call
        self._count(pt)

    @blocking
    def _count(self, pt: MeasurementPoint) -> None:

        print(f'MeasurementHandler: counting {self._filename, pt.base.intent, pt.base.t}')

        # set up active count process; this allows clean exiting in stop() using terminateCount()
        self.active_count = self.api.measurement_count(filePrefix=self._filename,
                        entry=pt.base.intent,
                        presetTime=pt.base.t, wait=False)
        self.api.wait(self.active_count)

        detectorName = self.api.read('counter.countAgainstDetector')
        print(f'MeasurementHandler: reading count value from {detectorName}')
        counts = self.api.read(f'{detectorName}.counts')
        counts = [int(ct) for ct in counts]

        # select subset of counts corresponding to desired detector bank (only for multiDetector)
        if pt.bank is not None:
            strides = self.api.read(f'{detectorName}.strides')
            small_stride = int(strides[0])
            counts = counts[int(pt.bank)::(small_stride + 1)]
            
        livetime = float(self.api.read('counter.liveTime'))

        print(f'MeasurementHandler: got {counts} counts with live time {livetime}')

        self._handle_data(pt, counts, livetime)

    def _handle_data(self, basedatapoint: MeasurementPoint, counts: List[int], livetime: float):
        """
        Handler for incoming data.
        """

        datapoint = basedatapoint.base

        # only update count value if use_simulated_data flag is False
        if not self.use_simulated_data:
            basedata = list(datapoint.data)

            cts = np.array(counts, ndmin=1)
            basedata[data_attributes.index('N')] = cts
            datapoint.data = basedata

            # Shouldn't be necessary as interrupted counts will be ignored
            datapoint.t = livetime

        self._get_current_list(basedatapoint.step_id).append(datapoint)

    def run(self):

        print(f'MeasurementHandler: starting measurement task')

        # start trajectory
        self.api.measurement_start(control_nodes=self.motors_to_move, title='test_traj')

        # blocking: will wait until configuration comes through
        #self.signals.new_trajectory_acquired.wait()
        #scaninfo = self.signals.trajectory_queue.get()

        self._filename = self.filename #scaninfo['data']['trajectoryData.fileName']

        # wait for global start signal to come in
        self.signals.global_start.wait()

        print(f'MeasurementHandler: starting measurement loop with filename {self._filename}')

        #### Waits on measurement queue ####
        while (not self.stopped()):

            # TESTING ONLY: produce and queue up new measurement point
            #self._produce_random_point()

            # TODO: figure out if use of qsize is okay or if mutex lock required
            #with self.signals.measurement_queue.mutex:
            if self.signals.measurement_queue.qsize() == 0:
                # send signal that queue is empty
                self.signals.measurement_queue_empty.set()
            
            # blocking call
            self.signals.measurement_queue_updated.wait()

            self._measure_queue()
            
        self.api.measurement_end()

    def stop(self):
        print('MeasurementHandler: stopping')
        self._stop_event.set()
        #super().stop()

        # does this to achieve instant stopping. Ugly, but prevents having to wait on
        # more than one event. May have downstream effects if this event is ever used for something else
        self.signals.global_start.set()
        self.signals.measurement_queue_updated.set()
        self.signals.first_measurement_complete.set()
        if self.active_count is not None:
            if not self.active_count.isFinished():
                self.api.terminateCount()


if __name__ == '__main__':

    # python -m remote.nicedata

    from autorefl.instrument import MAGIK
    import signal

    instr = MAGIK()

    signaller = Signaller()

    datalistener = DataListener(signaller)
    #nicedata = NiceDataListener(data_queue=data_queue, event_newdata=event_newdata, event_ready=event_ready)
    queuedata = DataQueueListener(signaller)

    measuredata = MeasurementHandler(signaller, motors_to_move=instr.trajectoryMotors(), filename='test')

    # TODO: make part of launcher thread

    def sigint_handler(signal, frame):
        print('KeyboardInterrupt is caught')
        print('Caught KeyboardInterrupt')
        measuredata.stop()
        time.sleep(1)
        queuedata.stop()

    signal.signal(signal.SIGINT, sigint_handler)

    print("launching")
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
#    except KeyboardInterrupt:
        #nicedata.stop()
#        print('Caught KeyboardInterrupt')
#        measuredata.stop()
#        time.sleep(1)
#        queuedata.stop()
    #for _ in range(data_queue.qsize()):
    #    print(data_queue.get())