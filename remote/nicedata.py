import time
from threading import Event, Timer
from util import StoppableThread, NICEInteractor
from queue import Queue

# temp
import numpy as np

import sys
from nicepath import nicepath
sys.path.append(nicepath)
import nice.datastream
import nice.core
import nice.writer

class DataListener(nice.datastream.NiceData):
    """
    Message processor for data stream.
    """

    def __init__(self, queue: Queue, event_newdata: Event, event_ready: Event) -> None:
        self.queue = queue
        self.event_newdata = event_newdata
        self.event_ready = event_ready

    def emit(self, message: bytes, current=None) -> None:
        #script_api.consolePrint("Hello there!")
        record = nice.writer.stream.loads(nice.writer.util.bytes_to_str(message))
        #script_api.consolePrint(record.keys())
        #print(message.decode()[:250])
        #print(record['command'])
        #print(f'DataListener: event_newdata {self.event_newdata.is_set()}')
        if record['command'] == 'Counts':
            self.queue.put(record)
            print(f'DataListener: setting event_newdata')
            self.event_newdata.set()
            self.event_ready.set()
        elif record['command'] == 'Open':
            self.queue.put(record)
        #elif record['command'] == 'End'
        #self.queue.put(record)

class NiceDataListener(NICEInteractor):
    """
    NICE data stream listener.
    Deprecated. Now part of MeasurementHandler
    """

    def __init__(self, data_queue: Queue, event_newdata: Event, event_ready: Event, host=None, port=None, *args, **kwargs):
        super().__init__(host=host, port=port, *args, **kwargs)
        self.data_queue = data_queue
        self.event_newdata = event_newdata
        self.event_ready = event_ready

    def run(self):
        # Connect to NICE
        api = nice.connect(**self.nice_connection)

        # Subscribe to data stream. All activity is handled in the DataListener.emit callback
        api.subscribe('data', DataListener(self.data_queue, self.event_newdata, self.event_ready))

        # wait until thread is terminated
        self._stop_event.wait()

        # clean up
        api.close()
    
class DataQueueListener(StoppableThread):
    """
    Listens to data queue and gets the data
    """

    def __init__(self, data_queue: Queue, event_newdata: Event, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_queue = data_queue
        self.event_newdata = event_newdata
        self.data = []

    def run(self):

        # run forever until thread is stopped
        while not self.stopped():

            # wait for new data to come in
            self.event_newdata.wait()

            print(f'DataQueueListener: event_newdata triggered')

            # event will be triggered upon stop as well
            if not self.stopped():
                # TODO: qsize is not reliable! Need to lock before getting?
                # Could also use try, get_nowait, except
                for _ in range(data_queue.qsize()):
                    record = data_queue.get()
                    self.data.append(record)
                    print(record)
                
                print(f'DataQueueListener: resetting event_newdata')

                # reset the event
                self.event_newdata.clear()

            else:
                break

    def stop(self):
        super().stop()

        # does this to achieve instant stopping. Ugly, but prevents having to wait on
        # more than one event
        self.event_newdata.set()

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

    def __init__(self, data_queue: Queue,
                 measure_queue: Queue,
                 event_ready: Event,
                 event_meas_avail: Event,
                 datalistener: DataListener,
                 host=None, port=None,
                 *args, **kwargs) -> None:
        
        super().__init__(host=host, port=port, *args, **kwargs)
        self.data_queue = data_queue
        self.measure_queue = measure_queue
        self.event_ready = event_ready
        self.event_meas_avail = event_meas_avail
        self.counter = StoppableNiceCounter(None, None)
        self.datalistener = datalistener

    def _produce_random_point(self) -> None:

        # for testing only

        self.measure_queue.put(np.random.uniform(0, 15))
        self.event_meas_avail.set()

    def run(self):

        # create NICE connection
        api = nice.connect(**self.nice_connection)

        # TODO: Enable for production
        #api.lock()

        # subscribe data listener
        api.subscribe('data', self.datalistener)

        # start trajectory
        api.startTrajectory(1, ['detectorAngle'], 'test', 'entry', None, 'test_traj', True)

        # TODO: start 3 scans for spec, bkgp, bkgm
        # TODO: can we implement a system to look at existing background data and calculate
        # uncertainties on all putative measurement points? If expected uncertainty in measured
        # data is less than the interpolated uncertainty in the background, then don't bother
        # measuring the background
        api.startScan(1, ['detectorAngle'], 'test', 'entry', None, 'test_traj', True)

        # blocking: will wait until configuration comes through
        scaninfo = self.data_queue.get()

        filename = scaninfo['data']['trajectoryData.fileName']

        # run forever until thread is stopped
        while not self.stopped():

            # wait for ready signal to come in
            self.event_ready.wait()

            print(f'MeasurementHandler: event_ready triggered')

            # event will be triggered upon stop as well
            if not self.stopped():
                # produce and queue up new measurement point
                self._produce_random_point()

                # blocks until new measurement is available, but can be interrupted
                # using the meas_avail event
                self.event_meas_avail.wait()
                da = self.measure_queue.get()
                meas_time = 2.1

                print(f'MeasurementHandler: moving to {da}')

                # blocking
                api.queue.wait_for(api.move(['detectorAngle', str(da)], False).UUID, end_states)

                # check again for stoppage after blocking call
                if not self.stopped():

                    print(f'MeasurementHandler: counting')
                    self.counter = StoppableNiceCounter(api, (meas_time, -1, -1, ''))

                    api.startCount(1, ['detectorAngle'], 'test', 'entry', filename, 'test_traj', True)
                    #api.queue.wait_for(api.count(2.1, -1, -1, '').UUID, end_states)
                    self.counter.start()
                    self.counter.join()
                    self.counter.stop()
                    api.endCount(1, ['detectorAngle'], 'test', 'entry', filename, 'test_traj')

                    print(f'MeasurementHandler: resetting event_ready')

                    # reset the event
                    self.event_ready.clear()

            else:
                break

        api.endScan(1, ['detectorAngle'], 'test', 'entry', filename, 'test_traj')
        api.endTrajectory(1, ['detectorAngle'], 'test', 'entry', filename, 'test_traj')

        # TODO: enable for production
        #api.unlock()

        # clean up: note that this does NOT unlock the NICE queue
        api.close()

    def stop(self):
        print('MeasurementHandler: stopping')
        super().stop()

        # does this to achieve instant stopping. Ugly, but prevents having to wait on
        # more than one event. May have downstream effects if this event is ever used for something else
        self.event_ready.set()
        self.event_meas_avail.set()
        self.counter.stop()


if __name__ == '__main__':

    data_queue = Queue()
    event_newdata = Event()
    event_ready = Event()

    datalistener = DataListener(data_queue, event_newdata, event_ready)
    #nicedata = NiceDataListener(data_queue=data_queue, event_newdata=event_newdata, event_ready=event_ready)
    queuedata = DataQueueListener(data_queue, event_newdata)

    measure_queue = Queue()
    event_meas_avail = Event()
    measuredata = MeasurementHandler(data_queue, measure_queue, event_ready, event_meas_avail, datalistener)

    try:
        #nicedata.start()
        queuedata.start()
        measuredata.start()
        t = Timer(5, lambda: event_ready.set())
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