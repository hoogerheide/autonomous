
from threading import Thread, Event

import sys
from remote.nicepath import nicepath
sys.path.append(nicepath)
import nice

class StoppableThread(Thread):
    """
    Stoppable thread. From
    https://stackoverflow.com/questions/323972/is-there-any-way-to-kill-a-thread
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._stop_event = Event()

    def stop(self):
        self._stop_event.set()
    
    def stopped(self):
        return self._stop_event.is_set()

    def clear(self):
        self._stop_event.clear()

class NICEInteractor(StoppableThread):
    """
    Stoppable threaded class that sets up a NICE connection
    """

    def __init__(self, host=None, port=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if (host is not None) & (port is not None):
            self.nice_connection = {'host': host, 'port': port}
        else:
            self.nice_connection = {}

    def connect(self, lock=False):

        # create NICE connection
        self.api = nice.connect(**self.nice_connection)

        # TODO: Enable for production
        if lock:
            self.api.lock()
            self.api_locked = True

    def disconnect(self):

        if self.api_locked:
            self.api.unlock()
        
        self.api.close()
