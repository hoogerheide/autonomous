
from threading import Thread, Event

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