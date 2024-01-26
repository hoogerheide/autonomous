
from threading import Thread, Event

import sys

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