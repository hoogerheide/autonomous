from typing import List

import time
import asyncio
from aiohttp import web
import socketio
from queue import Queue
from bumps.monitor import TimedUpdate
from bumps.fitproblem import nllf_scale
from bumps.formatnum import format_uncertainty
from remote.nicedata import Signaller
from remote.util import StoppableThread
from autorefl.datastruct import MeasurementPoint

sio = socketio.AsyncServer(async_mode='aiohttp')
app = web.Application()
sio.attach(app)
socketlock: asyncio.Lock = asyncio.Lock()
sockethistory: list = []

async def hello(request):
    #return web.Response(text="Hello, world")
    return web.FileResponse('remote/socketpage.html')

app.add_routes([web.get('/', hello)])
app.add_routes([web.static('/lib', 'remote/lib')])

@sio.on('connect')
async def dump_history(sid, data):
    print(f'New connection from {sid}')
    async with socketlock:
        for item in sockethistory:
            #print(item)
            await sio.emit(*item, to=sid)

class SocketServer(StoppableThread):

    def __init__(self, *args, host='localhost', port=5012, **kwargs):
        super().__init__(*args, **kwargs)
        self.host = host
        self.port = port
        self.inqueue: Queue = Queue()

    def run(self):

        asyncio.run(self.serve())

    async def serve(self):

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        qlisten = asyncio.create_task(self.emit_queue())

        await asyncio.to_thread(self._stop_event.wait)
        self.inqueue.put((None, None), block=False)
        await asyncio.gather(runner.cleanup(),
                             qlisten)

    async def emit_queue(self):

        while not self.stopped():
            event, data = await asyncio.to_thread(self.inqueue.get)
            print(event, data)
            if event is not None:
                async with socketlock:
                    await sio.emit(event, data)
                    sockethistory.append((event, data))

    def write(self, data):
        self.inqueue.put(data)

    def flush(self):
        pass

class SocketMonitor(TimedUpdate):

    def __init__(self, problem, queue: Queue, progress=1, improvement=60) -> None:
        super().__init__(progress=progress, improvement=improvement)

        self.queue = queue
        self.problem = problem

    def config_history(self, history):
        history.requires(time=1, value=1, point=1, step=1)

    #def __call__(self, history):
    #    record = {'step': history.step[0], 'value': history.value[0]}
    #    self.queue.put(('fit_update', record))
    def show_improvement(self, history):
        pass
        #record = f'step {history.step[0]} cost {history.value[0]:0.4f}'
        #self.queue.put(('fit_update', record))

    def show_progress(self, history):
        scale, err = nllf_scale(self.problem)
        chisq = format_uncertainty(scale*history.value[0], err)
        record = f'step {history.step[0]} cost {chisq}'
        self.queue.put(('fit_update', record))

class QueueMonitor:

    def __init__(self, data_queue: Queue, monitor_queue: Queue) -> None:
        
        self.data_queue = data_queue
        self.monitor_queue = monitor_queue

    def update(self):

        data: List[List[List[MeasurementPoint]]] = self.data_queue.queue
        queue_data = []
        for plistlist in data:
            for plist in plistlist:
                for i, pt in enumerate(plist):
                    queue_item = {'step_id': pt.step_id,
                                  'point_id': pt.point_id,
                                  'point_index': i,
                                  'model': pt.base.model,
                                  'x': f'{pt.base.x:0.4f}',
                                  'intent': pt.base.intent,
                                  'time': f'{pt.base.t:0.1f}'
                                  }
                    queue_data.append(queue_item)

        self.monitor_queue.put(('queue_update', queue_data))


def update_measurement_queue(signals: Signaller):
    
    #print('queue_update', {'queue': [asdict(pt) for ptlistlist in signals.measurement_queue.queue for ptlist in ptlistlist for pt in ptlist]})
    fut = asyncio.run_coroutine_threadsafe(sio.emit('queue_update', {'queue': signals.measurement_queue.queue}), asyncio.get_event_loop())
    fut.result()

if __name__=='__main__':

    signals = Signaller()
    #signals.measurement_queue.put('hello')
    s = SocketServer(daemon=False)
    s.start()
    time.sleep(10)
    """if False:
        s.inqueue.put(('fit_update', 'hello'))
        time.sleep(10)
        s.inqueue.put(('fit_update', 'hello2'))
        time.sleep(10)
        s.inqueue.put(('fit_update', 'hello3'))
    else:
        for i in range(200):
            s.inqueue.put(('fit_update', i))
            time.sleep(0.2)"""
    s.stop()
    s.join()
