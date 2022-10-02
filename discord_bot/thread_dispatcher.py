from queue import Queue
import threading
import time
import asyncio

class ThreadDispatcher:
    def __init__(self, daemon=True, name=None, mode="threading"):
        self.mode = mode

        self.action_queue = Queue()

        if mode == "threading":
            self.thread = threading.Thread(target=self.run)
            self.thread.daemon = daemon


            if name is not None:
                self.thread.name = name
        elif mode == "asyncio":
            self.loop = asyncio.new_event_loop()
            #self.action_queue = asyncio.Queue()


        self.running = False
        

    def start(self):
        self.running = True

        if self.mode == "threading":
            self.thread.start()
        elif self.mode == "asyncio":
            self.loop.create_task(self.run_asyncio())

    def run(self):
        while self.running:
            if not self.action_queue.empty():
                action = self.action_queue.get()
                action()
            else:
                time.sleep(0.5)

    async def run_asyncio(self):
        while self.running:
            if not self.action_queue.empty():
                action = self.action_queue.get()
                action()
            else:
                time.sleep(0.5)

    def invoke(self, func):
        self.action_queue.put(func)
