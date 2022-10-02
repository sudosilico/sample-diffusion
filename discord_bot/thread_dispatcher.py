from queue import Queue
import threading
import time


class ThreadDispatcher:
    def __init__(self, daemon=True, name=None):
        self.thread = threading.Thread(target=self.run)
        self.action_queue = Queue()

        if name is not None:
            self.thread.name = name

        self.running = False
        self.thread.daemon = daemon

    def start(self):
        self.running = True
        self.thread.start()

    def run(self):
        while self.running:
            if not self.action_queue.empty():
                action = self.action_queue.get()
                action()
            else:
                time.sleep(0.5)

    def invoke(self, func):
        self.action_queue.put(func)
