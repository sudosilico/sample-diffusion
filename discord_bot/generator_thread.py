import json
from queue import Queue
import threading
import time


class GeneratorThread:
    def __init__(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.running = False
        self.request_queue = Queue()

    def start(self):
        self.running = True
        self.thread.start()

    def add_request(self, request):
        print(f"Adding request to queue: {request}")
        self.request_queue.put(request)

    def run(self):
        while self.running:
            if not self.request_queue.empty():
                request_count = self.request_queue.qsize()
                print(f"There are currently {request_count} requests in the queue.")

                # request = self.request_queue.get()
                # print(f"Request: {request}")

            time.sleep(1)
