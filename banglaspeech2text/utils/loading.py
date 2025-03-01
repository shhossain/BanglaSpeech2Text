import sys
import threading
import time
import itertools

class LoadingIndicator:
    def __init__(self, message="Processing"):
        self.message = message
        self.spinner = itertools.cycle(['-', '/', '|', '\\'])
        self.running = False
        self.spinner_thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self.running = True
        self.spinner_thread = threading.Thread(target=self._spin)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()

    def stop(self):
        self.running = False
        if self.spinner_thread:
            self.spinner_thread.join()
        sys.stdout.write('\r' + ' ' * (len(self.message) + 10) + '\r')
        sys.stdout.flush()

    def _spin(self):
        while self.running:
            sys.stdout.write('\r' + self.message + ' ' + next(self.spinner))
            sys.stdout.flush()
            time.sleep(0.1)
