import pynvml
import time
import csv

class PowerMonitor:
    def __init__(self, gpu_index=0, interval=0.5, outfile="power_log.csv"):
        self.gpu_index = gpu_index
        self.interval = interval
        self.outfile = outfile
        self.running = False

    def __enter__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        self.running = True
        self.logs = []
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        duration = time.time() - self.start_time
        with open(self.outfile, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "power_watts"])
            for t, p in self.logs:
                writer.writerow([t, p])
        print(f"[Monitor] Duration: {duration:.2f}s, Saved: {self.outfile}")
        pynvml.nvmlShutdown()

    def start(self):
        while self.running:
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            now = time.time() - self.start_time
            self.logs.append((now, power))
            time.sleep(self.interval)

