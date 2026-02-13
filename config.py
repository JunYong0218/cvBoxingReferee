#配置文件

from multiprocessing import Value, Lock
import threading

# 原始全域變數
ip_addr = "192.168.0.158"
g_width = 1920
g_height = 1080

# ThreadTimer 類別
class ThreadTimer:
    def __init__(self):
        self.timers = [0, 0, 0]  # 四個線程的計時器
        self.lock = threading.Lock()

    def increment(self, thread_id):
        with self.lock:
            self.timers[thread_id] += 1

    def should_wait(self, thread_id):
        with self.lock:
            current = self.timers[thread_id]
            others = self.timers[:thread_id] + self.timers[thread_id+1:]
            return current > min(others) + 2  # 允許2幀的差異

# SharedVars 類別
class SharedVars:
    def __init__(self):
        self.player1_bodyhit = Value('i', 0)
        self.player2_bodyhit = Value('i', 0)
        self.player1_headhit = Value('i', 0)
        self.player2_headhit = Value('i', 0)
        self.model_lock = Lock()
        self.running = Value('i', 1)