"""ctypes wrapper for the native build of w4.c (4 worker threads; ctypes releases the GIL)."""
import ctypes, subprocess, numpy as np
from concurrent.futures import ThreadPoolExecutor

def build(src, out):
    subprocess.run(['gcc', '-O3', '-msse4.1', '-ffast-math', '-shared', '-fPIC', '-o', out, src], check=True)

class W4:
    def __init__(self, so, threads=4):
        self.lib = L = ctypes.CDLL(so); self.threads = threads
        L.load.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        L.set_io.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        L.setup_thread.argtypes = [ctypes.c_int, ctypes.c_int]
        L.run_job.argtypes = [ctypes.c_int] * 3; L.take_job.restype = ctypes.c_int
        self.pool = ThreadPoolExecutor(threads); self.m = {}; self.keep = []
    def load(self, slot, data: bytes, max_jobs):
        L = self.lib; buf = ctypes.create_string_buffer(data, len(data) + 64); self.keep.append(buf)
        addr = (ctypes.addressof(buf) + 63) & ~63          # 64-byte aligned copy
        ctypes.memmove(addr, data, len(data))
        if L.load(slot, addr, len(data)) != 0: raise RuntimeError('bad model file')
        h = np.frombuffer(data, '<u4', 16); nio, nin = int(h[7]), int(h[8])
        T = np.frombuffer(data, '<i4', count=int(h[1]) * 5, offset=int(h[9])).reshape(-1, 5)
        io = np.frombuffer(data, '<i4', count=nio, offset=int(h[11]))
        sizes = [int(T[t, 1]) for t in io]
        bufs = [np.zeros(s * max_jobs, np.float32) for s in sizes]
        for i, b in enumerate(bufs): L.set_io(slot, i, b.ctypes.data)
        for t in range(self.threads):
            if L.setup_thread(slot, t): raise MemoryError('thread setup failed')
        self.m[slot] = dict(nin=nin, nio=nio, sizes=sizes, bufs=bufs, max=max_jobs)
    def run(self, slot, inputs, n):
        md = self.m[slot]; L = self.lib
        for i in range(md['nin']):
            md['bufs'][i][:n * md['sizes'][i]] = np.ascontiguousarray(inputs[i], np.float32).reshape(-1)[:n * md['sizes'][i]]
        L.reset_jobs()
        def work(t):
            while True:
                j = L.take_job()
                if j >= n: return
                L.run_job(slot, t, j)
        for f in [self.pool.submit(work, t) for t in range(self.threads)]: f.result()
        return [md['bufs'][i][:n * md['sizes'][i]].copy() for i in range(md['nin'], md['nio'])]
