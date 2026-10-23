"""Bounded Win32 DIB presenter; WM_TIMER also paints during modal window moves.

All HWND operations live on the window thread. The producer holds at most one
pending canvas and waits only until its deadline; no HighGUI global UI lock.
"""
import collections
import ctypes as C
from ctypes import wintypes as W
import threading
import time
import numpy as np


class Win32VideoWindow:
    def __init__(self, title):
        self._title = title
        self._lock = threading.Lock()
        self._pending = None
        self._late_commits = collections.deque(maxlen=32)
        self._last = None
        self._shape = None
        self._key = 255
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._error = None
        self.modal_active = False
        self.modal_entries = 0
        self.modal_paints = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(5):
            raise RuntimeError("Win32 video window startup timeout")
        if self._error:
            raise RuntimeError(self._error)

    def show(self, image, deadline, tag=None):
        if self._error:
            raise RuntimeError(self._error)
        image = np.ascontiguousarray(image)
        job = dict(image=image, deadline=deadline, tag=tag, done=threading.Event(), shown=None)
        with self._lock:
            self._pending = job
        job['done'].wait(max(0, deadline-time.perf_counter()))
        return job['shown']

    def take_late_commits(self):
        with self._lock:
            result = list(self._late_commits)
            self._late_commits.clear()
        return result

    def poll_key(self):
        if self._closed.is_set():
            return 27
        with self._lock:
            key, self._key = self._key, 255
        return key

    def close(self):
        if hasattr(self, '_hwnd'):
            C.windll.user32.PostMessageW(self._hwnd, 0x0010, 0, 0)
        self._thread.join(2)

    def _run(self):
        user, gdi, kernel = C.windll.user32, C.windll.gdi32, C.windll.kernel32
        LRESULT = C.c_ssize_t
        CALLBACK = C.WINFUNCTYPE(LRESULT, W.HWND, W.UINT, W.WPARAM, W.LPARAM)

        class WndClass(C.Structure):
            _fields_ = [('style', W.UINT), ('proc', CALLBACK), ('cls_extra', C.c_int),
                        ('wnd_extra', C.c_int), ('instance', W.HINSTANCE),
                        ('icon', W.HICON), ('cursor', W.HANDLE), ('background', W.HBRUSH),
                        ('menu', W.LPCWSTR), ('name', W.LPCWSTR)]

        class BitmapHeader(C.Structure):
            _fields_ = [('size', W.DWORD), ('width', W.LONG), ('height', W.LONG),
                        ('planes', W.WORD), ('bits', W.WORD), ('compression', W.DWORD),
                        ('image_size', W.DWORD), ('x', W.LONG), ('y', W.LONG),
                        ('used', W.DWORD), ('important', W.DWORD)]

        user.DefWindowProcW.argtypes = [W.HWND, W.UINT, W.WPARAM, W.LPARAM]
        user.DefWindowProcW.restype = LRESULT
        user.CreateWindowExW.argtypes = [W.DWORD, W.LPCWSTR, W.LPCWSTR, W.DWORD,
                                        C.c_int, C.c_int, C.c_int, C.c_int,
                                        W.HWND, W.HMENU, W.HINSTANCE, W.LPVOID]
        user.CreateWindowExW.restype = W.HWND
        kernel.GetModuleHandleW.argtypes = [W.LPCWSTR]
        kernel.GetModuleHandleW.restype = W.HINSTANCE
        user.GetDC.argtypes = [W.HWND]; user.GetDC.restype = W.HDC
        user.ReleaseDC.argtypes = [W.HWND, W.HDC]
        user.GetClientRect.argtypes = [W.HWND, C.POINTER(W.RECT)]
        user.SetTimer.argtypes = [W.HWND, C.c_size_t, W.UINT, W.LPVOID]
        user.SetTimer.restype = C.c_size_t
        user.PostMessageW.argtypes = [W.HWND, W.UINT, W.WPARAM, W.LPARAM]
        user.ShowWindow.argtypes = [W.HWND, C.c_int]
        user.DestroyWindow.argtypes = [W.HWND]
        user.SetWindowPos.argtypes = [W.HWND, W.HWND, C.c_int, C.c_int, C.c_int, C.c_int, W.UINT]
        user.AdjustWindowRectEx.argtypes = [C.POINTER(W.RECT), W.DWORD, W.BOOL, W.DWORD]
        user.ValidateRect.argtypes = [W.HWND, W.LPVOID]
        gdi.StretchDIBits.argtypes = [W.HDC, C.c_int, C.c_int, C.c_int, C.c_int,
                                     C.c_int, C.c_int, C.c_int, C.c_int,
                                     W.LPVOID, W.LPVOID, W.UINT, W.DWORD]
        gdi.PatBlt.argtypes = [W.HDC, C.c_int, C.c_int, C.c_int, C.c_int, W.DWORD]

        def paint(hwnd, image):
            h, w = image.shape[:2]
            stride = ((w*3+3)//4)*4
            if stride != w*3:
                packed = np.zeros((h, stride), dtype=np.uint8)
                packed[:, :w*3] = image.reshape(h, w*3)
            else:
                packed = image
            header = BitmapHeader(C.sizeof(BitmapHeader), w, -h, 1, 24, 0,
                                  stride*h, 0, 0, 0, 0)
            rect = W.RECT(); user.GetClientRect(hwnd, C.byref(rect))
            cw, ch = max(1, rect.right), max(1, rect.bottom)
            ratio = min(cw/w, ch/h); dw, dh = max(1, int(w*ratio)), max(1, int(h*ratio))
            dc = user.GetDC(hwnd)
            try:
                gdi.PatBlt(dc, 0, 0, cw, ch, 0x00000042)
                if gdi.StretchDIBits(dc, (cw-dw)//2, (ch-dh)//2, dw, dh,
                                    0, 0, w, h, packed.ctypes.data,
                                    C.byref(header), 0, 0x00CC0020) == -1:
                    raise RuntimeError('StretchDIBits failed')
            finally:
                user.ReleaseDC(hwnd, dc)

        def tick(hwnd):
            with self._lock:
                job, self._pending = self._pending, None
            if job is None:
                return
            try:
                if time.perf_counter() <= job['deadline']:
                    shape = job['image'].shape[:2]
                    if shape != self._shape:
                        h, w = shape
                        rect = W.RECT(0, 0, w, h)
                        user.AdjustWindowRectEx(C.byref(rect), 0x00CF0000, False, 0)
                        user.SetWindowPos(hwnd, None, 0, 0, rect.right-rect.left,
                                          rect.bottom-rect.top, 0x0006)
                        self._shape = shape
                    paint(hwnd, job['image'])
                    job['shown'] = time.perf_counter()
                    if job['shown'] > job['deadline']:
                        with self._lock:
                            self._late_commits.append(dict(index=job['tag'],
                                deadline=job['deadline'], shown=job['shown'],
                                overdue_ms=(job['shown']-job['deadline'])*1000))
                    self._last = job['image']
                    if self.modal_active:
                        self.modal_paints += 1
            except Exception as error:
                self._error = str(error)
                user.DestroyWindow(hwnd)
            finally:
                job['done'].set()

        @CALLBACK
        def procedure(hwnd, message, wp, lp):
            if message == 0x0231:
                self.modal_active = True; self.modal_entries += 1
            elif message == 0x0232:
                self.modal_active = False
            if message == 0x0113:  # WM_TIMER is dispatched in move/resize modal loops.
                tick(hwnd)
                return 0
            if message == 0x000F:  # WM_PAINT
                if self._last is not None:
                    paint(hwnd, self._last)
                user.ValidateRect(hwnd, None)
                return 0
            if message == 0x0100:
                with self._lock:
                    self._key = int(wp)
                return 0
            if message == 0x0010:
                user.DestroyWindow(hwnd)
                return 0
            if message == 0x0002:
                self._closed.set(); user.PostQuitMessage(0)
                return 0
            return user.DefWindowProcW(hwnd, message, wp, lp)

        try:
            instance = kernel.GetModuleHandleW(None)
            name = 'VideoCompressDib_'+str(id(self))
            wc = WndClass(0, procedure, 0, 0, instance, None, None, None, None, name)
            user.RegisterClassW.argtypes = [C.POINTER(WndClass)]
            if not user.RegisterClassW(C.byref(wc)):
                raise C.WinError()
            self._hwnd = user.CreateWindowExW(0, name, self._title, 0x00CF0000,
                                              40, 40, 580, 850, None, None, instance, None)
            if not self._hwnd:
                raise C.WinError()
            user.ShowWindow(self._hwnd, 5)
            if not user.SetTimer(self._hwnd, 1, 5, None):
                raise C.WinError()
            self._ready.set()
            msg = W.MSG()
            while user.GetMessageW(C.byref(msg), None, 0, 0) > 0:
                user.TranslateMessage(C.byref(msg)); user.DispatchMessageW(C.byref(msg))
        except Exception as error:
            self._error = str(error); self._ready.set(); self._closed.set()
