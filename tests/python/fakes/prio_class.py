import ctypes
from ctypes import wintypes
k = ctypes.WinDLL("kernel32", use_last_error=True)
k.GetCurrentProcess.restype = wintypes.HANDLE
k.GetPriorityClass.argtypes = [wintypes.HANDLE]; k.GetPriorityClass.restype = wintypes.DWORD
print(hex(k.GetPriorityClass(k.GetCurrentProcess())))
