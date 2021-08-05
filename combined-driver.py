import ctypes
import time
import keyboard as key

SendInput = ctypes.windll.user32.SendInput

# C struct redefinitions
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


# Main
key.wait('p')
up = False
down = False
left = False
right = False
with open("final-gas.txt", 'r+') as file, open("final-direction.txt", 'r+') as file2:
    commands = file.read()
    commands2 = file2.read()
    for a in range(len(commands)):
        if commands[a] == '2':
            if down:
                ReleaseKey(0x1F)
                down = False
            PressKey(0x11)
            up = True
        elif commands[a] == '1':
            if up:
                ReleaseKey(0x11)
                up = False
            PressKey(0x1F)
            down = True
        else:
            if up:
                ReleaseKey(0x11)
                up = False
            if down:
                ReleaseKey(0x1F)
                down = False
        if commands2[a] == '2':
            if right:
                ReleaseKey(0x20)
                right = False
            PressKey(0x1E)
            left = True
        elif commands2[a] == '1':
            if left:
                ReleaseKey(0x1E)
                left = False
            PressKey(0x20)
            right = True
        else:
            if right:
                ReleaseKey(0x20)
                right = False
            if left:
                ReleaseKey(0x1E)
                left = False
        time.sleep(0.08)