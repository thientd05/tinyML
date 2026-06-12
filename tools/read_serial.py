"""Reset the ESP32 and capture its serial output for a few seconds (no TTY needed).

PlatformIO's miniterm needs an interactive terminal; this is a headless replacement
for grabbing the one-shot benchmark print.

Usage:  ./env/bin/python tools/read_serial.py [port] [max_seconds] [stop_marker]

Resets the board, then prints serial lines until `stop_marker` is seen (default
"loop idle") or `max_seconds` elapses — so it returns as soon as one full run ends.
"""
import sys
import time

import serial

port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
marker = sys.argv[3] if len(sys.argv) > 3 else "loop idle"

ser = serial.Serial(port, 115200, timeout=0.2)
# ESP32 auto-reset to run mode: EN(=RTS) low pulse while IO0(=DTR) stays high.
ser.dtr = False
ser.rts = True
time.sleep(0.12)
ser.rts = False
ser.reset_input_buffer()

end = time.time() + secs
while time.time() < end:
    line = ser.readline()
    if line:
        text = line.decode("utf-8", "replace")
        sys.stdout.write(text)
        sys.stdout.flush()
        if marker in text:
            break
ser.close()
