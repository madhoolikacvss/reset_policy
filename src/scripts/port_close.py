# scripts/close_port.py

"""
Force close the Dynamixel serial port.
Run this if you get "port in use" errors.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dynamixel_sdk import PortHandler

PORT_NAME = (
    "/dev/serial/by-id/"
    "usb-FTDI_USB__-__Serial_Converter_FT89FK0C"
    "-if00-port0"
)

def force_close_port():
    port = PortHandler(PORT_NAME)
    if port.is_open:
        print(f"Port {PORT_NAME} is open. Closing...")
        port.closePort()
        print("Port closed.")
    else:
        print(f"Port {PORT_NAME} is not open.")

if __name__ == "__main__":
    force_close_port()