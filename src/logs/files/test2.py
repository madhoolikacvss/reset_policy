import sys
from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

# --- CONFIGURATION ---
TARGET_IDS = [16, 17, 18, 19]       # The specific IDs you want to check
BAUDRATES = [4000000, 1000000]     # Common DYNAMIXEL baud rates to test
DEVICES = ["/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT89FK0C-if00-port0"]       # Your U2D2 or USB-to-RS485 port path
PROTOCOL_VERSION = 2.0           # Use 1.0 for older AX/MX series, 2.0 for X-series/PRO

def check_specific_ids():
    packet_handler = PacketHandler(PROTOCOL_VERSION)

    for device in DEVICES:
        port_handler = PortHandler(device)
       
        # Try opening the serial port
        if not port_handler.openPort():
            print(f"[-] Failed to open port: {device}")
            continue

        for baud in BAUDRATES:
            if not port_handler.setBaudRate(baud):
                print(f"[-] Failed to set baudrate to {baud} on {device}")
                continue
           
            print(f"[*] Scanning {device} at {baud} bps for IDs: {TARGET_IDS}...")

            for dxl_id in TARGET_IDS:
                # Ping the specific ID directly
                model_num, comm_result, dxl_error = packet_handler.ping(port_handler, dxl_id)
               
                if comm_result == COMM_SUCCESS:
                    print(f" [->] SUCCESS: Found ID {dxl_id} (Model: {model_num})")
                elif dxl_error != 0:
                    print(f" [!] ID {dxl_id} responded with hardware error code: {dxl_error}")
                   
        port_handler.closePort()

if __name__ == "__main__":
    check_specific_ids()