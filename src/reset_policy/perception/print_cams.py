import glob
import subprocess

def get_all_camera_asics():
    """Finds all video nodes and pairs them with their hardware serial number."""
    devices = {}
    for device in sorted(glob.glob("/dev/video*")):
        try:
            # Query udev metadata properties for the video node
            res = subprocess.run(
                ["udevadm", "info", "--query=property", f"--name={device}"],
                capture_output=True, text=True, timeout=1.0, check=True
            )
            
            # Extract the unique short serial property (this maps to the Asic S/N)
            serial = None
            for line in res.stdout.splitlines():
                if line.startswith("ID_SERIAL_SHORT="):
                    serial = line.split("=", 1)[1].strip()
                    break
            
            if serial:
                if serial not in devices:
                    devices[serial] = []
                devices[serial].append(device)
                
        except (subprocess.SubprocessError, FileNotFoundError):
            continue
            
    return devices

# Run and print results
all_cameras = get_all_camera_asics()
print("Detected RealSense / Camera Asic Mappings:")
for asic, video_paths in all_cameras.items():
    print(f"Asic S/N: {asic} -> Paths: {video_paths}")
