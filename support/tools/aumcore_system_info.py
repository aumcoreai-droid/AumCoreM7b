import platform
import subprocess
import os

# ---------------------------
# Step 3: System Info + BIOS (Production Ready)
# ---------------------------
def get_system_info():
    info = {
        'system': platform.system(),
        'node': platform.node(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor()
    }

    # Check if running on real system with root access
    try:
        if os.geteuid() == 0:  # Root permission check
            bios_info = subprocess.check_output('dmidecode -t bios', shell=True, stderr=subprocess.DEVNULL).decode()
            info['bios'] = bios_info
        else:
            info['bios'] = "BIOS info not accessible: Requires root permissions on physical machine."
    except Exception as e:
        info['bios'] = f"BIOS info not available: {e}"

    return info

# ---------------------------
# Example Usage / Testing
# ---------------------------
if __name__ == "__main__":
    system_info = get_system_info()
    print("🌐 System Configuration (Production Ready):")
    for key, value in system_info.items():
        print(f"{key}: {value}")
