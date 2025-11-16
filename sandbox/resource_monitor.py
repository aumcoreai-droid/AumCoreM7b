
def within_limits(cpu_seconds=10, mem_mb=512):
    return True

def get_resource_usage():
    return {
        "cpu_percent": 0,
        "memory_mb": 0,
        "disk_usage": 0
    }
