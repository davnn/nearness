import json
import platform
import sys

import nearness


def get_system_info():
    info = {
        "library": nearness.get_version(),
        "python": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
    }
    return info


if __name__ == "__main__":
    print(json.dumps(get_system_info(), indent=4))
