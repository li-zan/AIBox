from typing import Dict, Type

from app.modules.base import BaseModule

_module_registry: Dict[str, Type[BaseModule]] = {}


def register_module(name: str, cls: Type[BaseModule]) -> None:
    _module_registry[name] = cls


def get_registered_modules() -> Dict[str, Type[BaseModule]]:
    return dict(_module_registry)

try:
    from app.modules.fire_yolo import FireYoloModule
    from app.modules.smoke_yolo import SmokeYoloModule
    from app.modules.intrusion_yolo import IntrusionYoloModule
    from app.modules.plate_yolo import PlateYoloModule

    register_module('fire', FireYoloModule)
    register_module('smoke', SmokeYoloModule)
    register_module('plate', PlateYoloModule)
    register_module('intrusion', IntrusionYoloModule)
    register_module('helmet', None)
    register_module('fall', None)
    register_module('fighting', None)
    register_module('loitering', None)
    register_module('crowd', None)
    register_module('vehicle', None)
    register_module('person', None)
    register_module('pet', None)
    register_module('face', None)
    register_module('mask', None)
    register_module('phone', None)
    register_module('defect', None)
    register_module('spill', None)
    register_module('door_open', None)
    register_module('run', None)
    register_module('crossline', None)

except Exception as e:
    # If ultralytics not installed, stub will remain
    print(f"Fire/Smoke YOLO unavailable: {e}")
    pass
