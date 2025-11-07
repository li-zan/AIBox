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
    from app.modules.helmet_yolo import HelmetYoloModule
    from app.modules.fatigue_yolo import FatigueYoloModule

    register_module('fire', FireYoloModule)
    register_module('smoke', SmokeYoloModule)
    register_module('plate', PlateYoloModule)
    register_module('intrusion', IntrusionYoloModule)
    register_module('helmet', HelmetYoloModule)
    register_module('fall', None)
    register_module('fighting', None)
    register_module('loitering', None)
    register_module('crowd', None)
    register_module('vehicle', None)
    register_module('vehicle_count', None)
    register_module('smoking', None)
    register_module('illegal_parking', None)
    register_module('mask', None)
    register_module('phone', None)
    register_module('ebike_in_elevator', None)
    register_module('fatigue', FatigueYoloModule)
    register_module('ebike_illegal_parking', None)
    register_module('run', None)
    register_module('people_count', None)

except Exception as e:
    # If ultralytics not installed, stub will remain
    print(f"Fire/Smoke YOLO unavailable: {e}")
    pass
