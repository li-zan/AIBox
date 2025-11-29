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
    from app.modules.vehicle_yolo import VehicleYoloModule
    from app.modules.smoking_yolo import SmokingYoloModule
    from app.modules.fall_yolo import FallYoloModule
    from app.modules.fighting_yolo import FightingYoloModule
    from app.modules.vehicle_count_yolo import VehicleCountYoloModule
    from app.modules.illegal_parking_yolo import IllegalParkingYoloModule
    from app.modules.mask_yolo import MaskYoloModule
    from app.modules.phone_yolo import PhoneYoloModule
    from app.modules.ebike_illegal_parking_yolo import EbikeIllegalParkingYoloModule
    from app.modules.ebike_in_elevator_yolo import EbikeInElevatorYoloModule
    from app.modules.crowd_yolo import CrowdYoloModule
    from app.modules.people_count_yolo import PeopleCountYoloModule
    from app.modules.loitering_yolo import LoiteringYoloModule
    from app.modules.run_yolo import RunYoloModule

    register_module('fire', FireYoloModule)
    register_module('smoke', SmokeYoloModule)
    register_module('plate', PlateYoloModule)
    register_module('intrusion', IntrusionYoloModule)
    register_module('helmet', HelmetYoloModule)
    register_module('fall', FallYoloModule)
    register_module('fighting', FightingYoloModule)
    register_module('loitering', LoiteringYoloModule)
    register_module('crowd', CrowdYoloModule)
    register_module('vehicle', VehicleYoloModule)
    register_module('vehicle_count', VehicleCountYoloModule)
    register_module('smoking', SmokingYoloModule)
    register_module('illegal_parking', IllegalParkingYoloModule)
    register_module('mask', MaskYoloModule)
    register_module('phone', PhoneYoloModule)
    register_module('ebike_in_elevator', EbikeInElevatorYoloModule)
    register_module('fatigue', FatigueYoloModule)
    register_module('ebike_illegal_parking', EbikeIllegalParkingYoloModule)
    register_module('run', RunYoloModule)
    register_module('people_count', PeopleCountYoloModule)

except Exception as e:
    # If ultralytics not installed, stub will remain
    print(f"Fire/Smoke YOLO unavailable: {e}")
    pass
