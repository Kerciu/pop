from dataclasses import dataclass
from src.models.acceleration_range import AccelerationRange


@dataclass
class AccelerationTable:
    ranges: list['AccelerationRange']

    def get_max_acceleration_for_weight(self, weight: float) -> int:
        for acceleration_range in self.ranges:
            if acceleration_range.min_weight_exclusive < weight <= acceleration_range.max_weight_inclusive:
                return acceleration_range.max_accel
        return 0