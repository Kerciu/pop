from dataclasses import dataclass


@dataclass
class AccelerationRange:
    min_weight_exclusive:   float
    max_weight_inclusive:   float
    max_accel:              int
