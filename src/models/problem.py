from typing import List
from dataclasses import dataclass
from input.input_parser import InputParser
from models.acceleration_range import AccelerationRange
from models.gift import Gift

@dataclass
class Problem:
    T: int
    D: int
    W: int
    G: int
    acceleration_ranges: List[AccelerationRange]
    gifts: List[Gift]

    def __init__(self, data_path: str):
        ip = InputParser(data_path=data_path)
        T, D, W, G, acceleration_ranges, gifts = ip.parse_information()
        self.T = T
        self.D = D
        self.W = W
        self.G = G
        self.acceleration_ranges = acceleration_ranges
        self.gifts = gifts

    @property
    def data(self):
        return (self.T, self.D, self.W, self.G,
                self.acceleration_ranges, self.gifts)
