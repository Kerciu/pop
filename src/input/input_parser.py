from models.acceleration_range import AccelerationRange
from models.gift import Gift
from typing import List


class InputParser:
    def __init__(self, data_path):
        self.data_path = data_path

    def parse_information(self):
        lines = self.read_file()
        assert len(lines) > 0, "Input file is empty"

        T, D, W, G = self.parse_configuration(lines[0])

        accel_lines = lines[1: W + 1]
        acceleration_ranges = self.parse_acceleration_ranges(accel_lines)

        gift_lines = lines[W + 1: W + 1 + G]
        gifts = self.parse_gifts(gift_lines)

        gifts_dict = {g.name: g for g in gifts}

        return T, D, W, G, acceleration_ranges, gifts_dict

    def parse_gifts(self, gift_lines: List[str]) -> List[Gift]:
        gifts = []
        for line in gift_lines:
            parts = line.strip().split()
            gifts.append(Gift(
                name=parts[0],
                score=int(parts[1]),
                weight=int(parts[2]),
                x=int(parts[3]),
                y=int(parts[4])
            ))
        return gifts

    def parse_acceleration_ranges(
            self,
            accel_lines: List[str]) -> List[AccelerationRange]:
        ranges = []
        last_max_weight = 0.0

        for line in accel_lines:
            l_i, a_i = map(int, line.strip().split())

            ranges.append(AccelerationRange(
                min_weight_exclusive=last_max_weight,
                max_weight_inclusive=l_i,
                max_accel=a_i
            ))
            last_max_weight = l_i

        return ranges

    def parse_configuration(self, data):
        config = data[0].strip().split()

        T, D, W, G = map(int, config)

        return T, D, W, G

    def read_file(self):
        try:
            with open(self.data_path, 'r') as file:
                data = file.readlines()
        except FileNotFoundError:
            print(f"File not found: {self.data_path}")
            return []
        except Exception as e:
            print(f"Error reading file: {e}")
            return []

        return data
