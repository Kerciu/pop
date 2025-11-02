import pytest
from pathlib import Path
from src.input.input_parser import InputParser
from src.models.gift import Gift
from src.models.acceleration_range import AccelerationRange


MOCK_INPUT_DATA = """15 3 4 4
15 8
30 6
45 4
60 2
Kacper 1 10 5 1
Alek 2 10 -10 1
John 5 10 8 4
Bob 10 15 0 -100
"""


@pytest.fixture
def mock_input_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "test_data.in"
    with open(file_path, 'w') as f:
        f.write(MOCK_INPUT_DATA)
    return file_path


@pytest.fixture
def parser(mock_input_file: Path) -> InputParser:
    return InputParser(data_path=str(mock_input_file))


def test_parse_configuration(parser: InputParser):
    config_line = "15 3 4 4"
    T, D, W, G = parser.parse_configuration(config_line)

    assert T == 15
    assert D == 3
    assert W == 4
    assert G == 4


def test_parse_acceleration_ranges(parser: InputParser):
    lines = ["15 8", "30 6", "45 4", "60 2"]
    ranges = parser.parse_acceleration_ranges(lines)

    assert len(ranges) == 4
    assert ranges[0] == AccelerationRange(min_weight_exclusive=0.0, max_weight_inclusive=15, max_accel=8)
    assert ranges[1] == AccelerationRange(min_weight_exclusive=15, max_weight_inclusive=30, max_accel=6)
    assert ranges[2] == AccelerationRange(min_weight_exclusive=30, max_weight_inclusive=45, max_accel=4)
    assert ranges[3] == AccelerationRange(min_weight_exclusive=45, max_weight_inclusive=60, max_accel=2)


def test_parse_gifts(parser: InputParser):
    lines = [
        "Olivia 1 10 5 1",
        "Emma 2 10 -10 1",
        "Liam 5 10 8 4",
        "Bob 10 15 0 -100"
    ]
    gifts = parser.parse_gifts(lines)

    assert len(gifts) == 4
    assert gifts[0] == Gift(name="Olivia", score=1, weight=10, c=5, r=1)
    assert gifts[2] == Gift(name="Liam", score=5, weight=10, c=8, r=4)
    assert gifts[3].r == -100


def test_read_file_not_found():
    parser_bad = InputParser("this file does not exist")
    data = parser_bad.read_file()
    assert data == []


def test_parse_information_full(parser: InputParser):
    T, D, W, G, accel_ranges, gifts = parser.parse_information()

    assert T == 15
    assert D == 3
    assert W == 4
    assert G == 4

    assert len(accel_ranges) == 4
    assert accel_ranges[0].max_accel == 8
    assert accel_ranges[3].min_weight_exclusive == 45

    assert len(gifts) == 4
    assert "Kacper" in map(lambda g: g.name, gifts)
    assert "Bob" in map(lambda g: g.name, gifts)

    expected_john = Gift(name="John", score=5, weight=10, c=8, r=4)
    expected_alek = Gift(name="Alek", score=2, weight=10, c=-10, r=1)

    assert gifts[2] == expected_john
    assert gifts[1] == expected_alek
