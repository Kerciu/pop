from src.models.coordinate import Coordinate


def distance(source: 'Coordinate', target: 'Coordinate') -> float:
    dx = source.x - target.x
    dy = source.y - target.y
    return (dx * dx + dy * dy) ** 0.5