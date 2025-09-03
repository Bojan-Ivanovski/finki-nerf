class Point:
    def __init__(self, x, y, z, step):
        self.x: float = x
        self.y: float = y
        self.z: float = z
        self.step = step

    def __repr__(self):
        return f"Point({self.x}, {self.y}, {self.z}, (Step : {self.step}))"
