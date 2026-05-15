class Random:
    GOLDEN = 0x9E3779B97F4A7C15
    MASK64 = 0xFFFFFFFFFFFFFFFF

    def __init__(self, seed: int = 0):
        self.state = seed & self.MASK64

    def seed(self, s: int):
        self.state = s & self.MASK64

    def next_u64(self) -> int:
        self.state = (self.state + self.GOLDEN) & self.MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self.MASK64
        z = z ^ (z >> 31)
        return z & self.MASK64

    def randint(self, low: int, high: int) -> int:
        if high <= low:
            raise ValueError("high must be > low")
        width = high - low
        r = self.next_u64()
        return low + (r % width)  # high is exclusive

    def get_state(self) -> int:
        return self.state

    def set_state(self, value: int) -> None:
        self.state = value & self.MASK64
