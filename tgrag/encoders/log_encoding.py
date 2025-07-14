from tgrag.encoders.encoder import Encoder


class LogEncoder(Encoder):
    def __init__(self, scale: float | None = None):
        self.scale = scale

    def __call__(self, input: float) -> float:
        return input
