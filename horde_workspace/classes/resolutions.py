from enum import Enum


class Sizes(Enum):
    # CivitAI
    SQUARE = (1024, 1024)
    PORTRAIT = (832, 1216)
    LANDSCAPE = (1216, 832)

    @property
    def width(self) -> int:
        return self.value[0]

    @property
    def height(self) -> int:
        return self.value[1]
