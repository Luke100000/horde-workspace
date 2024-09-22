import os
import uuid
from os import PathLike
from pathlib import Path

from PIL import Image
from dotenv import load_dotenv

load_dotenv()


class Workspace:
    def __init__(self, directory: PathLike | str = "output") -> None:
        super().__init__()

        self.directory = Path(directory)

        self.trusted_workers = False
        self.slow_workers = True
        self.nsfw = True
        self.censor_nsfw = False
        self.shared = True
        self.apikey = os.getenv("HORDE_API_KEY") or "0000000000"
        self.workers = []
        self.kudos = 0

    def save(self, image: Image.Image, name: str | None = None) -> str:
        if name is None:
            name = f"{uuid.uuid4()}.webp"

        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)

        return name

    def exists(self, name: str) -> bool:
        return (self.directory / name).exists()

    def load(self, name: str) -> Image.Image:
        return Image.open(self.directory / name)

    def add_kudos(self, kudos: int) -> None:
        self.kudos += kudos

    def get_kudos(self) -> int:
        return self.kudos
