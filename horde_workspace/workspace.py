import os
import shelve
import uuid
from os import PathLike
from pathlib import Path
from PIL import Image


class Workspace:
    def __init__(self, directory: PathLike | str):
        super().__init__()

        self.directory = Path(directory)

        self.trusted_workers = False
        self.slow_workers = True
        self.shared = True
        self.apikey = os.getenv("HORDE_API_KEY")
        self.workers = []

        self.directory.mkdir(parents=True, exist_ok=True)
        self.shelve = shelve.open(str(self.directory / ".stats"))
        self.shelve["kudos"] = self.shelve.get("kudos", 0)

    def save(self, image: Image.Image, name: str | None = None) -> str:
        if name is None:
            name = f"{uuid.uuid4()}.webp"

        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)

        return name

    def load(self, name: str) -> Image.Image:
        return Image.open(self.directory / name)

    def add_kudos(self, kudos: int) -> None:
        self.shelve["kudos"] += kudos
        self.shelve.sync()

    def get_kudos(self) -> int:
        return self.shelve["kudos"]
