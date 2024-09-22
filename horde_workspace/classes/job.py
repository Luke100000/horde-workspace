from PIL import Image
from pydantic import BaseModel, ConfigDict

from horde_workspace.classes.embedding import Embedding
from horde_workspace.classes.lora import Lora
from horde_workspace.classes.model import Model
from horde_workspace.classes.resolutions import Sizes


class Job(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt: str
    negprompt: str = ""
    steps: int | None = None
    width: int = 1024
    height: int = 1024
    size: Sizes | None = None
    n: int = 1
    model: str | Model = "AlbedoBase XL (SDXL)"
    seed: str | None = None
    tis: list[Embedding] = []
    loras: list[Lora] = []
    cfg_scale: float | None = None
    denoising_strength: float = 1.0
    transparent: bool = False
    hires: bool = False
    control_type: str | None = None
    source_image: Image.Image | None = None
