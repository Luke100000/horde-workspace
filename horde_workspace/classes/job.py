from PIL import Image
from horde_sdk.ai_horde_api.consts import KNOWN_CONTROLNETS
from pydantic import BaseModel, ConfigDict

from horde_workspace.classes.embedding import Embedding
from horde_workspace.classes.lora import Lora
from horde_workspace.classes.model import Model
from horde_workspace.classes.resolutions import Sizes


class Job(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt: str
    negprompt: str = ""
    steps: int = 25
    width: int = 1024
    height: int = 1024
    size: Sizes | None = None
    n: int = 1
    model: str | Model = "AlbedoBase XL (SDXL)"
    seed: str | None = None
    tis: list[Embedding] = []
    loras: list[Lora] = []
    cfg_scale: float = 8.5
    denoising_strength: float = 1.0
    transparent: bool = False
    hires: bool = False
    control_type: KNOWN_CONTROLNETS | None = None
    source_image: Image.Image | None = None
