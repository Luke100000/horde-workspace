from pydantic import BaseModel

from horde_workspace.classes.embedding import Embedding
from horde_workspace.classes.lora import Lora


class Model(BaseModel):
    name: str
    base_model: str

    clip_skip: int = 2
    resolution: int = 1024

    default_steps: int = 30
    default_cfg_scale: float = 7.5

    sampler: str = "k_dpmpp_2m"

    base_positive: str = ""
    base_negative: str = ""

    base_loras: list[str | Lora] = []
    base_tis: list[str | Embedding] = []
