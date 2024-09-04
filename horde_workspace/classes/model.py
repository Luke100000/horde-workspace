from pydantic import BaseModel


class Model(BaseModel):
    name: str
    base_model: str

    clip_skip: int = 2
    resolution: int = 1024

    base_positive: str = (
        "score_9, score_8_up, score_7_up, score_6_up, score_5_up, masterpiece"
    )
    base_negative: str = "%watermark%"

    base_loras: list[str] = []
    base_tis: list[str] = []
