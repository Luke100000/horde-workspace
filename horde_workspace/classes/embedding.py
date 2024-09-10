from typing import Literal

from horde_sdk.ai_horde_api.apimodels import TIPayloadEntry
from pydantic import BaseModel

from horde_workspace.utils import get


class Embedding(BaseModel):
    name: str = ""
    id: int = -1
    strength: float = 1.0
    inject: Literal["prompt", "negprompt"] = "prompt"

    @staticmethod
    def from_id(model_id: int) -> "Embedding":
        e = Embedding(id=model_id)
        e.resolve()
        return e

    def resolve(self) -> bool:
        if not self.name:
            data = get(f"https://civitai.com/api/v1/models/{self.id}")
            self.name = data["name"]
            return True
        return False

    def __str__(self) -> str:
        return f"{self.name} ({self.id})"

    def to_payload(self) -> TIPayloadEntry:
        return TIPayloadEntry(
            name=str(self.id),
            strength=self.strength,
            inject_ti=self.inject,
        )
