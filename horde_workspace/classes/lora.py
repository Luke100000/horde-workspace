from horde_sdk.ai_horde_api.apimodels import LorasPayloadEntry
from pydantic import BaseModel

from horde_workspace.utils import get


class Lora(BaseModel):
    name: str = ""
    id: int = -1
    version_id: int = -1
    strength: float = 1.0

    @staticmethod
    def from_id(version_id: int) -> "Lora":
        e = Lora(version_id=version_id)
        e.resolve()
        return e

    def resolve(self):
        if self.version_id == -1 and self.id == -1:
            raise ValueError("model_id or version_id must be set")
        elif self.id == -1:
            data = get(f"https://civitai.com/api/v1/model-versions/{self.version_id}")
            self.name = data["model"]["name"]
            self.id = data["modelId"]
            return True
        elif self.version_id == -1:
            data = get(f"https://civitai.com/api/v1/models/{self.id}")
            self.name = data["name"]
            self.version_id = data["modelVersions"][0]["id"]
            return True
        return False

    def __str__(self) -> str:
        return f"{self.name} ({self.id}@{self.version_id})"

    def to_payload(self) -> LorasPayloadEntry:
        return LorasPayloadEntry(
            name=str(self.version_id),
            model=self.strength,
            is_version=True,
        )
