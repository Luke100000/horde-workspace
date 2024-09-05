from pathlib import Path

from pydantic_yaml import parse_yaml_raw_as, to_yaml_str

from horde_workspace.classes.embedding import Embedding
from horde_workspace.classes.lora import Lora
from horde_workspace.classes.model import Model

MODELS: dict[str, Model] = {}
LORAS: dict[str, Lora] = {}
EMBEDDINGS: dict[str, Embedding] = {}

SNIPPETS = {
    "watermark": "watermark, signature, logo, branding, copyright",
}

for yaml in Path("horde_workspace/data/models").glob("*.yaml"):
    with open(yaml, "r") as f:
        model = parse_yaml_raw_as(Model, f.read())

    MODELS[yaml.stem] = model

for yaml in Path("horde_workspace/data/loras").glob("*.yaml"):
    with open(yaml, "r") as f:
        lora = parse_yaml_raw_as(Lora, f.read())

    if lora.resolve():
        with open(yaml, "w") as f:
            f.write(to_yaml_str(lora))

    LORAS[yaml.stem] = lora

for yaml in Path("horde_workspace/data/embeddings").glob("*.yaml"):
    with open(yaml, "r") as f:
        embedding = parse_yaml_raw_as(Embedding, f.read())

    if embedding.resolve():
        with open(yaml, "w") as f:
            f.write(to_yaml_str(embedding))

    EMBEDDINGS[yaml.stem] = embedding
