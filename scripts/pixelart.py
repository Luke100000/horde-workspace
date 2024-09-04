from horde_workspace.classes.job import Job
from horde_workspace.data import LORAS
from horde_workspace.processors import pixelize, generate_images
from horde_workspace.workspace import Workspace


if __name__ == "__main__":
    ws = Workspace("output/pixelart")

    job = Job(
        prompt="A seamless pixelart texture of an old mossy brick wall.",
        width=512,
        height=512,
        model="ICBINP",
        loras=[LORAS["Faithful 32px seamless blocks v1"]],
    )

    image = generate_images(ws, job).get_image()
    image = pixelize(image)

    ws.save(image)
