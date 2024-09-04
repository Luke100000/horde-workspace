from horde_workspace.classes.job import Job
from horde_workspace.classes.resolutions import Sizes
from horde_workspace.data import LORAS, EMBEDDINGS
from horde_workspace.processors import upscale, generate_images, nsfw, caption
from horde_workspace.workspace import Workspace


def main():
    ws = Workspace("output")

    job = Job(
        prompt="A beautiful landscape of a mountain range.",
        size=Sizes.LANDSCAPE,
        model="AlbedoBase XL (SDXL)",
        loras=[LORAS["Detail Tweaker XL"]],
        tis=[EMBEDDINGS["FastNegative"]],
    )

    image = generate_images(ws, job).get_image()

    print("NSFW:", nsfw(ws, image))
    print("Caption:", caption(ws, image))

    # TODO: Interrogation is not working
    # print("Interrogation:", interrogation(ws, image))

    image = upscale(ws, image)

    name = ws.save(image)
    print(f"Saved image as {name}")


if __name__ == "__main__":
    main()
