import asyncio
import io
import logging

import aiohttp
from PIL import Image
from attr import dataclass
from horde_sdk import RequestErrorResponse
from horde_sdk.ai_horde_api import (
    KNOWN_SAMPLERS,
    AIHordeAPIAsyncClientSession,
    AIHordeAPIAsyncSimpleClient,
    KNOWN_SOURCE_PROCESSING,
)
from horde_sdk.ai_horde_api.apimodels import (
    ImageGenerateAsyncRequest,
    ImageGenerateStatusResponse,
    ImageGenerationInputPayload,
)

from horde_workspace.classes.job import Job
from horde_workspace.data import MODELS, LORAS, EMBEDDINGS, SNIPPETS
from horde_workspace.utils import download_image, b64_encode_image, GenerationError
from horde_workspace.workspace import Workspace

try:
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except AttributeError:
    pass


@dataclass
class Generation:
    images: list[bytes]
    kudos: int

    def get_images(self) -> list[Image.Image]:
        return [Image.open(io.BytesIO(i)) for i in self.images]

    def get_image(self) -> Image.Image:
        return Image.open(io.BytesIO(self.images[0]))


def generate_images(ws: Workspace, job: Job) -> Generation:
    return asyncio.run(async_generate_images(ws, job))


async def async_generate_images(ws: Workspace, job: Job) -> Generation:
    aiohttp_session = aiohttp.ClientSession()
    horde_client_session = AIHordeAPIAsyncClientSession(aiohttp_session)

    model = MODELS[job.model]
    loras = job.loras + [LORAS[lora] for lora in model.base_loras]
    tis = job.tis + [EMBEDDINGS[ti] for ti in model.base_tis]

    async with aiohttp_session, horde_client_session:
        client = AIHordeAPIAsyncSimpleClient(
            aiohttp_session=aiohttp_session,
            horde_client_session=horde_client_session,
        )

        kwargs = {}
        if job.source_image is not None:
            kwargs["source_processing"] = KNOWN_SOURCE_PROCESSING.img2img

        if job.size is None:
            width = job.width
            height = job.height
        else:
            width = job.size.width
            height = job.size.height

        prompt = (
            model.base_positive
            + ", "
            + job.prompt
            + "###"
            + job.negprompt
            + ", "
            + model.base_negative
        )

        # Apply snippets
        for name, value in SNIPPETS.items():
            prompt = prompt.replace("%" + name + "%", value)

        if "%" in prompt:
            logging.warning("Unresolved snippet in prompt: %s", prompt)

        response: ImageGenerateStatusResponse
        response, _ = await client.image_generate_request(
            ImageGenerateAsyncRequest(
                trusted_workers=ws.trusted_workers,
                slow_workers=ws.slow_workers,
                shared=ws.shared,
                apikey=ws.apikey,
                workers=ws.workers,
                prompt=prompt.strip(",").strip(),
                models=[model.name],
                source_image=b64_encode_image(job.source_image)
                if job.source_image
                else None,
                **kwargs,
                params=ImageGenerationInputPayload(
                    steps=job.steps,
                    seed=job.seed,
                    cfg_scale=job.cfg_scale,
                    clip_skip=model.clip_skip,
                    denoising_strength=job.denoising_strength,
                    sampler_name=KNOWN_SAMPLERS.k_dpmpp_2m,
                    height=height,
                    width=width,
                    tis=[ti.to_payload() for ti in tis],
                    loras=[lora.to_payload() for lora in loras],
                    n=job.n,
                    transparent=job.transparent,
                    hires_fix=job.hires,
                    control_type=job.control_type,
                    image_is_control=job.control_type is not None,
                ),
            ),
        )

        if isinstance(response, RequestErrorResponse):
            raise GenerationError(response.message)

        tasks = [
            asyncio.create_task(download_image(aiohttp_session, generation.img))
            for generation in response.generations
        ]

        images = await asyncio.gather(*tasks)

        ws.add_kudos(int(response.kudos))

        # noinspection PyTypeChecker
        return Generation(
            images=[i for i in images if i is not None],
            kudos=int(response.kudos),
        )  # pyright: ignore [reportArgumentType]
    return Generation(images=[], kudos=0)
