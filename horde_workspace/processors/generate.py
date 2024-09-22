import asyncio
import io
import logging

import aiohttp
from PIL import Image
from attr import dataclass

from horde_workspace.classes.job import Job
from horde_workspace.data import MODELS, LORAS, EMBEDDINGS, SNIPPETS
from horde_workspace.utils import b64_encode_image, GenerationError, download_image
from horde_workspace.workspace import Workspace

try:
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # pyright: ignore [reportAttributeAccessIssue]
except AttributeError:
    pass


class APIError(Exception):
    pass


@dataclass
class Generation:
    uuids: list[str] = []
    images: list[bytes] = []
    kudos: int = 0

    def get_images(self) -> list[Image.Image]:
        return [Image.open(io.BytesIO(i)) for i in self.images]

    def get_image(self) -> Image.Image:
        if not self.images:
            raise GenerationError("No images generated")
        return Image.open(io.BytesIO(self.images[0]))


def generate_images(ws: Workspace, job: Job) -> Generation:
    return asyncio.run(async_generate_images(ws, job))


async def async_generate_images(ws: Workspace, job: Job) -> Generation:
    model = MODELS[job.model] if isinstance(job.model, str) else job.model
    loras = job.loras + [
        (LORAS[lora] if isinstance(lora, str) else lora) for lora in model.base_loras
    ]
    tis = job.tis + [
        (EMBEDDINGS[ti] if isinstance(ti, str) else ti) for ti in model.base_tis
    ]

    # Dynamic kwargs to make API happy
    kwargs = {}
    if job.source_image is not None:
        kwargs["source_image"] = b64_encode_image(job.source_image)
        kwargs["source_processing"] = "img2img"

    if ws.workers:
        kwargs["workers"] = ws.workers

    params_kwargs = {}
    if job.control_type is not None:
        params_kwargs["control_type"] = job.control_type
        params_kwargs["image_is_control"] = True

    # Set size
    if job.size is None:
        width = job.width
        height = job.height
    else:
        width = job.size.width
        height = job.size.height

    # Construct prompt
    prompt = f"{model.base_positive}, {job.prompt} ### {job.negprompt}, {model.base_negative}"

    # Apply snippets
    for name, value in SNIPPETS.items():
        prompt = prompt.replace("%" + name + "%", value)

    if "%" in prompt:
        logging.warning("Unresolved snippet in prompt: %s", prompt)

    payload = dict(
        trusted_workers=ws.trusted_workers,
        slow_workers=ws.slow_workers,
        nsfw=ws.nsfw,
        censor_nsfw=ws.censor_nsfw,
        shared=ws.shared,
        prompt=prompt.strip().strip(",").strip().strip("#").strip(),
        models=[model.name],
        **kwargs,
        params=dict(
            steps=model.default_steps if job.steps is None else job.steps,
            seed=job.seed,
            cfg_scale=model.default_cfg_scale
            if job.cfg_scale is None
            else job.cfg_scale,
            clip_skip=model.clip_skip,
            denoising_strength=job.denoising_strength,
            sampler_name=model.sampler,
            height=height,
            width=width,
            tis=[ti.to_payload() for ti in tis],
            loras=[lora.to_payload() for lora in loras],
            n=job.n,
            transparent=job.transparent,
            hires_fix=job.hires,
            **params_kwargs,
        ),
    )

    generation = await async_generate_images_inner(payload, ws.apikey)

    ws.add_kudos(int(generation.kudos))

    return generation


async def request(func, url: str, headers: dict, payload: dict | None = None) -> dict:
    while True:
        try:
            async with func(url, json=payload, headers=headers) as response:
                if response.status == 429:
                    logging.info("Rate limited, waiting 1s")
                    await asyncio.sleep(1)
                    continue
                if response.status not in [200, 202]:
                    raise APIError(
                        f"Error during request {url}: {response.status}, {await response.text()}"
                    )

                response_data = await response.json()
                logging.debug(
                    "Response from %s %s: %s", func.__name__, url, response_data
                )

                return response_data
        except (aiohttp.ClientError, Exception) as e:
            raise APIError(e)


async def async_generate_images_inner(
    payload: dict, apikey: str, timeout: int = 1000
) -> Generation:
    headers = {
        "apikey": apikey,
        "Client-Agent": "horde-workspace:0:https://github.com/Luke100000/horde-workspace",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        # Get the UUID from the generation response
        response_data = await request(
            session.post,
            "https://stablehorde.net/api/v2/generate/async",
            headers,
            payload,
        )
        request_id = response_data.get("id")
        if not request_id:
            raise APIError("No request ID found in the response")

        if "warnings" in response_data:
            for warning in response_data["warnings"]:
                logging.warning(warning)

        # Poll
        done = False
        not_possible = False
        url_check = f"https://stablehorde.net/api/v2/generate/check/{request_id}"
        for _ in range(timeout):
            check_data = await request(session.get, url_check, headers)

            # Check if the request is completed
            if check_data.get("done"):
                done = True
                break
            elif not check_data.get("is_possible"):
                logging.debug("Not possible: %s", payload)
                not_possible = True
                break
            elif check_data.get("faulted"):
                raise APIError("Request faulted")
            else:
                logging.debug(
                    f"{response_data.get('wait_time', 0)}s remaining, {response_data} processing."
                )

            await asyncio.sleep(1)

        # Fetch
        url_status = f"https://stablehorde.net/api/v2/generate/status/{request_id}"
        if done:
            check_data = await request(session.get, url_status, headers)
            valid_gens = [
                gen for gen in check_data.get("generations", []) if not gen["censored"]
            ]

            if not valid_gens:
                raise APIError("No images generated")

            tasks = [
                asyncio.create_task(download_image(session, gen["img"]))
                for gen in valid_gens
            ]
            uuids = [gen["id"] for gen in valid_gens]

            # noinspection PyTypeChecker
            images: list[bytes] = await asyncio.gather(*tasks)

            return Generation(
                uuids=uuids,
                images=images,
                kudos=int(response_data["kudos"]),
            )

        # Cancel on timeout
        await request(session.delete, url_status, headers)
        raise APIError("Not Possible") if not_possible else APIError("Timeout")
