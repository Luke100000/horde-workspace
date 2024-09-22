import asyncio
import io

import aiohttp
from PIL import Image
from attr import dataclass
from pydantic import BaseModel

from horde_workspace.processors.generate import request, APIError
from horde_workspace.utils import (
    download_image,
    b64_encode_image,
    GenerationError,
    assert_none,
)
from horde_workspace.workspace import Workspace

try:
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # pyright: ignore [reportAttributeAccessIssue]
except AttributeError:
    pass


class InterrogationResultItem(BaseModel):
    text: str
    confidence: float


class InterrogationDetails(BaseModel):
    tags: list[InterrogationResultItem]
    sites: list[InterrogationResultItem]
    artists: list[InterrogationResultItem]
    flavors: list[InterrogationResultItem]
    mediums: list[InterrogationResultItem]
    movements: list[InterrogationResultItem]
    techniques: list[InterrogationResultItem]


@dataclass
class AlchemyGeneration:
    image: bytes | None = None
    caption: str | None = None
    nsfw: bool | None = None
    interrogation: InterrogationDetails | None = None

    def get_image(self) -> Image.Image:
        if self.image is None:
            raise GenerationError("No image available")
        return Image.open(io.BytesIO(self.image))


def caption(ws: Workspace, image: Image.Image) -> str:
    return assert_none(alchemist(ws, image, ["caption"]).caption)


def interrogation(ws: Workspace, image: Image.Image) -> InterrogationDetails:
    return assert_none(alchemist(ws, image, ["interrogation"]).interrogation)


def nsfw(ws: Workspace, image: Image.Image) -> bool:
    return assert_none(alchemist(ws, image, ["nsfw"]).nsfw)


def upscale(ws: Workspace, image: Image.Image) -> Image.Image:
    return alchemist(ws, image, ["NMKD_Siax"]).get_image()


def alchemist(ws: Workspace, image: Image.Image, forms: list[str]) -> AlchemyGeneration:
    return asyncio.run(async_alchemist(ws, image, forms))


async def async_alchemist(
    ws: Workspace, image: Image.Image, forms: list[str], timeout: int = 1000
) -> AlchemyGeneration:
    payload = dict(
        apikey=ws.apikey,
        slow_workers=ws.slow_workers,
        source_image=b64_encode_image(image),
        forms=[{"name": form} for form in forms],
    )

    headers = {
        "apikey": ws.apikey,
        "Client-Agent": "horde-workspace:0:https://github.com/Luke100000/horde-workspace",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        # Get the UUID from the generation response
        response_data = await request(
            session.post,
            "https://stablehorde.net/api/v2/interrogate/async",
            headers,
            payload,
        )
        request_id = response_data.get("id")
        if not request_id:
            raise APIError("No request ID found in the response")

        # Poll
        url_status = f"https://stablehorde.net/api/v2/interrogate/status/{request_id}"
        for _ in range(timeout):
            status_data = await request(session.get, url_status, headers)
            print(status_data)

            # Check if the request is completed
            if status_data["state"] == "done":
                forms_by_type = {}
                for form in status_data["forms"]:
                    if form["state"] == "done":
                        if form["form"] in ("caption", "nsfw", "interrogation"):
                            forms_by_type[form["form"]] = form["result"]
                        else:
                            forms_by_type["upscale"] = form["result"][form["form"]]

                return AlchemyGeneration(
                    image=await download_image(session, forms_by_type["upscale"])
                    if "upscale" in forms_by_type
                    else None,
                    caption=forms_by_type["caption"]["caption"]
                    if "caption" in forms_by_type
                    else None,
                    nsfw=forms_by_type["nsfw"]["nsfw"]
                    if "nsfw" in forms_by_type
                    else None,
                    interrogation=InterrogationDetails(
                        **forms_by_type["interrogation"]["interrogation"]
                    )
                    if "interrogation" in forms_by_type
                    else None,
                )
            elif status_data["state"] == "faulted":
                raise APIError("Request faulted")

            await asyncio.sleep(1)

        # Cancel on timeout
        await request(session.delete, url_status, headers)
        raise APIError("Timeout")
