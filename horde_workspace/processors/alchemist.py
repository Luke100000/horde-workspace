import asyncio
import io

import aiohttp
from PIL import Image
from attr import dataclass
from horde_sdk import RequestErrorResponse
from horde_sdk.ai_horde_api import (
    AIHordeAPIAsyncClientSession,
    AIHordeAPIAsyncSimpleClient,
)
from horde_sdk.ai_horde_api.apimodels import (
    AlchemyStatusResponse,
    AlchemyAsyncRequest,
    AlchemyAsyncRequestFormItem,
    AlchemyInterrogationDetails,
)
from horde_sdk.ai_horde_api.consts import KNOWN_ALCHEMY_TYPES

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


@dataclass
class AlchemyGeneration:
    image: bytes | None = None
    caption: str | None = None
    nsfw: bool | None = None
    interrogation: AlchemyInterrogationDetails | None = None

    def get_image(self) -> Image.Image:
        if self.image is None:
            raise GenerationError("No image available")
        return Image.open(io.BytesIO(self.image))


def caption(ws: Workspace, image: Image.Image) -> str:
    return assert_none(alchemist(ws, image, [KNOWN_ALCHEMY_TYPES.caption]).caption)


def interrogation(ws: Workspace, image: Image.Image) -> AlchemyInterrogationDetails:
    return assert_none(
        alchemist(ws, image, [KNOWN_ALCHEMY_TYPES.interrogation]).interrogation
    )


def nsfw(ws: Workspace, image: Image.Image) -> bool:
    return assert_none(alchemist(ws, image, [KNOWN_ALCHEMY_TYPES.nsfw]).nsfw)


def upscale(ws: Workspace, image: Image.Image) -> Image.Image:
    return alchemist(ws, image, [KNOWN_ALCHEMY_TYPES.NMKD_Siax]).get_image()


def alchemist(
    ws: Workspace, image: Image.Image, forms: list[KNOWN_ALCHEMY_TYPES]
) -> AlchemyGeneration:
    return asyncio.run(async_alchemist(ws, image, forms))


async def async_alchemist(
    ws: Workspace, image: Image.Image, forms: list[KNOWN_ALCHEMY_TYPES]
) -> AlchemyGeneration:
    aiohttp_session = aiohttp.ClientSession()
    horde_client_session = AIHordeAPIAsyncClientSession(aiohttp_session)

    async with aiohttp_session, horde_client_session:
        client = AIHordeAPIAsyncSimpleClient(
            aiohttp_session=aiohttp_session,
            horde_client_session=horde_client_session,
        )

        response: AlchemyStatusResponse
        response, _ = await client.alchemy_request(
            AlchemyAsyncRequest(
                apikey=ws.apikey,
                slow_workers=ws.slow_workers,
                source_image=b64_encode_image(image),
                forms=[AlchemyAsyncRequestFormItem(name=name) for name in forms],
            ),
        )

        if isinstance(response, RequestErrorResponse):
            raise GenerationError(response.message)

        # noinspection PyTypeChecker
        return AlchemyGeneration(
            image=None
            if len(response.all_upscale_results) == 0
            else await download_image(
                aiohttp_session, response.all_upscale_results[0].url
            ),
            caption=None
            if len(response.all_caption_results) == 0
            else response.all_caption_results[0].caption,
            nsfw=None
            if len(response.all_nsfw_results) == 0
            else response.all_nsfw_results[0].nsfw,
            interrogation=None
            if len(response.all_interrogation_results) == 0
            else response.all_interrogation_results[0],
        )
    return AlchemyGeneration()
