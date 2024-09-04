import base64
import io
import urllib.parse
from typing import TypeVar

import aiohttp
import requests
from PIL import Image


def get(url) -> dict:
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(f"Failed to resolve model: {response.text}")


async def download_image(
    aiohttp_session: aiohttp.ClientSession, url: str
) -> bytes | None:
    """Asynchronously convert from base64 or download an image from a response."""
    if urllib.parse.urlparse(url).scheme in {"http", "https"}:
        async with aiohttp_session.get(url) as response:
            if response.status != 200:
                response.raise_for_status()

            return await response.read()
    else:
        return base64.b64decode(url)


def b64_encode_image(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="webp")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class GenerationError(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason

    def __str__(self) -> str:
        return self.reason


T = TypeVar("T")


def assert_none(value: T | None) -> T:
    if value is None:
        raise GenerationError("No value available")
    return value
