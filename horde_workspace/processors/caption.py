from functools import cache

from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


@cache
def get_model():
    return AutoModelForCausalLM.from_pretrained(
        "MiaoshouAI/Florence-2-base-PromptGen-v1.5", trust_remote_code=True
    )


@cache
def get_processor():
    return AutoProcessor.from_pretrained(
        "MiaoshouAI/Florence-2-base-PromptGen-v1.5", trust_remote_code=True
    )


class CaptionType:
    GENERATE_TAGS = "<GENERATE_TAGS>"
    CAPTION = "<CAPTION>"
    DETAILED_CAPTION = "<DETAILED_CAPTION>"
    MORE_DETAILED_CAPTION = "<MORE_DETAILED_CAPTION>"
    MIXED_CAPTION = "<MIXED_CAPTION>"


def get_caption(
    image: Image.Image, caption_type: str = CaptionType.MORE_DETAILED_CAPTION
):
    model = get_model()
    processor = get_processor()

    inputs = processor(text=caption_type, images=image, return_tensors="pt")

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    caption = processor.post_process_generation(
        generated_text, task=caption_type, image_size=(image.width, image.height)
    )

    return caption[caption_type].replace("\r", " ").replace("\n", " ").strip()
