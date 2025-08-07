import pathlib

import torch

# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from scripts.safety_checker import StableDiffusionSafetyChecker

from transformers import AutoFeatureExtractor
from PIL import Image
import numpy as np
from modules import scripts, shared

safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = None
safety_checker = None
adjustment = -0.01


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


# check and replace nsfw content
def check_safety(x_image, safety_checker_adj: float = -0.01):
    global safety_feature_extractor, safety_checker

    if safety_feature_extractor is None:
        safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

    safety_checker_input = safety_feature_extractor(
        numpy_to_pil(x_image), return_tensors="pt"
    )
    x_checked_image, has_nsfw_concept = safety_checker(
        images=x_image,
        clip_input=safety_checker_input.pixel_values,
        safety_checker_adj=safety_checker_adj,
    )
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def censor_batch(x):
    x_samples_ddim_numpy = x.cpu().permute(0, 2, 3, 1).numpy()
    x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim_numpy, safety_checker_adj=adjustment)
    x = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

    return x


class NsfwCheckScript(scripts.Script):
    def title(self):
        return "NSFW check"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def postprocess_batch(self, p, *args, **kwargs):
        images = kwargs["images"]
        images[:] = censor_batch(images)[:]

def load_replacement(x):
    try:
        hwc = x.shape
        image_path = pathlib.Path(__file__).absolute().parent.joinpath("NSFW_replace.png")
        y = Image.open(image_path).convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception as e:
        return x
