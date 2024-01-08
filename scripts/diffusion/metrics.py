from datasets import load_dataset
import torch
from torchmetrics.functional.multimodal import clip_score
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import functional as F
from torchvision import transforms
from functools import partial
import random
from utils import get_logger

def load_prompts_clip(size, batch_size, seed):
    """Generate prompts for CLIP Score metric"""
    prompts = load_dataset("nateraw/parti-prompts", split='train')
    if size == 0:
        size = len(prompts)
    else:
        random.seed(seed)
        prompts = prompts.shuffle()
    prompts = [prompts[i]["Prompt"] for i in range(size)]
    return prompts, [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]

def load_prompts_fid(size):
    """Generate prompts for FID metric"""
    prompts = load_dataset("nlphuji/mscoco_2014_5k_test_image_text_retrieval", split='test')
    prompts = prompts.shuffle()
    return ([random.choice(prompts[i]["caption"]) for i in range(size)], [prompts[i]["image"] for i in range(size)])

def calculate_clip_score(images, prompts):
    """Calculate CLIP Score metric, assumes images and prompts have same size"""
    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
    images_int = (images * 255).astype("uint8")
    clip = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip), 2)

def calculate_fid_score(gen_images, real_images):
    """Calculate FID Score metric"""
    logger = get_logger()

    gen_images = torch.tensor(gen_images).permute(0, 3, 1, 2)  
    logger.info(gen_images.shape) 
    image_shape = (gen_images.shape[2], gen_images.shape[3])

    def preprocess_image(image):
        totensor = transforms.ToTensor()
        image = totensor(image).unsqueeze(0)
        image = F.center_crop(image, image_shape) / 255.0
        return image
    real_images = torch.cat([preprocess_image(image) for image in real_images])

    fid = FrechetInceptionDistance(normalize=True)
    logger.info(real_images.shape)
    fid.update(real_images, real=True)
    fid.update(gen_images, real=False)
    return round(float(fid.compute()), 2)