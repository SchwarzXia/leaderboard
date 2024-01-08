import torch, gc, tyro
import numpy as np
from diffusers import DiffusionPipeline, AutoPipelineForText2Image
from zeus.monitor import ZeusMonitor
from utils import get_logger, CsvHandler
from metrics import *

# default parameters
DEVICE = "cuda"
WEIGHT_DTYPE = torch.float16
SEED = 0
OUTPUT_FILE = "results.csv"


def get_pipeline(model, pipeline=DiffusionPipeline, device=DEVICE, weight_dtype=WEIGHT_DTYPE):
    try:
        return AutoPipelineForText2Image.from_pretrained(model, torch_dtype=weight_dtype, safety_checker = None).to(device)
    except:
        return pipeline.from_pretrained(model, torch_dtype=weight_dtype, safety_checker = None).to(device)
    
def gpu_warmup():
    logger = get_logger()
    logger.info("Warming up GPU")
    generator = torch.manual_seed(2)
    pipeline = get_pipeline("runwayml/stable-diffusion-v1-5")
    prompts = load_prompts_clip(10)
    _ = pipeline(prompts, num_images_per_prompt=10, generator=generator, output_type="numpy").images




def benchmark(
        model: str,
        prompt_size: int = 0,
        batch_size: int = 1,
        output_file: str = OUTPUT_FILE,
        device: str = DEVICE,
        seed: int = SEED, 
        weight_dtype: torch.dtype = WEIGHT_DTYPE,
        write_header: bool = False,
        warmup: bool = False,
        settings: dict = {}
) -> None:
    """
        Main benchmark script.
    """
    logger = get_logger()
    logger.info("Running benchmark for model: " + model)

    csv_handler = CsvHandler(output_file)
    if write_header:
        csv_handler.write_header(
            ["model", "GPU", "image_num", "batch_size", "clip_score", "avg_latency(s)", "avg_energy(J)", "peak_memory(GB)"])

    prompts, batched_prompts = load_prompts_clip(prompt_size, batch_size, seed)
    logger.info("Loaded prompts")

    if warmup:
        gpu_warmup()

    generator = torch.manual_seed(seed)
    monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])
    pipeline = get_pipeline(model, device=device, weight_dtype=weight_dtype)

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device=device)

    monitor.begin_window("generate")
    images = []
    for batch in batched_prompts:
        image = pipeline(batch, generator=generator, output_type="np", **settings).images
        images.append(image)
    images = np.concatenate(images)
    result_monitor = monitor.end_window("generate")

    peak_memory = torch.cuda.max_memory_allocated(device=device)

    clip_score = calculate_clip_score(images, prompts)

    result = {
        'model' : model,
        'GPU' : torch.cuda.get_device_name(torch.cuda.current_device()),
        'image_num' : prompt_size,
        'batch_size' : batch_size,
        'clip_score' : clip_score,
        'avg_latency' : round(result_monitor.time/prompt_size, 2),
        'avg_energy' : round(result_monitor.total_energy/prompt_size, 2),
        'peak_memory' : round(peak_memory/1024/1024/1024, 4)
    }

    logger.info("Results for model " + model + ":")
    logger.info(result)

    csv_handler.write_results(result)

    logger.info("Finished benchmarking for " + model)


if __name__ == "__main__":
    tyro.cli(benchmark)