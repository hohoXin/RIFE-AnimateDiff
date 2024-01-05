import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf

import torch
from torch.profiler import profile, record_function, ProfilerActivity

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.util import load_weights
from diffusers.utils.import_utils import is_xformers_available

from rife.rife_module import Interpolation

from einops import rearrange, repeat

import csv, pdb, glob
import math
from pathlib import Path


def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/{Path(args.config).stem}-{time_str}"
    os.makedirs(savedir)

    config  = OmegaConf.load(args.config)
    samples = []
    
    sample_idx = 0
    for model_idx, (config_key, model_config) in enumerate(list(config.items())):
        
        motion_modules = model_config.motion_module
        motion_modules = [motion_modules] if isinstance(motion_modules, str) else list(motion_modules)
        for motion_module in motion_modules:
            inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))

            ### >>> create validation pipeline >>> ###
            tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
            vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")            
            unet         = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))

            if is_xformers_available(): unet.enable_xformers_memory_efficient_attention()
            else: assert False

            pipeline = AnimationPipeline(
                vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            ).to("cuda")

            pipeline = load_weights(
                pipeline,
                # motion module
                motion_module_path         = motion_module,
                motion_module_lora_configs = model_config.get("motion_module_lora_configs", []),
                # image layers
                dreambooth_model_path      = model_config.get("dreambooth_path", ""),
                lora_model_path            = model_config.get("lora_model_path", ""),
                lora_alpha                 = model_config.get("lora_alpha", 0.8),
            ).to("cuda")

            prompts      = model_config.prompt
            n_prompts    = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt
            
            random_seeds = model_config.get("seed", [-1])
            random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
            random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
            
            config[config_key].random_seed = []

            enable_profiling = model_config.get("profiler", False)

            if enable_profiling:
                # Start profiling
                prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                            record_shapes=True, profile_memory=True)
                prof.__enter__()  # Manually start the profiler

            for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):
                
                # manually set random seed for reproduction
                if random_seed != -1: torch.manual_seed(random_seed)
                else: torch.seed()
                config[config_key].random_seed.append(torch.initial_seed())
                
                print(f"current seed: {torch.initial_seed()}")
                print(f"sampling {prompt} ...")             

                with record_function(f"animatediff_{prompt_idx}"):
                    sample = pipeline(
                        prompt,
                        negative_prompt     = n_prompt,
                        num_inference_steps = model_config.steps,
                        guidance_scale      = model_config.guidance_scale,
                        width               = args.W,
                        height              = args.H,
                        video_length        = args.L,
                    ).videos
                
                VFI_flag = model_config.get("VFI_flag", False)

                with record_function(f"rife_{prompt_idx}"):
                    if VFI_flag == True:
                        # use VFI, VFI_num is the multiplier factor, e.g. VFI_num = 2 means 2x frames
                        VFI_num = model_config.get("VFI_num", 2)
                        # VFI plugin TO DO
                        VFI = Interpolation(v_tensor=sample, multi=VFI_num)
                        sample = VFI.start_processing()
                    else:
                        pass
                
                samples.append(sample)
                
                fps = model_config.get("fps", 8)
                prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
                save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt}.gif", fps=fps)
                print(f"save to {savedir}/sample/{prompt}.gif")
                # torch.save(sample, f"{savedir}/sample/tensor_{sample_idx}.pt")
                
                sample_idx += 1

            # quit the profiler
            if enable_profiling:
                # Stop profiling
                prof.__exit__(None, None, None)
                # processing the profiling data
                events = prof.key_averages()
                # Filtering events for those that match your custom record_function names
                filtered_events = [event for event in events if "animatediff_" in event.key or "rife_" in event.key]

                # Formatting the filtered events
                print("event.key", "event.cpu_time_total", "event.cuda_time_total", "event.cpu_memory_usage", "event.cuda_memory_usage")
                for event in filtered_events:        
                    print(f"{event.key:<30} {event.cpu_time_total:<15} {event.cuda_time_total:<15} {event.cpu_memory_usage:<15} {event.cuda_memory_usage}")

                print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10, top_level_events_only=True))
                print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10, top_level_events_only=True))


    samples = torch.concat(samples)
    save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4, fps=fps)
    OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="/home/ubuntu/AnimateDiff/models/StableDiffusion",)
    parser.add_argument("--inference_config",      type=str, default="/home/ubuntu/AnimateDiff/configs/inference/inference-v2.yaml")    
    parser.add_argument("--config",                type=str, required=True)
    
    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)

    args = parser.parse_args()
    main(args)
