# AnimateDiff RIFE Plugin

This repository is an extension of the official implementation of [AnimateDiff](https://arxiv.org/abs/2307.04725). It adds a video frame interpolation module as a plugin. The plugin is based on [RIFE](https://github.com/hzwer/arXiv2020-RIFE).

## Frame Interpolation

<table class="center">
    <tr style="line-height: 0">
        <td width=20% style="border: none; text-align: center">Original AnimateDiff</td>
        <td width=80% style="border: none"><img src="__assets__/rife/toonyou-original.gif" style="width:100%"></td>        
    </tr>
    <tr>
        <td width=20% style="border: none; text-align: center">AnimateDiff with RIFE</td>
        <td width=80% style="border: none"><img src="__assets__/rife/toonyou-rife.gif" style="width:100%"></td>
    </tr>
</table>

The frame interporlation module is using a light-weight Neural Network to generate a higher fps video. The fps can be adjusted by the user.

## New Features

This plugin adds the following new features to AnimateDiff:

1. **VFI-RIFE**: VFI stands for the video frame interpolation. Based on the original inference result, the RIFE model will guess the interpolation frames. Finally, the plugin will combine the original frames and the interpolation frames to generate a higher fps video.
2. **fps adjustment**: Add a fps control to the original inference task.
3. **pytorch profiler**: Add a method to use pytorch profiler to monitor the performance metrics during the inference.

## Installation

To install the plugin, follow these steps:

### 1.Prepare Environment:

```bash
git clone https://github.com/hohoXin/RIFE-AnimateDiff.git
cd AnimateDiff

conda env create -f environment.yaml
conda activate animatediff
```

### 2. Download Base T2I & Motion Module Checkpoints:
We provide two versions of our Motion Module, which are trained on stable-diffusion-v1-4 and finetuned on v1-5 seperately.
It's recommanded to try both of them for best results.

```bash
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 models/StableDiffusion/

bash download_bashscripts/0-MotionModule.sh
```
You may also directly download the motion module checkpoints from [Google Drive](https://drive.google.com/drive/folders/1EqLC65eR1-W-sGD0Im7fkED6c8GkiNFI?usp=sharing) / [HuggingFace](https://huggingface.co/guoyww/animatediff) / [CivitAI](https://civitai.com/models/108836/animatediff-motion-modules), then put them in `models/Motion_Module/` folder.

### 3. Prepare Personalize T2I
Here we provide inference configs for 6 demo T2I on CivitAI.
You may run the following bash scripts to download these checkpoints.
```bash
bash download_bashscripts/1-ToonYou.sh
bash download_bashscripts/2-Lyriel.sh
bash download_bashscripts/3-RcnzCartoon.sh
bash download_bashscripts/4-MajicMix.sh
bash download_bashscripts/5-RealisticVision.sh
bash download_bashscripts/6-Tusun.sh
bash download_bashscripts/7-FilmVelvia.sh
bash download_bashscripts/8-GhibliBackground.sh
```

### 4. Download RIFE Model:
Download the latest RIFE model provided by repository [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE.git)

4.13.1 - 2023.12.05 | [Google Drive](https://drive.google.com/file/d/1mj9lH6Be7ztYtHAr1xUUGT3hRtWJBy_5/view?usp=sharing) | [百度网盘](https://pan.baidu.com/s/1e0I-ERSYQThANP7BQmz3Vw?pwd=e2h8)

4.13.**lite** - 2023.11.27 | [Google Drive](https://drive.google.com/file/d/1l3lH9QxQQeZVWtBpdB22jgJ-0kmGvXra/view?usp=sharing) | [百度网盘](https://pan.baidu.com/s/12arDG3wAG9soGBmzUkkbeQ?pwd=2fy8) || v4.12.2 - 2023.11.13 | [Google Drive](https://drive.google.com/file/d/1ZHrOBL217ItwdpUBcBtRE3XBD-yy-g2S/view?usp=share_link) | [百度网盘](https://pan.baidu.com/s/1zyAw-qZJsIsAyFOIZKumYQ?pwd=gwij) 

Download a model from the model list and put *.py and flownet.pkl on rife/train_log/

### 5. Inferece
After downloading the above peronalized T2I checkpoints, run the following commands to generate animations. The results will automatically be saved to `samples/` folder.
```bash
python -m scripts.RIFE-animate --config configs/prompts/v2-RIFE-ToonYou-test.yaml
```

To generate animations with a new DreamBooth/LoRA model, you may create a new config `.yaml` file in the following format:
```
NewModel:
  inference_config: "[path to motion module config file]"

  motion_module:
    - "models/Motion_Module/mm_sd_v14.ckpt"
    - "models/Motion_Module/mm_sd_v15.ckpt"
    
    motion_module_lora_configs:
    - path:  "[path to MotionLoRA model]"
      alpha: 1.0
    - ...

  dreambooth_path: "[path to your DreamBooth model .safetensors file]"
  lora_model_path: "[path to your LoRA model .safetensors file, leave it empty string if not needed]"

  seed:           114514
  steps:          25
  guidance_scale: 7.5

  VFI_flag: True
  VFI_num: 3

  fps: 24

  profiler: True

  prompt:
    - "[positive prompt]"

  n_prompt:
    - "[negative prompt]"
```
Then run the following commands:
```
python -m scripts.RIFE-animate --config [path to the config file]
```

### ORIGINAL README
Please refer to the [original README](__assets__/docs/ORIGINAL_README.md) for more details.

## Citation
official animatediff repo: [AnimateDiff](https://github.com/guoyww/AnimateDiff.git)

official RIFE repo: [RIFE](https://github.com/hzwer/Practical-RIFE.git)